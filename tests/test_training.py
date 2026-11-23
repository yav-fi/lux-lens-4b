import asyncio
import json
import re
import sys

import pytest

from lux_frontier.grpo import Completion, rl_arrays
from lux_frontier.tinker_backend import AuthenticationUnavailable
from lux_frontier.training import Settings, preview, run_live, workload_estimate


class FakeBackend:
    """Simulates SDK boundaries only; rewards still use the real contract/grader."""

    def __init__(self, *, constant=False, fail_update=False, missing_auth=False):
        self.constant, self.fail_update, self.missing_auth = constant, fail_update, missing_auth
        self.events = []
        self.version = self.sampler_version = 0
        self.closed = False

    async def initialize(self, training):
        self.events.append(("initialize", training))
        if self.missing_auth:
            raise AuthenticationUnavailable()
        return {"mock_backend": True}

    def encode(self, text):
        self.ids = [int(i) for i in re.findall(r"Candidate ID (\d+):", text)]
        return [11, 12, 13]

    async def sample(self, tokens, count, seed):
        self.events.append(("sample", self.sampler_version, count, seed))
        results = []
        for i in range(count):
            index = 0 if self.constant else i % 7
            raw = json.dumps({"starter_ids": self.ids[index:index + 2]})
            results.append(Completion(raw, raw, [31, 32], [-0.1, -0.2], "stop", True))
        return results

    async def update(self, examples):
        assert self.version == self.sampler_version  # on-policy, never stale
        assert any(a for _, _, a in examples)
        for tokens, completion, advantage in examples:
            rl_arrays(tokens, completion, advantage)
        self.events.append(("update", self.version))
        if self.fail_update:
            raise RuntimeError("fake provider failure; should not be serialized")
        self.version += 1
        return {"forward_backward": {"loss:sum": 1.2}, "optimizer": {"mock_update": 1}}

    async def refresh_sampler(self):
        self.sampler_version = self.version
        self.events.append(("refresh", self.sampler_version))

    async def save_checkpoint(self):
        self.events.append(("save", self.version))
        return {"training_state": "mock://training", "sampler_weights": "mock://sampling"}

    async def reload_checkpoint(self, checkpoint):
        assert checkpoint["sampler_weights"] == "mock://sampling"
        self.events.append(("reload", self.version))

    async def close(self, status):
        self.closed = True


def test_default_workload_is_bounded_and_estimate_is_not_provider_limit():
    settings = Settings()
    assert (settings.lora_rank, settings.steps, settings.briefs_per_step,
            settings.samples_per_brief, settings.max_tokens) == (16, 3, 4, 8, 128)
    estimate = workload_estimate(settings, "train")
    assert estimate["maximum_sampled_completions"] == 113
    assert 0 < estimate["estimated_upper_cost_usd"] < 5
    assert estimate["provider_enforced_spending_limit"] is False
    with pytest.raises(ValueError, match="exceeds"):
        workload_estimate(Settings(application_budget_usd=0.01), "train")
    for values in [dict(steps=4), dict(briefs_per_step=5), dict(samples_per_brief=9), dict(max_tokens=129),
                   dict(max_prompt_tokens=4097), dict(application_budget_usd=6)]:
        with pytest.raises(ValueError):
            Settings(**values)


def test_offline_preview_validates_without_a_backend_or_credentials(tmp_path, monkeypatch):
    monkeypatch.delenv("TINKER_API_KEY", raising=False)
    monkeypatch.setitem(sys.modules, "tinker", None)
    monkeypatch.setitem(sys.modules, "tinker_cookbook", None)
    record = preview(Settings(), tmp_path)
    data = json.loads(record.path.read_text())
    assert data["synthetic"] and data["status"] == "offline_validated"
    assert len(data["cases"]) == data["usable_groups"] == 8
    assert data["token_usage"] == dict(sampling_prefill_tokens=0, generated_tokens=0, training_tokens_submitted=0)
    assert data["optimizer_updates_completed"] == 0 and not data["smoke_success"]
    assert "pydantic" in data["dependency_versions"]
    assert all(g["completions"][-1]["reward"] == 0 for g in data["groups"])


def test_mock_training_flow_updates_refreshes_evaluates_and_reloads(tmp_path):
    backend = FakeBackend()
    record = asyncio.run(run_live(Settings(), "train", tmp_path, backend=backend))
    data = json.loads(record.path.read_text())
    assert data["backend"]["mock_backend"]  # this test is never evidence of real training
    assert backend.closed and "cleanup_error_type" not in data
    assert data["status"] == "completed" and data["smoke_success"]
    assert data["optimizer_updates_completed"] == 3
    assert data["usable_groups"] == 12
    assert data["checkpoint_reload_sampled"]
    initial, final = data["evaluations"]["initial"], data["evaluations"]["final"]
    assert initial["brief_ids"] == final["brief_ids"]
    assert initial["mean_reward"] == final["mean_reward"]  # improvement is not required
    assert len(data["cases"]) == 16  # twelve train, four held out
    assert [e for e in backend.events if e[0] == "refresh"] == [("refresh", 1), ("refresh", 2), ("refresh", 3)]
    phases = {g["phase"]: g for g in data["groups"]}
    assert phases["initial"]["sampling_seed"] == phases["final"]["sampling_seed"]
    assert backend.events[-2][0] == "reload" and backend.events[-1][0] == "sample"
    assert data["token_usage"]["generated_tokens"] == 226
    assert data["token_usage"]["training_tokens_submitted"] == 384
    assert "forward_backward" in data["steps"][0]["metrics"]
    assert all("raw_response" in c and "parsed_choices" in c and "reward" in c
               for g in data["groups"] for c in g["completions"])


def test_constant_groups_do_not_update_or_claim_success(tmp_path):
    backend = FakeBackend(constant=True)
    record = asyncio.run(run_live(Settings(steps=1), "train", tmp_path, backend=backend))
    assert record.data["status"] == "no_learning_signal"
    assert record.data["usable_groups"] == record.data["optimizer_updates_completed"] == 0
    assert record.data["checkpoint_reload_sampled"] and not record.data["smoke_success"]
    assert record.data["steps"][0]["constant_reward_groups"] == 4
    assert not any(e[0] == "update" for e in backend.events)


def test_failed_update_is_not_counted_and_raw_responses_survive(tmp_path):
    record = asyncio.run(run_live(Settings(steps=1), "train", tmp_path, backend=FakeBackend(fail_update=True)))
    assert record.data["status"] == "failed"
    assert record.data["error_type"] == "RuntimeError"
    assert not record.data["steps"][0]["optimizer_update_completed"]
    assert record.data["optimizer_updates_completed"] == 0
    assert record.data["groups"][-1]["completions"][0]["raw_response"]
    assert "fake provider failure" not in record.path.read_text()


def test_missing_auth_creates_inspectable_record_without_sampling(tmp_path):
    backend = FakeBackend(missing_auth=True)
    record = asyncio.run(run_live(Settings(), "train", tmp_path, backend=backend))
    assert record.data["status"] == "authentication_unavailable"
    assert record.data["groups"] == [] and record.data["estimated_cost_usd"] == 0
    assert record.path.is_file()


def test_sample_mode_never_updates(tmp_path):
    backend = FakeBackend()
    record = asyncio.run(run_live(Settings(), "sample", tmp_path, backend=backend))
    assert record.data["status"] == "sampled"
    assert len(record.data["groups"]) == 1
    assert record.data["optimizer_updates_completed"] == 0
    assert not record.data["smoke_success"]


def test_grader_failure_keeps_raw_response_without_claiming_update(tmp_path):
    def broken_grader(*_):
        raise RuntimeError("broken grader")
    record = asyncio.run(run_live(Settings(), "sample", tmp_path, backend=FakeBackend(), grader=broken_grader))
    assert record.data["status"] == "failed"
    assert record.data["groups"][0]["completions"][0]["raw_response"]
    assert record.data["optimizer_updates_completed"] == 0


def test_billing_rejection_records_status_without_server_request_details(tmp_path):
    class BillingError(RuntimeError):
        status_code = 402

    class UnfundedBackend(FakeBackend):
        async def initialize(self, training):
            raise BillingError("sensitive server request details")

    record = asyncio.run(run_live(Settings(), "train", tmp_path, backend=UnfundedBackend()))
    assert record.data["status"] == "billing_unavailable"
    assert record.data["provider_status_code"] == 402
    assert record.data["optimizer_updates_completed"] == 0
    assert "sensitive server request details" not in record.path.read_text()

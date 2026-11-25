"""Regression coverage for the V2 grading, optics and training boundaries."""

import asyncio
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

from lux_frontier.grpo import Completion, rl_arrays
from lux_frontier.v2.artifacts import export_harbor, snapshot
from lux_frontier.v2.benchmark import Protocol, refine, render, requirements, validate_splits
from lux_frontier.v2.evaluation import evaluate
from lux_frontier.v2.grading import PHASES, VERSION, grade, submit
from lux_frontier.v2.optics import (
    PROFILES, InvalidBuild, build_optic, draw, focus_position, measurements, reference,
)
from lux_frontier.v2.rl import Settings, assess_group, inference, train
from lux_frontier.v2.validation import compare_sensor_traces


@pytest.fixture(scope="module")
def optical():
    p = reference("CookeTriplet")
    pr = PROFILES["standard-adjusted"]
    brief = requirements(p, pr, (1, 0., .022, 0., .13, .025, .68), "optical-example")
    initial = measurements(p, brief, 16)
    return p, pr, brief, initial


@pytest.fixture
def task(optical):
    p, pr, brief, initial = optical
    candidates = [dict(id=k, identity=f"design-{k}", prescription=p.model_dump(mode="json"), initial=initial,
                       sensitivity={"focus": None}) for k in range(1, 9)]
    return dict(version=VERSION, task_id="optical-example", brief=brief.model_dump(mode="json"), profile=pr.record(),
                protocol=Protocol(evaluations=3, pupil_samples=16).model_dump(mode="json"),
                yield_target=.75, candidates=candidates, split="train", group="one", cohort="one",
                construction={"families": ["A", "E"]}, capability=True,
                outcomes={str(k): dict(nominal_pass=k in (1, 2, 3), near_miss=.49,
                    simulations={phase: dict(passed=(8 if k == 1 else 4 if k == 2 else 0), tested=8)
                                 for phase in PHASES} if k in (1, 2, 3) else {})
                          for k in range(1, 9)})


@pytest.mark.parametrize("response", [
    '["starter_id",1]', '{"starter_id":true}', '{"starter_id":1.0}', '{"starter_id":"1"}',
    '{"starter_ids":[1,2]}', '{"starter_id":1,"starter_id":2}', '{"starter_id":1,"other":2}',
    '```json\n{"starter_id":1}\n```', '{"starter_id":1} extra', '{"starter_id":-1}', '{"starter_id":19}',
])
def test_invalid_responses_are_never_repaired(task, response):
    result = grade(task, response)
    assert result.starter_id is None and result.error and not result.reward


def test_nominal_and_yield_rewards_are_ordered_and_truncation_is_zero(task):
    rewards = [grade(task, submit(k)).reward for k in range(1, 5)]
    assert rewards == sorted(rewards, reverse=True)
    assert rewards[2] > rewards[3] > 0
    assert grade(task, submit(1)).final_pass
    assert grade(task, submit(1), clean=False).reward == 0
    task["outcomes"]["1"]["simulations"] = {}
    nominal = grade(task, submit(1), mode="nominal")
    assert nominal.reward == 1 and nominal.yield_fraction == 0
    with pytest.raises(ValueError, match="phase is unavailable"):
        grade(task, submit(1))


def test_prompt_hides_optical_results_and_has_no_anchored_answer(task):
    task["outcomes"] = {"secret": "DO NOT EXPOSE"}
    for representation in ("summary", "full", "physics"):
        prompt = render(task, representation)
        assert '"starter_id"' in prompt and "DO NOT EXPOSE" not in prompt and '{"starter_id":' not in prompt
        assert "Candidate 4 |" in prompt
        assert ("signed radii" in prompt) == (representation != "summary")
        assert ("Probe changes" in prompt) == (representation == "physics")


def test_leakage_rejects_disjoint_local_ids_with_shared_lineage(task):
    other = dict(task, task_id="different", split="test", group="new", cohort="new")
    other["candidates"] = [dict(c, id=c["id"]+10, identity="other"+c["identity"]) for c in task["candidates"]]
    with pytest.raises(ValueError, match="split leakage"):
        validate_splits([task, other])


def test_refinement_architecture_and_budget_and_independent_real_rays(optical):
    p, _, brief, _ = optical
    protocol = Protocol(evaluations=7, pupil_samples=16)
    result = refine(p, brief, protocol)
    assert len(result["trajectory"]) == protocol.evaluations
    chosen = result["final"]["prescription"]
    assert tuple(map(tuple, chosen["glasses"])) == p.glasses
    assert chosen["stop_surface"] == p.stop_surface
    assert chosen["stop_diameter_mm"] == p.stop_diameter_mm
    assert abs(chosen["sensor_z_mm"]-p.sensor_z_mm) <= .5
    assert len(compare_sensor_traces(p, build_optic(p)[0])) == 3
    with pytest.raises(InvalidBuild, match="nonpositive"):
        build_optic(p.model_copy(update={"sensor_z_mm": 1.}))


def test_assembly_focus_is_bounded_and_agrees_with_an_explicit_rebuild(optical):
    p, profile, brief, _ = optical
    errors = draw(p, profile, [17291, 123, 4, 1])
    lens, geometry = build_optic(p, errors)
    shift, _ = focus_position(lens, p, profile)
    assert abs(shift) <= profile.focus_travel_mm
    assert shift / profile.focus_increment_mm == pytest.approx(round(shift / profile.focus_increment_mm))
    lens.image_surface.geometry.cs.z = p.sensor_z_mm+shift
    measured = measurements(p, brief, 16, optic=(lens, geometry), focus=shift)
    rebuilt = measurements(p, brief, 16, errors=errors, focus=shift)
    assert measured["metrics"] == rebuilt["metrics"]
    assert measured["throughput"] == rebuilt["throughput"]
    assert errors == draw(p, profile, [17291, 123, 4, 1])
    assert errors != draw(p, profile, [17291, 123, 3, 1])


def test_replay_preserves_real_sensor_coordinates_and_calibrated_mtf(optical):
    p, _, brief, _ = optical
    content = snapshot(p, brief, 16, curves=True)
    assert len(content["surface_geometry"]) == len(p.radii_mm)
    for trace in content["traces"]:
        intersections = np.asarray([point for point in trace["sensor_intersections_mm"] if point is not None])
        assert intersections.size > 0
        assert np.allclose(intersections[:, 2], p.sensor_z_mm)
        assert trace["sensor_intersections_mm"] == trace["surface_nodes_mm"][-1]
    for row in content["measured_with_curves"]["trace"]["field_mapping"]:
        assert np.all(np.diff(row["curve"]["frequency_lp_per_mm"]) > 0)
        assert row["curve"]["mtf"][0] == pytest.approx(1)


def completion(action, clean=True):
    answer = submit(action)
    return Completion(answer, answer, [int(action), 99], [-.3, -.2], "stop", clean)


def test_group_detects_malformed_only_and_ignores_invalid_samples(task):
    bad = completion(1, False)
    _, adv, stats = assess_group(task, [completion(1), bad], "training", "yield")
    assert stats["malformed_only"] and not stats["informative"] and adv == [0., 0.]
    _, adv, stats = assess_group(task, [completion(1), completion(3), bad], "training", "yield")
    assert stats["informative"] and stats["unique_valid_actions"] == 2 and stats["invalid"] == 1
    assert adv == pytest.approx([.475, -.475, 0.])


class LocalBackend:
    def __init__(self, settings, fail_update=None):
        self.settings = settings
        self.weights = 0
        self.calls = 0
        self.fail_update = fail_update
        self.updates = 0
        self.resumed = False

    async def initialize(self, training, resume=None):
        assert training
        assert resume is None or "training_state" in resume
        if resume:
            self.resumed = True
            self.weights = resume["step"]

    def encode(self, prompt):
        assert "starter_id" in prompt
        return [42, 40]

    async def sample(self, prompt, count, seed):
        self.calls += 1
        if count == 1:
            return [completion(1)]
        if self.calls == 2:
            return [completion(1)]*count
        if self.calls == 3:
            return [completion(1), completion(3, False)]
        return [completion(1), completion(3)]

    async def probe(self, prompt, response):
        return [-.6+.05*self.weights, -.2]

    async def update(self, examples):
        self.updates += 1
        if self.updates == self.fail_update:
            raise RuntimeError("simulated remote failure")
        for args in examples:
            arrays = rl_arrays(*args)
            assert arrays["advantages"][0] == 0
        self.weights += 1
        return {"step": self.weights}

    async def save(self, index):
        return {"training_state": f"local://{index}/state", "sampler_weights": f"local://{index}/sample", "step": index}

    async def refresh_sampler(self):
        pass

    async def reload_checkpoint(self, checkpoint):
        self.weights = checkpoint["step"]

    async def close(self, status):
        pass


def test_dynamic_sampling_update_probe_and_resumption(task, tmp_path):
    settings = Settings(steps=2, groups_per_step=1, group_size=2, minimum_groups=1,
                        max_attempts_per_step=3, sampling="batched", eval_samples=1, eval_briefs=1,
                        eval_every=2, checkpoint_every=1, curriculum=((0., {"A": 1.}),))
    dev = dict(task, split="development", task_id="dev-example")
    failed = tmp_path/"failed.json"
    backend = LocalBackend(settings, fail_update=2)
    with pytest.raises(RuntimeError, match="simulated remote failure"):
        asyncio.run(train(settings, [task], [dev], failed, backend=backend))
    record = json.loads(failed.read_text())["run"]
    assert record["status"] == "failed"
    assert record["checkpoints"][0]["step"] == 1
    assert record["steps"][0] == dict(step=1, attempted=3, accepted=1, rejected=2, updated=True, metrics={"step": 1})
    journal = [json.loads(line) for line in (tmp_path/record["group_journal"]).read_text().splitlines()]
    assert Counter(r["phase"] for r in journal)["training"] == 4
    new_backend = LocalBackend(settings)
    result = asyncio.run(train(settings, [task], [dev], tmp_path/"resume.json", backend=new_backend, resume=failed))
    assert new_backend.resumed and result["status"] == "completed"
    assert result["probes"][0]["max_abs_delta"] > 0
    assert result["checkpoint_reload_probe"]["max_abs_delta"] == 0
    assert [x["step"] for x in result["checkpoints"]] == [2]


def test_harbor_runs_the_same_verifier_and_keeps_rewards_out_of_agent_context(task, tmp_path):
    root = tmp_path/"harbor"
    export_harbor([task], root, phase="training")
    folder = root/task["task_id"]
    assert not (folder/"environment"/"cache.json").exists()
    expected = Path("lux_frontier/v2/grading.py").read_bytes()
    assert (folder/"tests"/"grading.py").read_bytes() == expected
    answer = tmp_path/"answer.json"
    reward = tmp_path/"reward.txt"
    for response in (submit(2), '{"starter_id":2,"starter_id":1}', ""):
        answer.write_text(response)
        subprocess.run(["python3", str(folder/"tests"/"grading.py"), str(folder/"tests"/"cache.json"),
                        str(answer), str(reward)], check=True)
        assert float(reward.read_text()) == pytest.approx(grade(task, response, phase="training").reward)


def test_inference_records_feed_matched_evaluation_with_exact_prompt_hash(task, tmp_path):
    class Sampler(LocalBackend):
        async def initialize(self, training, *, checkpoint):
            assert not training and checkpoint is None

        async def sample(self, prompt, count, seed):
            return [completion(2)]*count

    evaluation_task = dict(task, split="test")
    for row in evaluation_task["outcomes"].values():
        row["failures"] = [] if row["nominal_pass"] else [{"constraint": "mtf[0]"}]
    settings = Settings(eval_samples=2)
    record = asyncio.run(inference(settings, [evaluation_task], tmp_path/"sample.json", backend=Sampler(settings)))
    assert record["execution_backend"] == "offline_injected_backend"
    assert len(record["by_task_reward_samples"][task["task_id"]]["physics/separate/1.25"]) == 2
    result = evaluate([evaluation_task], phase="final", predictions={"fixture": record})
    scores = result["overall"]["methods"]["fixture"]
    assert scores["mean"]["yield_fraction"] == .5
    assert scores["mean"]["regret"] == pytest.approx(.475)
    assert scores["paired_vs_reference"]["regret"]["ci95"] is None
    record["responses"][0]["prompt_sha256"] = "different-prompt"
    with pytest.raises(ValueError, match="did not match"):
        evaluate([evaluation_task], phase="final", predictions={"fixture": record})

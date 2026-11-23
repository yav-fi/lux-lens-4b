"""Bounded SYNTHETIC selection preview, Tinker sampling, and GRPO smoke CLI.

Examples (training extra is needed only for live commands):
  python -m lux_frontier.training preview
  python -m lux_frontier.training sample
  python -m lux_frontier.training train --steps 3 --briefs-per-step 4 --samples-per-brief 8
"""

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated
from uuid import uuid4

from pydantic import Field

from lux_frontier.contracts import Contract
from lux_frontier.grpo import group_advantages
from lux_frontier.prompting import render_prompt
from lux_frontier.synthetic_training import Grader, grade_response, make_environment, synthetic_grader
from lux_frontier.tinker_backend import AuthenticationUnavailable, MODEL, RENDERER, TinkerBackend
from lux_frontier.training_tasks import FixedTaskSource, TrainingTask, load_optical_task, synthetic_task

# Conservative application estimate, NOT a provider-enforced spending limit.
# Public Qwen3.5-4B rates checked 2026-11-23: prefill .33, sample 1.005,
# training .737 (64K) / 1.34 (256K), USD per million tokens. Use the higher
# training rate, no prompt-cache discount, a 3x margin, and $.10 storage reserve.
# Source: https://thinkingmachines.ai/tinker/pricing/
RATES = {"sampling_prefill_tokens": 0.33, "generated_tokens": 1.005,
         "training_tokens_submitted": 1.34}


class Settings(Contract):
    seed: Annotated[int, Field(ge=0, le=2**31 - 10000)] = 17
    lora_rank: Annotated[int, Field(ge=1, le=64)] = 16
    steps: Annotated[int, Field(ge=1, le=3)] = 3
    briefs_per_step: Annotated[int, Field(ge=1, le=4)] = 4
    samples_per_brief: Annotated[int, Field(ge=2, le=8)] = 8
    max_tokens: Annotated[int, Field(ge=1, le=128)] = 128
    max_prompt_tokens: Annotated[int, Field(ge=1, le=4096)] = 4096
    eval_briefs: Annotated[int, Field(ge=1, le=4)] = 4
    eval_samples: Annotated[int, Field(ge=1, le=4)] = 2
    temperature: Annotated[float, Field(gt=0, le=2)] = 1
    learning_rate: Annotated[float, Field(gt=0, le=0.001)] = 1e-5
    application_budget_usd: Annotated[float, Field(gt=0, le=5)] = 5


def estimated_cost(usage: dict, *, checkpoint: bool = False) -> float:
    return round(3 * sum(usage.get(k, 0) * rate / 1e6 for k, rate in RATES.items())
                 + (0.10 if checkpoint else 0), 6)


def workload_estimate(settings: Settings, mode: str) -> dict:
    if mode == "validate":
        return dict(maximum_sampled_completions=0, token_bounds={}, estimated_upper_cost_usd=0,
                    provider_enforced_spending_limit=False, offline_only=True)
    training = settings.steps * settings.briefs_per_step * settings.samples_per_brief if mode == "train" else 0
    samples = training + 2 * settings.eval_briefs * settings.eval_samples + 1 if mode == "train" else settings.samples_per_brief
    usage = dict(sampling_prefill_tokens=samples * settings.max_prompt_tokens,
                 generated_tokens=samples * settings.max_tokens,
                 training_tokens_submitted=training * (settings.max_prompt_tokens + settings.max_tokens - 1))
    cost = estimated_cost(usage, checkpoint=mode == "train")
    if cost > settings.application_budget_usd:
        raise ValueError(f"projected application estimate ${cost:.3f} exceeds configured budget ${settings.application_budget_usd:g}")
    return dict(maximum_sampled_completions=samples, token_bounds=usage, estimated_upper_cost_usd=cost,
                rates_usd_per_million=RATES, safety_multiplier=3, storage_reserve_usd=0.10 if mode == "train" else 0,
                provider_enforced_spending_limit=False,
                exclusions="Provider billing is authoritative; estimate excludes taxes and storage after the 24-hour checkpoint TTL.")


class RunRecord:
    def __init__(self, settings: Settings, mode: str, output_root: Path):
        self.directory = output_root / f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{mode}-{uuid4().hex[:8]}"
        self.directory.mkdir(parents=True)
        self.path = self.directory / "run.json"
        self.data = dict(
            synthetic=True, mode=mode, status="running", model=MODEL, renderer=RENDERER,
            settings=settings.model_dump(),
            dependency_versions={d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
            workload=workload_estimate(settings, "train" if mode == "preview" else mode),
            token_usage=dict(sampling_prefill_tokens=0, generated_tokens=0, training_tokens_submitted=0),
            token_usage_basis="Application counts; prompt prefill conservatively counted once per completion, without caching.",
            cases={}, groups=[], steps=[], evaluations={}, usable_groups=0,
            optimizer_updates_completed=0, checkpoints={}, checkpoint_reload_sampled=False,
            smoke_success=False, estimated_cost_usd=0,
        )
        self.write()

    def write(self):
        self.data["estimated_cost_usd"] = estimated_cost(self.data["token_usage"], checkpoint=bool(self.data["checkpoints"]))
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.data, indent=2, allow_nan=False) + "\n")
        temporary.replace(self.path)

    def add_case(self, environment, text: str):
        self.data["cases"][environment.case.brief.brief_id] = dict(
            case=environment.case.model_dump(mode="json"),
            resolved_profile=environment.tolerance_profile.model_dump(mode="json"),
            prompt=text, prompt_sha256=hashlib.sha256(text.encode()).hexdigest())


def preview(settings: Settings, output_root: Path) -> RunRecord:
    record = RunRecord(settings, "preview", output_root)
    for split in ("train", "eval"):
        for index in range(settings.briefs_per_step if split == "train" else settings.eval_briefs):
            env = make_environment(settings.seed, split, index)
            text = render_prompt(env.case, env.tolerance_profile)
            record.add_case(env, text)
            ids = [c.candidate_id for c in env.case.candidates]
            # Deterministic mock responses exercise the same ingress and grading
            # functions as live sampling. These are NOT model completions.
            responses = [json.dumps({"starter_ids": [ids[i], ids[(i + 1) % 8]]}) for i in range(8)]
            responses += ['{', json.dumps({"starter_ids": [ids[0], ids[0]]}), '{"starter_ids":[999,998]}']
            grades = [asdict(grade_response(env, r)) for r in responses]
            advantages = group_advantages([g["reward"] for g in grades[:8]])
            record.data["groups"].append(dict(phase=f"preview-{split}", brief_id=env.case.brief.brief_id,
                                               mock_responses=True, completions=grades, advantages=advantages))
            record.data["usable_groups"] += int(any(advantages))
    record.data.update(status="offline_validated", preview_only=True)
    record.write()
    return record


def validate_task(task: TrainingTask, response: str, output_root: Path, settings: Settings | None = None) -> RunRecord:
    """No sampler: use the exact renderer/parser/grader dispatch used below."""
    record = RunRecord(settings or Settings(), "validate", output_root)
    record.data.update(synthetic=task.provenance == "synthetic", model=None,
                       renderer="plain-text-requirements", source_record=task.source_record,
                       backend=dict(offline_validation=True, tinker_calls=0))
    record.add_case(task.environment, task.render())
    grade = task.grade(response)
    record.data["groups"] = [dict(phase="offline-validation", brief_id=task.environment.case.brief.brief_id,
                                  completions=[asdict(grade)])]
    if hasattr(task.grader, "last_run") and task.grader.last_run:
        record.data["optical_grader_result"] = task.grader.last_run["result"]
    record.data["status"] = "offline_validated" if grade.error is None else "invalid_submission"
    record.write()
    return record


async def run_live(settings: Settings, mode: str, output_root: Path, *, backend=None,
                   grader: Grader = synthetic_grader, task_source=None) -> RunRecord:
    if mode not in ("sample", "train"):
        raise ValueError("live mode must be sample or train")
    record = RunRecord(settings, mode, output_root)
    backend = backend if backend is not None else TinkerBackend(settings)
    get_task = task_source if task_source is not None else lambda seed, split, index: synthetic_task(seed, split, index, grader)
    if isinstance(task_source, FixedTaskSource):
        record.data.update(synthetic=task_source.task.provenance == "synthetic",
                           source_record=task_source.task.source_record,
                           data_split_policy="single development brief; no held-out-data claim")

    async def sample_group(task: TrainingTask, phase: str, count: int, seed: int):
        env = task.environment
        text = task.render()
        record.add_case(env, text)
        tokens = backend.encode(text)
        if not 0 < len(tokens) <= settings.max_prompt_tokens:
            raise ValueError(f"prompt has {len(tokens)} tokens, limit {settings.max_prompt_tokens}")
        completions = await backend.sample(tokens, count, seed)
        # Record all responses before parsing or invoking a replaceable grader.
        group = dict(phase=phase, brief_id=env.case.brief.brief_id, sampling_seed=seed,
                     prompt_tokens=len(tokens), completions=[asdict(c) for c in completions])
        record.data["groups"].append(group)
        usage = record.data["token_usage"]
        usage["sampling_prefill_tokens"] += len(tokens) * max(count, len(completions))
        usage["generated_tokens"] += sum(len(c.tokens) for c in completions)
        record.write()
        if len(completions) != count:
            raise ValueError(f"provider returned {len(completions)} completions, expected {count}")
        if any(len(c.tokens) > settings.max_tokens for c in completions):
            raise ValueError("provider exceeded the generated-token workload limit")
        rewards = []
        for completion, entry in zip(completions, group["completions"], strict=True):
            # Chat-template parsing removes framing only; JSON remains untouched.
            grade = task.grade(completion.response_text, truncated=not completion.clean)
            entry.update(parsed_choices=grade.parsed_choices, reward=grade.reward, error=grade.error)
            rewards.append(grade.reward)
        group["advantages"] = group_advantages(rewards)
        record.write()
        return tokens, completions, group

    async def evaluate(phase: str, held_out: list):
        entries = []
        for index, task in enumerate(held_out):
            _, _, group = await sample_group(task, phase, settings.eval_samples, settings.seed + 5000 + index)
            entries.extend(group["completions"])
        record.data["evaluations"][phase] = dict(
            brief_ids=[t.environment.case.brief.brief_id for t in held_out], completions=len(entries),
            mean_reward=sum(e["reward"] for e in entries) / len(entries),
            valid_fraction=sum(e["error"] is None for e in entries) / len(entries))
        record.write()

    try:
        record.data["backend"] = await backend.initialize(training=mode == "train")
        if mode == "sample":
            await sample_group(get_task(settings.seed, "eval", 0), "sample", settings.samples_per_brief, settings.seed)
            record.data["status"] = "sampled"
        else:
            held_out = [get_task(settings.seed, "eval", i) for i in range(settings.eval_briefs)]
            await evaluate("initial", held_out)
            for step in range(settings.steps):
                examples, step_groups = [], []
                for index in range(settings.briefs_per_step):
                    case_index = step * settings.briefs_per_step + index
                    task = get_task(settings.seed, "train", case_index)
                    tokens, completions, group = await sample_group(
                        task, f"train-{step}", settings.samples_per_brief, settings.seed + case_index)
                    step_groups.append(group)
                    if any(group["advantages"]):
                        record.data["usable_groups"] += 1
                        examples.extend((tokens, c, a) for c, a in zip(completions, group["advantages"], strict=True) if c.tokens)
                usable = sum(any(g["advantages"]) for g in step_groups)
                step_record = dict(step=step, usable_groups=usable, constant_reward_groups=len(step_groups) - usable,
                                   optimizer_update_completed=False,
                                   mean_reward=sum(c["reward"] for g in step_groups for c in g["completions"]) /
                                   (settings.briefs_per_step * settings.samples_per_brief))
                record.data["steps"].append(step_record)
                if examples:
                    record.data["token_usage"]["training_tokens_submitted"] += sum(len(p) + len(c.tokens) - 1 for p, c, _ in examples)
                    record.write()
                    step_record["metrics"] = await backend.update(examples)
                    step_record["optimizer_update_completed"] = True
                    record.data["optimizer_updates_completed"] += 1
                    record.write()
                    await backend.refresh_sampler()
                else:
                    step_record["skipped_reason"] = "all groups have constant rewards; zero advantages"
                record.write()
            record.data["checkpoints"] = await backend.save_checkpoint()
            record.write()
            await evaluate("final", held_out)
            await backend.reload_checkpoint(record.data["checkpoints"])
            _, completions, _ = await sample_group(held_out[0], "checkpoint-reload", 1, settings.seed + 9000)
            record.data["checkpoint_reload_sampled"] = bool(completions[0].tokens)
            success = (record.data["usable_groups"] > 0 and record.data["optimizer_updates_completed"] > 0
                       and record.data["checkpoint_reload_sampled"])
            record.data.update(smoke_success=success, status="completed" if success else "no_learning_signal")
    except AuthenticationUnavailable:
        record.data.update(status="authentication_unavailable",
                           error="No Tinker credentials; use tinker auth login or TINKER_API_KEY.")
    except Exception as exc:
        # Provider exception messages can contain request details. Serialize only
        # a safe class/status, never arbitrary server text or HTTP headers.
        record.data.update(status="failed", error_type=type(exc).__name__)
        status_code = getattr(exc, "status_code", None)
        if isinstance(status_code, int):
            record.data["provider_status_code"] = status_code
            record.data["status"] = {401: "authentication_unavailable", 402: "billing_unavailable"}.get(status_code, "failed")
        if isinstance(exc, ValueError):
            record.data["error"] = str(exc)
    finally:
        try:
            await backend.close("errored" if record.data["status"] == "failed" else "success")
        except Exception as exc:
            record.data["cleanup_error_type"] = type(exc).__name__
        record.write()
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("preview", "validate", "sample", "train"))
    parser.add_argument("--optical-record", type=Path, help="load one real optical development brief using its recorded settings and caches")
    parser.add_argument("--response", default='{"starter_ids":[3,7]}', help="submission for offline validate mode")
    for name, field in Settings.model_fields.items():
        parser.add_argument("--" + name.replace("_", "-"), type=type(field.default), default=field.default)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/lux"))
    args = vars(parser.parse_args())
    mode, output_dir = args.pop("mode"), args.pop("output_dir")
    optical_record, response = args.pop("optical_record"), args.pop("response")
    if not output_dir.resolve().is_relative_to((Path.cwd() / "outputs").resolve()):
        parser.error("run records must be inside the ignored outputs/ directory")
    try:
        settings = Settings(**args)
        workload_estimate(settings, "train" if mode == "preview" else mode)
    except ValueError as exc:
        parser.error(str(exc))
    task_source = FixedTaskSource(load_optical_task(optical_record)) if optical_record else None
    if mode == "preview" and task_source:
        parser.error("use validate --optical-record for the real optical offline path")
    if mode == "preview":
        record = preview(settings, output_dir)
    elif mode == "validate":
        task = task_source.task if task_source else synthetic_task(settings.seed, "eval", 0)
        record = validate_task(task, response, output_dir, settings)
    else:
        record = asyncio.run(run_live(settings, mode, output_dir, task_source=task_source))
    if mode == "preview":
        print(next(iter(record.data["cases"].values()))["prompt"])
    print(json.dumps(dict(synthetic=record.data["synthetic"], status=record.data["status"], record=str(record.path),
                          usable_groups=record.data["usable_groups"],
                          optimizer_updates_completed=record.data["optimizer_updates_completed"],
                          smoke_success=record.data["smoke_success"],
                          estimated_upper_cost_usd=record.data["workload"]["estimated_upper_cost_usd"],
                          estimated_cost_usd=record.data["estimated_cost_usd"],
                          evaluations=record.data["evaluations"],
                          offline_reward=record.data["groups"][0]["completions"][0]["reward"] if mode == "validate" else None), indent=2))
    if record.data["status"] in ("authentication_unavailable", "billing_unavailable", "failed", "no_learning_signal", "invalid_submission"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

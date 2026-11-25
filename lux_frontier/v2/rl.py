"""Bounded Tinker experiments with group-centered, completion-only advantages."""

from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
import random
import statistics
import time

from pydantic import Field, model_validator

from lux_frontier.contracts import Contract
from lux_frontier.grpo import group_advantages, rl_arrays
from lux_frontier.tinker_backend import AuthenticationUnavailable, MODEL, RENDERER, TinkerBackend, load_renderer
from lux_frontier.v2.benchmark import FAMILIES, REPRESENTATIONS, load_envelope, provenance, render, write_envelope
from lux_frontier.v2.grading import MODES, digest, grade

RATES = dict(prefill=.33, generated=1.005, training=.737)
DEFAULT_CURRICULUM = ((0., {"A": 3., "B": 2.}), (.32, {"A": 1., "B": 2., "C": 3.}),
                      (.64, {"B": 1., "C": 2., "D": 2., "E": 4.}))


class Settings(Contract):
    steps: int = Field(default=1, ge=1, le=10000)
    seed: int = Field(default=17, ge=0, le=2**31-1)
    groups_per_step: int = Field(default=2, ge=1, le=128)
    group_size: int = Field(default=8, ge=2, le=64)
    max_attempts_per_step: int = Field(default=8, ge=1, le=512)
    minimum_groups: int = Field(default=1, ge=1, le=128)
    lora_rank: int = Field(default=16, ge=1, le=128)
    learning_rate: float = Field(default=2e-5, gt=0, le=.01)
    temperature: float = Field(default=1.25, gt=0, le=2)
    top_p: float = Field(default=1., gt=0, le=1)
    max_tokens: int = Field(default=32, ge=8, le=512)
    max_prompt_tokens: int = Field(default=4096, ge=100, le=64000)
    representation: str = "physics"
    sampling: str = "separate"
    mode: str = "yield"
    checkpoint_every: int = Field(default=5, ge=1)
    eval_every: int = Field(default=5, ge=1)
    eval_samples: int = Field(default=2, ge=1, le=32)
    eval_briefs: int = Field(default=4, ge=1)
    ttl_seconds: int = Field(default=86400, ge=3600, le=7776000)
    max_completions: int = Field(default=2000, ge=1)
    budget_usd: float = Field(default=3, gt=0)
    curriculum: tuple[tuple[float, dict[str, float]], ...] = DEFAULT_CURRICULUM

    @model_validator(mode="after")
    def coherent(self):
        if not self.minimum_groups <= self.groups_per_step <= self.max_attempts_per_step:
            raise ValueError("minimum_groups <= groups_per_step <= max_attempts_per_step required")
        if self.representation not in REPRESENTATIONS or self.mode not in MODES or self.sampling not in ("batched", "separate"):
            raise ValueError("unsupported reward, representation or sampling mode")
        if self.max_prompt_tokens+self.max_tokens > 64000:
            raise ValueError("request exceeds the selected model's context")
        starts = [start for start, _ in self.curriculum]
        if not starts or starts[0] != 0 or sorted(set(starts)) != starts or starts[-1] >= 1:
            raise ValueError("curriculum requires unique increasing start fractions in [0,1)")
        for _, weights in self.curriculum:
            if not weights or any(f not in FAMILIES or v <= 0 or not math.isfinite(v) for f, v in weights.items()):
                raise ValueError("curriculum weights must be positive and refer to families A–E")
        return self


def seed_for(*keys):
    return int(digest(keys)[:8], 16) % (2**31-1)


def estimate(settings, *, groups, updates=0, other_samples=0, probes=0, checkpoints=0):
    attempted = groups*settings.group_size
    trained = updates*settings.groups_per_step*settings.group_size
    counts = dict(prefill=(attempted+other_samples)*settings.max_prompt_tokens
                  + probes*(settings.max_prompt_tokens+settings.max_tokens),
                  generated=(attempted+other_samples)*settings.max_tokens,
                  training=trained*(settings.max_prompt_tokens+settings.max_tokens-1))
    reserve = .10 * checkpoints * max(1., settings.ttl_seconds/(30*86400))
    cost = 3*sum(counts[k]*RATES[k]/1e6 for k in counts)+reserve
    return dict(maximum_sampled_completions=attempted+other_samples, token_bound=counts,
                storage_reserve_usd=reserve, budget_estimate_usd=round(cost, 6),
                rates_usd_per_million=RATES, application_margin=3, provider_limit=False)


def training_plan(settings, held_out_count):
    evals = 1 + math.ceil(settings.steps/settings.eval_every)
    checkpoints = math.ceil(settings.steps/settings.checkpoint_every)
    return estimate(settings, groups=settings.max_attempts_per_step*settings.steps, updates=settings.steps,
                    other_samples=evals*min(settings.eval_briefs, held_out_count)*settings.eval_samples,
                    probes=4, checkpoints=checkpoints)


def enforce(plan, settings):
    if plan["maximum_sampled_completions"] > settings.max_completions or plan["budget_estimate_usd"] > settings.budget_usd:
        raise ValueError(f"run exceeds limits: {plan['maximum_sampled_completions']} completions, "
                         f"${plan['budget_estimate_usd']:.3f} application estimate")


class Backend(TinkerBackend):
    async def initialize(self, training, *, checkpoint=None, resume=None):
        import tinker
        from tinker.auth import tinker_has_credentials

        if not tinker_has_credentials():
            raise AuthenticationUnavailable("Tinker credentials unavailable")
        self.tinker = tinker
        self.tokenizer, self.renderer = load_renderer()
        self.service = tinker.ServiceClient()
        if resume:
            self.training = await self.service.create_training_client_from_state_with_optimizer(resume["training_state"])
            await self.refresh_sampler()
        elif training:
            self.training = await self.service.create_lora_training_client(
                base_model=MODEL, rank=self.settings.lora_rank, seed=self.settings.seed)
            await self.refresh_sampler()
        elif checkpoint:
            self.sampler = await self.service.create_sampling_client(model_path=checkpoint)
        else:
            self.sampler = await self.service.create_sampling_client(base_model=MODEL)

    async def save(self, index):
        name = f"lux-v2-step-{index:06d}"
        ttl = self.settings.ttl_seconds
        state = await self.training.save_state(name=name, ttl_seconds=ttl)
        weights = await self.training.save_weights_for_sampler(name=name, ttl_seconds=ttl)
        return dict(step=index, training_state=state.path, sampler_weights=weights.path, ttl_seconds=ttl,
                    created_utc=datetime.now(timezone.utc).isoformat())

    async def probe(self, tokens, completion):
        full = tokens + completion
        values = await self.sampler.compute_logprobs(self.tinker.ModelInput.from_ints(full))
        filtered = values[len(tokens):]
        if len(values) != len(full) or any(v is None or not math.isfinite(v) for v in filtered):
            raise ValueError("probe likelihoods could not be aligned to completion tokens")
        return filtered


def entropy(values):
    counts = Counter(values)
    n = sum(counts.values())
    return -sum((c/n)*math.log2(c/n) for c in counts.values())


def assess_group(task, completions, phase, mode):
    grades = [grade(task, c.response_text, clean=c.clean, phase=phase, mode=mode) for c in completions]
    valid = [g for g in grades if g.error is None]
    optical = len({g.reward for g in valid}) > 1
    rewards = [g.reward for g in grades]
    # Keep malformed completions visible as failures, and exclude them from the
    # policy-gradient batch. They cannot determine the optical baseline.
    valid_advantages = iter(group_advantages([g.reward for g in valid]) if valid else ())
    advantages = [next(valid_advantages) if g.error is None else 0. for g in grades]
    logprobs = [v for c in completions for v in (c.logprobs or []) if math.isfinite(v)]
    positions = list(range(min(16, max((len(c.tokens) for c in completions), default=0))))
    token_entropy = [entropy([c.tokens[i] for c in completions if i < len(c.tokens)]) for i in positions]
    diagnostics = dict(unique_valid_actions=len({g.starter_id for g in valid}),
                       action_counts=dict(Counter(str(g.starter_id) if g.error is None else "invalid" for g in grades)),
                       valid_action_entropy_bits=entropy([g.starter_id for g in valid]),
                       response_entropy_bits=entropy([c.raw_response for c in completions]),
                       token_position_entropy_bits=token_entropy,
                       mean_sampled_token_surprisal_nats=-statistics.mean(logprobs) if logprobs else None,
                       reward_variance=statistics.pvariance(rewards), invalid=len(grades)-len(valid),
                       stop_reasons=dict(Counter(c.stop_reason for c in completions)), informative=optical,
                       malformed_only=not optical and max(rewards) > min(rewards),
                       primary_outcome_spread=len({(g.nominal_pass, g.yield_fraction if mode == "yield" else 0) for g in valid}) > 1)
    return grades, advantages, diagnostics


class Ledger:
    def __init__(self, path, settings, plan, mode, inputs):
        if path.exists():
            raise ValueError("run record already exists; select a new output path")
        self.path = path
        self.journal = path.with_name(path.name + ".groups.jsonl")
        if self.journal.exists():
            raise ValueError("run journal already exists; select a new output path")
        self.settings = settings
        self.started = time.perf_counter()
        self.data = dict(version="lux-v2-run-1", model=MODEL, renderer=RENDERER,
                         mode=mode, status="running", settings=settings.model_dump(mode="json"), plan=plan,
                         inputs=inputs, provenance=provenance(),
                         usage=dict(prefill=0, generated=0, training=0), sampled_completions=0,
                         group_journal=self.journal.name, group_count=0, steps=[], checkpoints=[], evaluations=[], responses=[], probes=[],
                         attempted=0, accepted=0, rejected=0)
        self.flush()

    def cost(self):
        return 3*sum(self.data["usage"][k]*RATES[k]/1e6 for k in RATES)

    def flush(self):
        self.data.update(runtime_seconds=time.perf_counter()-self.started, estimated_token_spend_usd=round(self.cost(), 6))
        write_envelope(self.path, self.data, "run")

    def record_group(self, group):
        with self.journal.open("a") as stream:
            stream.write(json.dumps(group, separators=(",", ":"), allow_nan=False)+"\n")
        self.data["group_count"] += 1
        self.flush()

    def reserve(self, *, prefill=0, generated=0, training=0, completions=0):
        projected = self.cost()+3*(prefill*RATES["prefill"]+generated*RATES["generated"]+training*RATES["training"])/1e6
        total_completions = self.data["sampled_completions"] + self.data.get("prior_sampled_completions", 0) + completions
        if (projected + self.data.get("prior_estimated_token_spend_usd", 0) + self.data["plan"]["storage_reserve_usd"] > self.settings.budget_usd
                or total_completions > self.settings.max_completions):
            raise ValueError("application spending or sampling limit reached")
        self.data["usage"]["prefill"] += prefill
        self.data["usage"]["generated"] += generated
        self.data["usage"]["training"] += training
        self.data["sampled_completions"] += completions
        self.flush()


async def sample_group(ledger, backend, task, *, phase, mode, count, seed, strategy, representation,
                       temperature=None, context=None):
    settings = ledger.settings
    prompt = render(task, representation)
    tokens = backend.encode(prompt)
    if not 0 < len(tokens) <= settings.max_prompt_tokens:
        raise ValueError("rendered prompt exceeds configured token bounds")
    if temperature is not None:
        backend.settings = settings.model_copy(update={"temperature": temperature})
    requests = [(count, seed)] if strategy == "batched" else [(1, seed_for(seed, i)) for i in range(count)]
    completions = []
    try:
        # Reserve conservatively before provider calls so partial network failures
        # leave an upper-bound token charge in the local ledger.
        ledger.reserve(prefill=len(tokens)*count, generated=settings.max_tokens*count, completions=count)
        for n, request_seed in requests:
            sampled = await backend.sample(tokens, n, request_seed)
            if len(sampled) != n or any(len(c.tokens) > settings.max_tokens for c in sampled):
                raise ValueError("provider returned an unexpected completion count or length")
            completions.extend(sampled)
        ledger.data["usage"]["generated"] += sum(len(c.tokens) for c in completions)-count*settings.max_tokens
    finally:
        backend.settings = settings
        if len(completions) != count:
            ledger.data.setdefault("partial_groups", []).append(dict(task_id=task["task_id"], prompt=prompt, requests=requests,
                                                                      completions=[asdict(c) for c in completions]))
            ledger.flush()
    grades, advantages, stats = assess_group(task, completions, phase, mode)
    group = dict(task_id=task["task_id"], task_sha256=digest(task), prompt=prompt, prompt_sha256=digest(prompt),
                 representation=representation, phase=phase, mode=mode, requests=requests, sampling=strategy,
                 temperature=temperature if temperature is not None else settings.temperature, prompt_tokens=len(tokens),
                 completions=[dict(asdict(c), grade=asdict(g), advantage=a)
                              for c, g, a in zip(completions, grades, advantages, strict=True)], diagnostics=stats)
    group.update(context or {})
    ledger.record_group(group)
    return group, tokens, completions, advantages


def staged_tasks(tasks, settings, step):
    stage = max((s for s in settings.curriculum if s[0] <= step/settings.steps), key=lambda s: s[0])
    candidates = {f: [t for t in tasks if t["capability"] and f in t["construction"]["families"]]
                  for f in stage[1]}
    present = {f: w for f, w in stage[1].items() if candidates[f]}
    if not present:
        raise ValueError("no eligible curriculum tasks for a configured stage")
    rng = random.Random(seed_for(settings.seed, step, "task-schedule"))
    seen = set()
    for _ in range(settings.max_attempts_per_step):
        available = {f: [t for t in candidates[f] if t["task_id"] not in seen] for f in present}
        nonempty = [f for f in present if available[f]]
        if not nonempty:
            seen.clear()
            available = candidates
            nonempty = list(present)
        family = rng.choices(nonempty, [present[f] for f in nonempty], k=1)[0]
        task = rng.choice(available[family])
        seen.add(task["task_id"])
        yield task, family


async def dev_evaluate(ledger, backend, tasks, step):
    if not tasks:
        return
    settings = ledger.settings
    entries = []
    for task in tasks:
        group, *_ = await sample_group(ledger, backend, task, phase="development", mode="yield", count=settings.eval_samples,
                                       seed=seed_for(settings.seed, task["task_id"], "dev"), strategy=settings.sampling,
                                       representation=settings.representation)
        entries.append(group)
    ledger.data["evaluations"].append(dict(step=step, task_ids=[t["task_id"] for t in tasks],
        metrics={m: statistics.mean(statistics.mean(float(c["grade"][m]) for c in g["completions"]) for g in entries)
                 for m in ("reward", "yield_fraction", "final_pass", "nominal_pass")}))
    ledger.flush()


def resume_state(path, settings, inputs, provenance_data):
    old = load_envelope(path, "run")
    if (old["mode"] != "train" or old["model"] != MODEL or old["settings"] != settings.model_dump(mode="json")
            or old["inputs"] != inputs or old["provenance"]["source_hashes"] != provenance_data["source_hashes"]):
        raise ValueError("resume requires the same data, schedule, settings and source files")
    if not old["checkpoints"]:
        raise ValueError("run has no recoverable optimizer checkpoint")
    checkpoint = old["checkpoints"][-1]
    if checkpoint["step"] >= settings.steps:
        raise ValueError("the checkpoint already reached the requested schedule length")
    return checkpoint, dict(prior_estimated_token_spend_usd=old["estimated_token_spend_usd"]+old.get("prior_estimated_token_spend_usd", 0.),
                           prior_sampled_completions=old["sampled_completions"]+old.get("prior_sampled_completions", 0))


async def train(settings, train_tasks, dev_tasks, path, *, backend=None, resume=None):
    if not train_tasks or not dev_tasks or any(t["split"] != "train" for t in train_tasks) or any(
            t["split"] != "development" for t in dev_tasks):
        raise ValueError("training requires disjoint train and development task lists")
    dev_order = sorted(dev_tasks, key=lambda t: digest([settings.seed, t["task_id"]]))
    certified = [t for t in dev_order if t["capability"]][:math.ceil(settings.eval_briefs/2)]
    dev_tasks = [*certified, *[t for t in dev_order if t not in certified]][:settings.eval_briefs]
    for start, _ in settings.curriculum:
        step = math.ceil(start*settings.steps)
        if step < settings.steps:
            next(staged_tasks(train_tasks, settings, step))
    plan = training_plan(settings, len(dev_tasks))
    enforce(plan, settings)
    inputs = dict(train=[digest(t) for t in train_tasks], development=[digest(t) for t in dev_tasks])
    ledger = Ledger(path, settings, plan, "train", inputs)
    ledger.data["execution_backend"] = "offline_injected_backend" if backend else "tinker"
    backend = backend or Backend(settings)
    checkpoint, first, updates = None, 0, 0
    probe = None
    try:
        if resume:
            checkpoint, prior_usage = resume_state(resume, settings, inputs, ledger.data["provenance"])
            first = checkpoint["step"]
            ledger.data.update(resumed_from=str(resume), original_checkpoint=checkpoint, **prior_usage)
        await backend.initialize(True, resume=checkpoint)
        await dev_evaluate(ledger, backend, dev_tasks, first)
        for step in range(first, settings.steps):
            used, attempted, examples = 0, 0, []
            for index, (task, family) in enumerate(staged_tasks(train_tasks, settings, step)):
                g, tokens, completions, adv = await sample_group(
                    ledger, backend, task, phase="training", mode=settings.mode, count=settings.group_size,
                    seed=seed_for(settings.seed, step, index, task["task_id"]), strategy=settings.sampling,
                    representation=settings.representation, context=dict(step=step+1, curriculum_family=family))
                attempted += 1
                accepted = g["diagnostics"]["informative"]
                if accepted:
                    selected = [(tokens, c, a) for c, a in zip(completions, adv, strict=True) if a]
                    for datum in selected:
                        rl_arrays(*datum)
                    examples.extend(selected)
                    used += 1
                ledger.data["attempted"] += 1
                ledger.data["accepted" if accepted else "rejected"] += 1
                ledger.flush()
                if used == settings.groups_per_step:
                    break
            rec = dict(step=step+1, attempted=attempted, accepted=used, rejected=attempted-used, updated=False)
            ledger.data["steps"].append(rec)
            ledger.flush()
            if used >= settings.minimum_groups:
                if probe is None:
                    p, c, _ = examples[0]
                    ledger.reserve(prefill=len(p)+len(c.tokens))
                    before = await backend.probe(p, c.tokens)
                ledger.reserve(training=sum(len(p)+len(c.tokens)-1 for p, c, _ in examples))
                rec["metrics"] = await backend.update(examples)
                rec["updated"] = True
                updates += 1
                ledger.flush()
                await backend.refresh_sampler()
                if probe is None:
                    ledger.reserve(prefill=len(p)+len(c.tokens))
                    after = await backend.probe(p, c.tokens)
                    probe = dict(step=step+1, prompt_tokens=p, completion_tokens=c.tokens, before=before, after=after,
                                 max_abs_delta=max(abs(a-b) for a, b in zip(before, after, strict=True)))
                    ledger.data["probes"].append(probe)
                ledger.flush()
            else:
                rec["skipped"] = "insufficient informative groups"
            final = step+1 == settings.steps
            if updates and ((step+1) % settings.checkpoint_every == 0 or final):
                ledger.data["checkpoints"].append(await backend.save(step+1))
                ledger.flush()
            if (step+1) % settings.eval_every == 0 or final:
                await dev_evaluate(ledger, backend, dev_tasks, step+1)
        if ledger.data["checkpoints"]:
            p, c = probe["prompt_tokens"], probe["completion_tokens"]
            ledger.reserve(prefill=2*(len(p)+len(c)))
            final_likelihoods = await backend.probe(p, c)
            await backend.reload_checkpoint(ledger.data["checkpoints"][-1])
            reloaded = await backend.probe(p, c)
            ledger.data["checkpoint_reload_probe"] = dict(logprobs=reloaded, before_reload=final_likelihoods,
                max_abs_delta=max(abs(a-b) for a, b in zip(final_likelihoods, reloaded, strict=True)))
        reload_ok = ledger.data.get("checkpoint_reload_probe", {}).get("max_abs_delta", math.inf) < .02
        ledger.data["status"] = ("completed" if probe and probe["max_abs_delta"] > 1e-6 and reload_ok else
                                 "update_unverified" if updates else "no_learning_signal")
    except Exception as exc:
        ledger.data.update(status="authentication_unavailable" if isinstance(exc, AuthenticationUnavailable) else "failed",
                           error_type=type(exc).__name__)
        raise
    finally:
        try:
            await backend.close("success" if ledger.data["status"] == "completed" else "errored")
        except Exception as exc:
            ledger.data["cleanup_error_type"] = type(exc).__name__
        finally:
            ledger.flush()
    return ledger.data


async def inference(settings, tasks, path, *, exploration=False, checkpoint=None, temperatures=(.6, 1., 1.35), backend=None):
    if not tasks:
        raise ValueError("no inference tasks selected")
    if exploration and (any(t["split"] == "test" for t in tasks) or not temperatures
                        or any(not 0 < t <= 2 for t in temperatures)):
        raise ValueError("exploration uses training/development tasks and temperatures in (0, 2]")
    combinations = ([(r, strategy, temp) for r in REPRESENTATIONS for strategy in ("batched", "separate")
                     for temp in temperatures] if exploration else [(settings.representation, settings.sampling, settings.temperature)])
    groups = len(tasks)*len(combinations)
    plan = estimate(settings, groups=groups if exploration else 0, other_samples=0 if exploration else groups*settings.eval_samples)
    enforce(plan, settings)
    ledger = Ledger(path, settings, plan, "exploration" if exploration else "inference", [digest(t) for t in tasks])
    ledger.data["checkpoint"] = checkpoint
    ledger.data["execution_backend"] = "offline_injected_backend" if backend else "tinker"
    backend = backend or Backend(settings)
    try:
        await backend.initialize(False, checkpoint=checkpoint)
        settings_summary = {}
        for representation, strategy, temp in combinations:
            measurements = []
            for task in tasks:
                phase = {"train": "training", "development": "development", "test": "final"}[task["split"]]
                g, *_ = await sample_group(ledger, backend, task, phase=phase, mode="yield",
                    count=settings.group_size if exploration else settings.eval_samples, seed=seed_for(settings.seed, task["task_id"]),
                    strategy=strategy, representation=representation, temperature=temp)
                measurements.append(g)
                ledger.data["responses"].extend(dict(task_id=task["task_id"], task_sha256=g["task_sha256"],
                    prompt_sha256=g["prompt_sha256"], representation=representation, sampling=strategy, temperature=temp,
                    response=c["response_text"], raw_response=c["raw_response"], stop_reason=c["stop_reason"],
                    clean=c["clean"], sampled_reward=c["grade"]["reward"]) for c in g["completions"])
            key = f"{representation}/{strategy}/{temp:g}"
            settings_summary[key] = dict(groups=len(measurements), mean_prompt_tokens=statistics.mean(g["prompt_tokens"] for g in measurements),
                mean_reward=statistics.mean(c["grade"]["reward"] for g in measurements for c in g["completions"]),
                informative_fraction=statistics.mean(g["diagnostics"]["informative"] for g in measurements),
                malformed_only_fraction=statistics.mean(g["diagnostics"]["malformed_only"] for g in measurements),
                invalid_fraction=sum(g["diagnostics"]["invalid"] for g in measurements)/sum(len(g["completions"]) for g in measurements),
                mean_unique_valid_actions=statistics.mean(g["diagnostics"]["unique_valid_actions"] for g in measurements))
        per_task = {}
        for row in ledger.data["responses"]:
            t = row["task_id"]
            key = f"{row['representation']}/{row['sampling']}/{row['temperature']:g}"
            per_task.setdefault(t, {}).setdefault(key, []).append(row["sampled_reward"])
        ledger.data.update(status="completed", diagnostics=settings_summary, by_task_reward_samples=per_task)
    except Exception as exc:
        ledger.data.update(status="authentication_unavailable" if isinstance(exc, AuthenticationUnavailable) else "failed",
                           error_type=type(exc).__name__)
        raise
    finally:
        try:
            await backend.close("success" if ledger.data["status"] == "completed" else "errored")
        except Exception as exc:
            ledger.data["cleanup_error_type"] = type(exc).__name__
        finally:
            ledger.flush()
    return ledger.data

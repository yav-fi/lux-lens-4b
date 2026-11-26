"""Frozen final protocols and lineage-level paired inference."""

from collections import defaultdict
from datetime import datetime, timezone
from itertools import product
import math
import random
import statistics
from statistics import NormalDist

from lux_frontier.v2.benchmark import load_envelope, load_tasks, render, validate_splits, write_envelope
from lux_frontier.v2.evaluation import METRICS, evaluate
from lux_frontier.v2.grading import digest
from lux_frontier.v2.manufacturing import STREAM_VERSION, namespace

SEAL_VERSION = "lux-v21-evaluation-1"
ARMS = ("untouched_qwen", "random_reward_qwen", "nominal_only_qwen", "yield_aware_qwen")
REFERENCES = (*ARMS[:3], "physics_heuristic", "closest_to_spec")


def clusters(tasks):
    """Connect mixed-ancestry briefs and related counterfactuals transitively."""
    parents = {}

    def find(key):
        parents.setdefault(key, key)
        while key != parents[key]:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key

    for task in tasks:
        keys = [("task", task["task_id"]), ("group", task["group"]), ("cohort", task["cohort"]),
                *(("ancestry", c["prescription"]["ancestry"]) for c in task["candidates"])]
        for key in keys[1:]:
            parents[find(key)] = find(keys[0])
    return [find(("task", t["task_id"]))[1] for t in tasks]


def geometry(candidate):
    p = candidate["prescription"]
    scale = p["sensor_z_mm"]
    normalized = [v/scale for v in [*p["radii_mm"], *p["spacings_mm"], *p["clear_semidiameters_mm"],
                                    p["stop_diameter_mm"]]]
    return (len(p["radii_mm"]), p["stop_surface"]), normalized


def check_isolation(tasks):
    validate_splits(tasks)
    prompts, physical = {}, defaultdict(dict)
    for task in tasks:
        split = task["split"]
        for representation in ("summary", "full", "physics"):
            key = digest(render(task, representation))
            if prompts.setdefault(key, split) != split:
                raise ValueError("identical prompt across splits")
        for c in task["candidates"]:
            physical[split].setdefault(c["identity"], geometry(c))
    splits = sorted(physical)
    for i, a in enumerate(splits):
        for b in splits[i+1:]:
            for topology_a, x in physical[a].values():
                for topology_b, y in physical[b].values():
                    if topology_a == topology_b and all(abs(u-v) <= .005*max(1., abs(u), abs(v))
                                                        for u, v in zip(x, y, strict=True)):
                        raise ValueError(f"near-duplicate geometry across {a} and {b}")


def freeze(source, output, settings, *, power_plan, used_manifests):
    if output.exists():
        raise ValueError("sealed protocol already exists")
    manifest = load_envelope(source, "manifest")
    tasks = load_tasks(source)
    prior = [task for path in used_manifests for task in load_tasks(path)]
    final = sorted((t for t in tasks if t["split"] == "test"), key=lambda t: t["task_id"])
    if not final or not used_manifests:
        raise ValueError("supply final tasks and at least one previously inspected manifest")
    # The original split may be called test, but an inspected pilot belongs on the development side of this audit.
    combined = {t["task_id"]: t for t in tasks if t["split"] != "test"}
    combined.update({f"inspected-{digest(t)[:20]}": dict(t, task_id=f"inspected-{digest(t)[:20]}",
                                                       split="development" if t["split"] == "test" else t["split"])
                     for t in prior})
    check_isolation([*combined.values(), *final])
    needed = power_plan.get("recommended_lineages")
    independent = len(set(clusters(final)))
    if not isinstance(needed, int) or needed < 2 or independent < needed:
        raise ValueError("final tasks do not meet the development-derived independent lineage requirement")
    simulation = manifest.get("manufacturing", {})
    if (simulation.get("phase") != "final" or simulation.get("fidelity") != "definitive"
            or set(simulation.get("task_ids", [])) != {t["task_id"] for t in final}):
        raise ValueError("run fresh definitive manufacturing across the complete final panel")
    registry = {}
    for task in final:
        details = load_envelope(source.parent/"replay-data"/(task["task_id"]+".json"), "details")
        if digest(details) != task["replay_details_sha256"]:
            raise ValueError("sealed build evidence differs from the task")
        for local, outcome in task["outcomes"].items():
            if not outcome["nominal_pass"]:
                continue
            row = outcome["simulations"]["final"]
            fresh = namespace(simulation["seed"], task["group"], "final", "definitive")
            rows = details[local]["builds"]["final"]
            if (row["fidelity"] != "definitive" or row["tested"] != simulation["builds"] or row["tested"] < 100
                    or row["stream_prefix"] != fresh or row["passed"] != sum(b["passed"] for b in rows)
                    or [b["stream"] for b in rows] != [[*fresh, i] for i in range(row["tested"])]):
                raise ValueError("final draw records are incomplete or use an unexpected stream")
            if registry.setdefault(tuple(fresh), task["group"]) != task["group"]:
                raise ValueError("final draw stream collision between independent groups")
    record = dict(version=SEAL_VERSION, created_utc=datetime.now(timezone.utc).isoformat(), manifest_sha256=digest(manifest), simulation=simulation,
                  manufacturing_stream=STREAM_VERSION, sample_settings=settings.model_dump(mode="json"),
                  independent_lineages=independent, power_plan=power_plan, estimand="equal_weight_per_brief",
                  primary="nominal_pass_and_observed_simulated_yield_at_target", arms=list(ARMS),
                  comparisons=list(REFERENCES), correction="Holm across primary pass-rate contrasts",
                  task_ids=[t["task_id"] for t in final], task_sha256=dict((t["task_id"], digest(t)) for t in final),
                  registered_training_tasks=sorted({digest(t) for t in [*tasks, *prior] if t["split"] == "train"}),
                  registered_development_tasks=sorted({digest(t) for t in [*tasks, *prior] if t["split"] == "development"}),
                  inspected_manifests=[digest(load_envelope(path, "manifest")) for path in used_manifests])
    write_envelope(output, record, "seal")
    return record


def verify_seal(path, source, tasks, settings=None):
    seal = load_envelope(path, "seal")
    if (seal["version"] != SEAL_VERSION or digest(load_envelope(source, "manifest")) != seal["manifest_sha256"]
            or {t["task_id"]: digest(t) for t in tasks} != seal["task_sha256"]):
        raise ValueError("tasks do not match the complete frozen final protocol")
    if settings is not None and type(settings).model_validate(seal["sample_settings"]) != settings:
        raise ValueError("sampling settings differ from the frozen protocol")
    return seal


def training_provenance(path, checkpoint, seal):
    run = load_envelope(path, "run")
    settings, inputs = run["settings"], run["inputs"]
    if (run["mode"] != "train" or run["status"] != "completed" or run["execution_backend"] != "tinker"
            or checkpoint not in {c["sampler_weights"] for c in run["checkpoints"]}
            or not settings["matched_attempts"] or run["started_utc"] < seal["created_utc"]):
        raise ValueError("checkpoint needs a completed, matched Tinker run started after the evaluation seal")
    if not set(inputs["train"]) <= set(seal["registered_training_tasks"]) or not set(inputs["development"]) <= set(
            seal["registered_development_tasks"]):
        raise ValueError("checkpoint used training or development inputs outside the sealed registry")
    return dict(sha256=digest(run), settings=settings, inputs=inputs, attempted_groups=run["attempted"],
                accepted_groups=run["accepted"], optimizer_updates=run["optimizer_updates"],
                source_hashes=run["provenance"]["source_hashes"], estimated_token_spend_usd=run["estimated_token_spend_usd"])


def signed_randomization(values, groups, *, seed=17):
    totals = list(group_totals(values, groups).values())
    if len(totals) < 2:
        return None
    observed = abs(sum(totals))
    if len(totals) <= 15:
        null = product((-1, 1), repeat=len(totals))
        offset = 0
    else:
        rng = random.Random(seed)
        null = (rng.choices((-1, 1), k=len(totals)) for _ in range(19999))
        offset = 1
    results = [abs(sum(s*v for s, v in zip(signs, totals, strict=True))) >= observed-1e-12 for signs in null]
    return (sum(results)+offset)/(len(results)+offset)


def group_totals(values, labels):
    totals = defaultdict(float)
    for value, label in zip(values, labels, strict=True):
        totals[label] += value
    return totals


def power(evaluation, tasks, *, arm, reference, effect, desired=.8, alpha=.05/len(REFERENCES)):
    if evaluation["phase"] != "development" or any(t["split"] != "development" for t in tasks):
        raise ValueError("sample-size planning requires development data")
    if not 0 < effect < 1 or not .5 < desired < 1 or not 0 < alpha < .5:
        raise ValueError("invalid sample-size assumptions")
    labels = clusters(tasks)
    rows = evaluation["task_results"]
    values = [rows[t["task_id"]][arm]["scores"]["final_pass"]-rows[t["task_id"]][reference]["scores"]["final_pass"]
              for t in tasks]
    by_cluster = [total/labels.count(label) for label, total in group_totals(values, labels).items()]
    sd = statistics.stdev(by_cluster) if len(by_cluster) > 1 else None
    minimum_randomization = next(n for n in range(2, 32) if 2/2**n <= alpha)
    needed = None if not sd else max(minimum_randomization, math.ceil(
        ((NormalDist().inv_cdf(1-alpha/2)+NormalDist().inv_cdf(desired))*sd/effect)**2))
    return dict(recommended_lineages=needed, development_lineages=len(by_cluster), paired_lineage_sd=sd,
                minimum_effect=effect, target_power=desired, alpha_per_contrast=alpha,
                minimum_exact_test_resolution_lineages=minimum_randomization, comparison=[arm, reference],
                planning_assumption="approximately equal numbers of briefs per lineage",
                development_evaluation_sha256=digest(evaluation))


def final_evaluate(tasks, training, predictions, seal):
    from lux_frontier.v2.evaluation import paired_interval

    if set(predictions) != set(ARMS):
        raise ValueError("all four predeclared Qwen arms must be supplied")
    setting_keys = ("seed", "sampling", "temperature", "top_p", "max_tokens", "representation", "format_example", "eval_samples")
    settings = seal["sample_settings"]
    from lux_frontier.tinker_backend import MODEL, RENDERER
    checkpoints = []
    matched = []
    for arm, record in predictions.items():
        if (record["model"] != MODEL or record.get("renderer") != RENDERER or record.get("execution_backend") != "tinker"
                or record.get("status") != "completed" or record.get("evaluation_seal_sha256") != digest(seal)):
            raise ValueError(f"{arm} is not a completed inference run under this sealed protocol")
        if any(record["settings"][key] != settings[key] for key in setting_keys):
            raise ValueError(f"{arm} does not match the sealed sampling settings")
        checkpoint = record.get("checkpoint")
        if bool(checkpoint) != (arm != "untouched_qwen"):
            raise ValueError(f"{arm} checkpoint provenance does not match its arm")
        if checkpoint:
            checkpoints.append(checkpoint)
            provenance = record.get("checkpoint_training")
            wanted = dict(zip(ARMS[1:], ("random", "nominal", "yield"), strict=True))[arm]
            if not provenance or provenance["settings"]["mode"] != wanted:
                raise ValueError(f"{arm} must use the correct reward arm training record")
            matched.append(digest([provenance["inputs"], provenance["attempted_groups"], provenance["source_hashes"],
                                   {k: v for k, v in provenance["settings"].items() if k != "mode"}]))
        counts = defaultdict(int)
        for row in record["responses"]:
            counts[row["task_id"]] += 1
        if dict(counts) != dict.fromkeys(seal["task_ids"], settings["eval_samples"]):
            raise ValueError(f"{arm} does not have the frozen number of attempts for each brief")
    if len(set(checkpoints)) != len(checkpoints):
        raise ValueError("trained arms need distinct checkpoints")
    if len(set(matched)) != 1:
        raise ValueError("trained arms must share source, task inputs, settings and attempted rollout workload")
    result = evaluate(tasks, phase="final", train_tasks=training, predictions=predictions, paired_reference=ARMS[0])
    cells, labels = result["task_results"], clusters(tasks)
    contrasts = {}
    for ref in REFERENCES:
        contrasts[ref] = {}
        for metric in METRICS:
            delta = [cells[t["task_id"]][ARMS[-1]]["scores"][metric]-cells[t["task_id"]][ref]["scores"][metric] for t in tasks]
            contrasts[ref][metric] = paired_interval(delta, labels, seed=17)
            if metric == "final_pass":
                contrasts[ref][metric]["paired_sign_flip_p"] = signed_randomization(delta, labels)
    order = sorted(REFERENCES, key=lambda r: contrasts[r]["final_pass"]["paired_sign_flip_p"])
    adjusted = 0.
    for i, ref in enumerate(order):
        row = contrasts[ref]["final_pass"]
        adjusted = max(adjusted, min(1., (len(order)-i)*row["paired_sign_flip_p"]))
        row["holm_adjusted_p"] = adjusted
    methods = list(next(iter(cells.values())))
    grouped = defaultdict(list)
    for task, cluster in zip(tasks, labels, strict=True):
        grouped[cluster].append(task)
    by_lineage = {}
    for panel in grouped.values():
        ancestry = " | ".join(sorted({c["prescription"]["ancestry"] for t in panel for c in t["candidates"]}))
        by_lineage[ancestry] = dict(tasks=len(panel), methods={
            method: {metric: statistics.mean(cells[t["task_id"]][method]["scores"][metric] for t in panel)
                     for metric in METRICS} for method in methods})
    sensitivity = {}
    for method in methods:
        proportions = []
        for task in tasks:
            counts = []
            for attempt in cells[task["task_id"]][method]["attempts"]:
                status = "below"
                if attempt["nominal_pass"]:
                    status = task["outcomes"][str(attempt["starter_id"])]["simulations"]["final"]["threshold_status"]
                counts.append((status == "above", status != "below"))
            proportions.append([statistics.mean(col) for col in zip(*counts, strict=True)])
        low, high = (statistics.mean(col) for col in zip(*proportions, strict=True))
        sensitivity[method] = dict(above_interval_fraction=low, including_unresolved_fraction=high)
    result.update(seal_sha256=digest(seal), primary_contrasts=contrasts, per_lineage=by_lineage,
                  simulated_threshold_sensitivity=sensitivity)
    return result

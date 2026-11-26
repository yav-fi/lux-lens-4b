"""Matched physical evaluation and task-level statistics."""

from collections import Counter, defaultdict
from dataclasses import asdict
import random
import statistics

import numpy as np

from lux_frontier.v2.benchmark import heuristic, render, spec_terms
from lux_frontier.v2.grading import digest, grade, submit
from lux_frontier.v2.optics import Requirements

METRICS = ("final_pass", "yield_fraction", "regret", "invalid", "nominal_pass")


def audit(tasks):
    """Describe the frozen construction landscape without model calls or regrading optics."""
    strata = {}
    contrasts = []
    groups = defaultdict(list)
    for task in tasks:
        groups[task["group"]].append(task)
    for siblings in groups.values():
        for i, a in enumerate(siblings):
            for b in siblings[i+1:]:
                by_identity = [{c["identity"]: grade(t, submit(c["id"]), phase="construction") for c in t["candidates"]}
                               for t in (a, b)]
                left, right = by_identity
                common = sorted(left.keys() & right.keys())
                pairs = [[x, y] for x in common for y in common if all(g[x].nominal_pass and g[y].nominal_pass
                         for g in by_identity) and left[x].yield_fraction > left[y].yield_fraction
                         and right[x].yield_fraction < right[y].yield_fraction]
                if pairs:
                    contrasts.append(dict(tasks=[a["task_id"], b["task_id"]], reversing_pairs=pairs))
    for split in sorted({t["split"] for t in tasks}):
        subset = [t for t in tasks if t["split"] == split]
        stats = [t["construction"] for t in subset]
        phase = {"train": "training", "development": "development", "test": "final"}[split]
        confirmed = 0
        sharper_but_invalid = 0
        for t in subset:
            row = t["construction"]
            grades = {c["id"]: grade(t, submit(c["id"]), phase=phase) for c in t["candidates"]}
            if any(grades[j].yield_fraction-grades[i].yield_fraction >= .25 for i, j in row["manufacturing_reversals"]):
                confirmed += 1
            if row["nominally_feasible"] and not grades[row["initial_sharpest_id"]].nominal_pass:
                sharper_but_invalid += 1
        strata[split] = dict(tasks=len(subset), capability=sum(t["capability"] for t in subset),
            rejection_reasons=dict(Counter(t["capability_rejection"] for t in subset if not t["capability"])),
            families=dict(Counter(f for s in stats for f in s["families"])),
            closest_to_spec_regret_cases=sum(s["heuristic_regret"] >= .1 for s in stats),
            sharpest_regret_cases=sum(s["initial_sharpest_regret"] >= .1 for s in stats),
            initially_sharpest_nominal_failures_on_feasible_tasks=sharper_but_invalid,
            construction_manufacturing_reversal_counts=sum(bool(s["manufacturing_reversals"]) for s in stats),
            independently_repeated_manufacturing_reversal_tasks=confirmed,
            brief_switch_cases=sum(bool(s.get("brief_switches")) for s in stats),
            disjoint_profile_optima_cases=sum(bool(s.get("profile_switches")) for s in stats),
            graded_design_draws=sum(b["tested"] for t in subset for row in t["outcomes"].values()
                                    for b in row["simulations"].values()),
            average_construction_random_reward=statistics.mean(s["random_expected_reward"] for s in stats))
    return dict(splits=strata, paired_profile_preference_reversals=contrasts)


def adjudicate(task, responses, phase):
    available = [grade(task, submit(c["id"]), phase=phase) for c in task["candidates"]]
    oracle = max(g.reward for g in available)
    graded = [grade(task, r["response"], clean=r.get("clean", True), phase=phase) for r in responses]
    if not graded:
        raise ValueError("every method requires at least one response for every selected task")
    rows = [dict(asdict(g), invalid=g.error is not None, regret=oracle-g.reward) for g in graded]
    scores = {name: statistics.mean(float(r[name]) for r in rows) for name in ("reward", *METRICS)}
    failures = Counter()
    for row in rows:
        if row["error"]:
            causes = {row["error"]}
        else:
            outcome = task["outcomes"][str(row["starter_id"])]
            causes = {f["constraint"] for f in outcome["failures"]}
            if not causes and not row["final_pass"]:
                causes = {"yield_below_target"}
        failures.update(causes)
    return dict(scores=scores, attempts=rows, failure_counts=dict(failures))


def features(candidate, brief):
    focal, fno, track, gaps, pupil = spec_terms(candidate["initial"], brief)
    m = candidate["initial"]
    probes = [1. if value is None else float(np.mean(np.abs(value["mtf_delta"])))
              for value in candidate["sensitivity"].values()]
    return [focal, fno, track, float(np.mean(gaps)), float(np.mean(pupil)),
            statistics.mean(t["mtf_fraction"] for t in m["metrics"]["mtf"]),
            m["minimum_glass_thickness_mm"], m["minimum_air_clearance_mm"], *probes]


def ranker(tasks):
    """Ridge regression on within-brief centered features and training rewards."""
    if not tasks or any(t["split"] != "train" for t in tasks):
        raise ValueError("the ranker requires a train-only calibration set")
    batch, targets = [], []
    for task in tasks:
        brief = Requirements.model_validate(task["brief"])
        x = np.array([features(c, brief) for c in task["candidates"]])
        y = np.array([grade(task, submit(c["id"]), phase="training").reward for c in task["candidates"]])
        batch.extend(x-x.mean(axis=0))
        targets.extend(y-y.mean())
    x, y = np.array(batch), np.array(targets)
    scale = np.maximum(x.std(axis=0), 1e-8)
    x = x / scale
    coefficient = np.linalg.solve(x.T @ x + 4*np.eye(x.shape[1]), x.T @ y) / scale
    return dict(coefficients=coefficient.tolist(), ridge=4., train_task_ids=[t["task_id"] for t in tasks])


def response_map(record, tasks, *, name):
    if record.get("model") is None:
        raise ValueError(f"{name} has no model identity")
    by_task = {task["task_id"]: task for task in tasks}
    result = {}
    for row in record["responses"]:
        task = by_task.get(row["task_id"])
        if task is None:
            continue
        offset = row.get("position_offset", 0)
        if offset or record.get("mode") == "exploration":
            raise ValueError(f"{name} must use the fixed evaluation prompt order and settings")
        prompt = render(task, row["representation"], format_example=row.get("format_example", False))
        if row.get("prompt_sha256") != digest(prompt) or row.get("task_sha256") != digest(task):
            raise ValueError(f"{name} prompt or task did not match the frozen evaluation")
        result.setdefault(row["task_id"], []).append(row)
    return result


def paired_interval(values, clusters, seed):
    """Resample independent ancestry clusters; equal task weight within a draw."""
    grouped = defaultdict(list)
    for value, cluster in zip(values, clusters, strict=True):
        grouped[cluster].append(value)
    point = float(np.mean(values))
    units = list(grouped.values())
    if len(units) < 2:
        return dict(difference=point, ci95=None, independent_clusters=len(units))
    rng = random.Random(seed)
    bootstrap = []
    for _ in range(4000):
        drawn = [v for g in rng.choices(units, k=len(units)) for v in g]
        bootstrap.append(statistics.mean(drawn))
    return dict(difference=point, ci95=np.percentile(bootstrap, [2.5, 97.5]).tolist(), independent_clusters=len(units))


def evaluate(tasks, *, phase, train_tasks=None, predictions=None, paired_reference="closest_to_spec", seed=17291):
    if not tasks or phase not in ("development", "final"):
        raise ValueError("evaluation requires tasks and a held-out draw phase")
    if any(t["split"] != ("test" if phase == "final" else "development") for t in tasks):
        raise ValueError("final uses test tasks; checkpoint selection uses development tasks")
    if predictions and set(predictions) & {"uniform_random", "closest_to_spec", "physics_heuristic", "oracle", "ridge_ranker"}:
        raise ValueError("prediction labels must not replace predeclared controls")
    control = ranker(train_tasks) if train_tasks else None
    responses = {name: response_map(record, tasks, name=name) for name, record in (predictions or {}).items()}
    rows = {}
    for task in tasks:
        brief = Requirements.model_validate(task["brief"])
        candidates = task["candidates"]
        order = lambda fn: sorted(candidates, key=lambda c: (fn(c), c["identity"]))
        oracle = order(lambda c: -grade(task, submit(c["id"]), phase=phase).reward)[0]
        analog = {
            "uniform_random": candidates,
            "closest_to_spec": order(lambda c: heuristic(c, brief))[:1],
            "physics_heuristic": order(lambda c: heuristic(c, brief, physics=True))[:1],
            "oracle": [oracle],
        }
        if control is not None:
            analog["ridge_ranker"] = order(lambda c: -float(np.dot(control["coefficients"], features(c, brief))))[:1]
        methods = {name: [{"response": submit(c["id"])} for c in choices] for name, choices in analog.items()}
        methods.update({name: by_task.get(task["task_id"], []) for name, by_task in responses.items()})
        rows[task["task_id"]] = {name: adjudicate(task, replies, phase) for name, replies in methods.items()}
    methods = list(next(iter(rows.values())))
    if paired_reference not in methods:
        raise ValueError("paired reference method absent from evaluation")

    def summary(subset):
        selected = [rows[t["task_id"]] for t in subset]
        # Multiple briefs and variant cohorts can share one optical ancestor.
        cluster = ["|".join(sorted({c["prescription"]["ancestry"] for c in t["candidates"]})) for t in subset]
        result = {}
        for name in methods:
            scores = [m[name]["scores"] for m in selected]
            failures = Counter()
            for row in selected:
                # Uniform random and repeated rollouts each have total task weight one.
                for cause, count in row[name]["failure_counts"].items():
                    failures[cause] += count/len(row[name]["attempts"])
            reference = [m[paired_reference]["scores"] for m in selected]
            result[name] = dict(
                mean={key: statistics.mean(r[key] for r in scores) for key in ("reward", *METRICS)},
                failure_fraction={cause: count/len(subset) for cause, count in failures.items()},
                paired_vs_reference={key: paired_interval(
                    [a[key]-b[key] for a, b in zip(scores, reference, strict=True)], cluster, seed)
                    for key in METRICS})
        return dict(tasks=len(subset), ancestry_clusters=len(set(cluster)), methods=result)

    families = sorted({f for t in tasks for f in t["construction"]["families"]})
    return dict(phase=phase, paired_reference=paired_reference, ranker=control, overall=summary(tasks),
                per_family={f: summary([t for t in tasks if f in t["construction"]["families"]]) for f in families},
                task_results=rows,
                prediction_sources={k: {"model": v["model"], "checkpoint": v.get("checkpoint"), "sha256": digest(v)}
                                    for k, v in (predictions or {}).items()})

"""Offline evidence for the V2.3 development panel and matched schedule."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from itertools import combinations
from pathlib import Path
import random
from statistics import mean, pvariance

from lux_frontier.tinker_backend import load_renderer
from lux_frontier.v2.benchmark import load_envelope, load_replay_details, load_tasks, render, write_json
from lux_frontier.v2.curriculum import optical_baselines, schedule
from lux_frontier.v2.grading import digest, grade, submit
from lux_frontier.v2.manufacturing import paired_yield
from lux_frontier.v2.optics import Lens, Requirements, failures, measurements
from lux_frontier.v2.rl import Settings, estimate, seed_for, training_plan
from lux_frontier.v2.validation import ranking_convergence


def task_evidence(path, task):
    construction = task["construction"]
    phase = "training" if task["split"] == "train" else "development"
    replay = load_replay_details(path, task)
    grades = {c["id"]: grade(task, submit(c["id"]), phase=phase) for c in task["candidates"]}
    rewards = [g.reward for g in grades.values()]
    resolutions = Counter()
    paired_streams = {}
    boundary_label_changes = Counter()
    boundary_deltas = []
    thresholds = [t["minimum_mtf_fraction"] for t in task["brief"]["mtf_thresholds"]]

    def inspect(kind, measured):
        if not measured:
            return
        trace = measured["trace"]
        resolutions[f"{kind}:{trace['pupil_samples']}"] += 1
        check = trace.get("boundary_check")
        if check:
            fine = [m["mtf_fraction"] for m in measured["metrics"]["mtf"]]
            coarse = check["screen_mtf"]
            for old, new, bound in zip(coarse, fine, thresholds, strict=True):
                if (old >= bound) != (new >= bound):
                    boundary_label_changes[kind] += 1
                boundary_deltas.append(abs(new-old))

    for key, data in replay.items():
        inspect("nominal", task["outcomes"][key]["final"])
        for purpose, builds in data["builds"].items():
            stream = [b["stream"] for b in builds]
            if purpose in paired_streams:
                assert stream == paired_streams[purpose]
            paired_streams[purpose] = stream
            for build in builds:
                inspect(purpose, build["measurements"])
    if paired_streams:
        assert {tuple(s) for s in paired_streams["construction"]}.isdisjoint(
            tuple(s) for s in paired_streams[phase])

    contrast = []
    target = task["yield_target"]
    for left, right in combinations(grades, 2):
        a, b = grades[left], grades[right]
        if not a.nominal_pass or not b.nominal_pass:
            continue
        lo, hi = sorted((left, right), key=lambda i: grades[i].yield_fraction)
        lo_grade, hi_grade = grades[lo], grades[hi]
        if hi_grade.yield_fraction-lo_grade.yield_fraction < .25:
            continue
        low = task["outcomes"][str(lo)]["simulations"][phase]
        high = task["outcomes"][str(hi)]["simulations"][phase]
        comparison = paired_yield(replay[str(lo)]["builds"][phase], replay[str(hi)]["builds"][phase])
        contrast.append(dict(ids=[lo, hi], passed=[low["passed"], high["passed"]], tested=low["tested"],
            gap=hi_grade.yield_fraction-lo_grade.yield_fraction, straddles_target=lo_grade.yield_fraction < target <= hi_grade.yield_fraction,
            intervals=[low.get("interval95"), high.get("interval95")],
            reverses_nominal_sharpness=construction["nominal_quality_order"].index(lo) <
                                       construction["nominal_quality_order"].index(hi),
            **comparison))
    return dict(id=task["task_id"], group=task["group"], profile=task["profile"]["name"],
        lineage=task["candidates"][0]["prescription"]["ancestry"], capability=task["capability"],
        families=construction["families"], nominal_candidates=sum(g.nominal_pass for g in grades.values()),
        passing_candidates=sum(g.final_pass for g in grades.values()), reward_range=max(rewards)-min(rewards),
        uniform_reward_variance=pvariance(rewards), best_reward=max(rewards),
        physics_regret=max(rewards)-grades[construction["physics_id"]].reward,
        resolution_counts=dict(resolutions), changed_mtf_decisions=dict(boundary_label_changes),
        maximum_boundary_delta=max(boundary_deltas, default=0.), contrasts=sorted(contrast, key=lambda c: -c["gap"]),
        **optical_baselines(task, phase))


def matched_schedule(training, development, train_settings):
    _, renderer = load_renderer()

    def tokens(task, candidates):
        prompt = render(dict(task, candidates=candidates), train_settings.representation,
                        format_example=train_settings.format_example)
        return len(renderer.build_generation_prompt([{"role": "user", "content": prompt}]).to_ints())

    steps = []
    for step in range(train_settings.steps):
        attempt_order, allocation = schedule(training, train_settings.mixture, [], seed=train_settings.seed,
                                             step=step, attempts=train_settings.max_attempts_per_step)
        rows = []
        for index, (task, family) in enumerate(attempt_order):
            candidates = list(task["candidates"])
            presentation = seed_for(train_settings.seed, "training-order", step, index, task["task_id"])
            random.Random(presentation).shuffle(candidates)
            rows.append(dict(id=task["task_id"], family=family, seed=presentation,
                             order=[c["id"] for c in candidates], prompt_tokens=tokens(task, candidates)))
        steps.append(dict(step=step+1, allocation=allocation["weights"], attempts=rows))
    evaluation_tokens = {}
    for task in development:
        c = task["candidates"]
        evaluation_tokens[task["task_id"]] = {str(i): tokens(task, c[i:]+c[:i]) for i in train_settings.position_offsets}
    assert max(row["prompt_tokens"] for step in steps for row in step["attempts"]) <= train_settings.max_prompt_tokens
    assert max(n for offsets in evaluation_tokens.values() for n in offsets.values()) <= train_settings.max_prompt_tokens
    return dict(steps=steps, evaluation_tokens=evaluation_tokens)


def check_pupil(task):
    brief = Requirements.model_validate(task["brief"])
    rows = []
    for key, outcome in task["outcomes"].items():
        design = Lens.model_validate(outcome["refined_prescription"])
        canonical = outcome["final"]
        grids = {canonical["trace"]["pupil_samples"]: canonical}
        for samples in (128, 256):
            if samples not in grids:
                grids[samples] = measurements(design, brief, samples)
        points = [dict(field_deg=t.field_angle_deg, wavelength_um=t.wavelength_um, orientation=t.orientation,
                       frequency=t.spatial_frequency_lp_per_mm, threshold=t.minimum_mtf_fraction,
                       mtf128=a["mtf_fraction"], mtf256=b["mtf_fraction"])
                  for t, a, b in zip(brief.mtf_thresholds, grids[128]["metrics"]["mtf"],
                                      grids[256]["metrics"]["mtf"], strict=True)]
        deltas = [abs(p["mtf128"]-p["mtf256"]) for p in points]
        if max(deltas) > .03:
            higher = measurements(design, brief, 512)
            for p, m in zip(points, higher["metrics"]["mtf"], strict=True):
                p["mtf512"] = m["mtf_fraction"]
        rows.append(dict(id=key, points=points, max_delta=max(deltas),
                         canonical_samples=canonical["trace"]["pupil_samples"],
                         nominal_pass128=not failures(grids[128], brief), nominal_pass256=not failures(grids[256], brief)))
    return dict(task_id=task["task_id"], candidates=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    audits = parser.add_mutually_exclusive_group()
    audits.add_argument("--optimizer-workers", type=int, help="audit the representative fixed-process briefs separately")
    audits.add_argument("--pupil-workers", type=int, help="audit all unique development nominal fits at both pupil grids")
    args = parser.parse_args()
    if args.optimizer_workers is not None:
        manifest = load_envelope(args.manifest, "manifest")
        representative = {row["representative"] for row in manifest["development_panel"]["selection"].values()}
        tasks = [t for t in load_tasks(args.manifest, split="development")
                 if t["group"] in representative and t["profile"]["name"] == "standard-fixed"]
        with ProcessPoolExecutor(max_workers=args.optimizer_workers) as pool:
            rows = list(pool.map(partial(ranking_convergence, pupil_grids=(128, 256), budgets=(49,)), tasks))
        write_json(args.output, dict(manifest_sha256=digest(manifest), reference_tasks=rows))
        return
    if args.pupil_workers is not None:
        tasks = [t for t in load_tasks(args.manifest, split="development") if t["profile"]["name"] == "standard-fixed"]
        with ProcessPoolExecutor(max_workers=args.pupil_workers) as pool:
            rows = list(pool.map(check_pupil, tasks))
        write_json(args.output, dict(manifest_sha256=digest(load_envelope(args.manifest, "manifest")), pupil_checks=rows))
        return
    root = Path(__file__).parent
    settings = Settings.model_validate_json((root/"v23-matched.json").read_text())
    evaluation = Settings.model_validate_json((root/"v23-evaluation.json").read_text())
    manifest = load_envelope(args.manifest, "manifest")
    assert not any(row["split"] == "test" for row in manifest["task_index"])
    train, dev = (load_tasks(args.manifest, split=split) for split in ("train", "development"))
    rows = {split: [task_evidence(args.manifest, t) for t in tasks]
            for split, tasks in (("train", train), ("development", dev))}
    means = {split: {field: mean(row[field] for row in results) for field in
                     ("random_pass", "oracle_pass", "closest_pass", "physics_pass", "reward_range")}
             for split, results in rows.items()}
    plan = training_plan(settings, len(dev))
    evaluation_plan = estimate(evaluation, groups=0, other_samples=len(dev)*evaluation.eval_samples*len(evaluation.position_offsets))
    write_json(args.output, dict(manifest_sha256=digest(manifest), protocol=manifest["protocol"], benchmark=rows,
        mean_baselines=means, panel=manifest["development_panel"], manufacturing=manifest["manufacturing"],
        schedule=matched_schedule(train, dev, settings), training_plan_per_arm=plan, evaluation_plan_per_arm=evaluation_plan,
        total_estimate_usd=round(2*plan["budget_estimate_usd"]+3*evaluation_plan["budget_estimate_usd"], 6)))


if __name__ == "__main__":
    main()

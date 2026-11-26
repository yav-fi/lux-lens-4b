"""Resolution, refinement-budget, and independent manufacturing studies."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import fields
from itertools import combinations
import json
import math
from pathlib import Path
import statistics
from time import perf_counter, process_time

import numpy as np

from lux_frontier.optical_engine import mtf_value, normalized_field
from lux_frontier.v2.benchmark import Protocol, load_tasks, merit, provenance, refine, write_envelope
from lux_frontier.v2.grading import digest
from lux_frontier.v2.manufacturing import namespace, paired_yield, yield_record
from lux_frontier.v2.optics import Lens, Profile, Requirements, build_optic, failures, measurements, simulate_build

PUPIL_GRIDS = (32, 64, 128, 256)
BUDGETS = (7, 25, 49, 73, 97)


def metrics(measured, brief):
    mtf = [v["mtf_fraction"] for v in measured["metrics"]["mtf"]]
    margins = [v-t.minimum_mtf_fraction for v, t in zip(mtf, brief.mtf_thresholds, strict=True)]
    reasons = failures(measured, brief)
    return dict(mtf=mtf, mtf_margins=margins, nominal=not reasons, failures=[row["constraint"] for row in reasons],
                throughput=[r["pupil_fraction"] for r in measured["throughput"]], objective=merit(measured, brief)[0])


def alias_probe(design, brief, index):
    from optiland.mtf import FFTMTF
    from optiland.wavefront import Wavefront

    optic, _ = build_optic(design)
    target = brief.mtf_thresholds[index]
    field = normalized_field(design, target.field_angle_deg)
    wavelength = target.wavelength_um
    rows = {}
    for size in (32, 64, 128, 256, 512):
        wave = Wavefront(optic, fields=[field], wavelengths=[wavelength], num_rays=size, distribution="uniform",
                         strategy="chief_ray", remove_tilt=False)
        data = wave.get_data(field, wavelength)
        lattice = np.linspace(-1, 1, size)
        x, y = np.meshgrid(lattice, lattice)
        pupil = x*x+y*y <= 1
        phase = np.full((size, size), np.nan)
        phase[pupil] = np.where(data.intensity > 0, data.opd, np.nan)
        changes = np.concatenate([np.abs(np.diff(phase, axis=i)).ravel() for i in (0, 1)])
        changes = changes[np.isfinite(changes)]
        pads = (2, 4, 8) if size == 32 else (2,)
        outputs = {}
        for factor in pads:
            start = process_time()
            calculator = FFTMTF(optic, fields=[field], wavelength=wavelength, num_rays=size, grid_size=size*factor,
                                strategy="chief_ray", remove_tilt=False)
            value, details = mtf_value(calculator, 0, target.orientation, target.spatial_frequency_lp_per_mm)
            outputs[factor] = dict(mtf=value, cpu_seconds=process_time()-start, **details)
        rows[size] = dict(max_adjacent_waves=float(np.max(changes)),
                          fraction_adjacent_above_half_wave=float(np.mean(changes>.5)), pads=outputs)
    return dict(target=target.model_dump(mode="json"), grids=rows)


def grid_worker(args):
    task, c, probe = args
    brief = Requirements.model_validate(task["brief"])
    outcome = task["outcomes"][str(c["id"])]
    design = Lens.model_validate(outcome["refined_prescription"])
    rows = {}
    for size in PUPIL_GRIDS:
        start = process_time()
        values = metrics(measurements(design, brief, size), brief)
        rows[size] = dict(**values, cpu_seconds=process_time()-start)
    if probe is not None:
        extra = alias_probe(design, brief, probe)
    else:
        extra = None
    return dict(task=task["task_id"], task_sha256=digest(task), candidate=c["id"], grids=rows, alias_probe=extra)


def summarize_grids(rows):
    reference = PUPIL_GRIDS[-1]
    comparisons = {}
    # Keep the threshold subset fixed across grid comparisons. A case is
    # decision-relevant when MTF is near a cutoff and other constraints pass.
    boundary = [r for r in rows if any(abs(m)<=.03 for m in r["grids"][reference]["mtf_margins"]) and all(
        reason.startswith("mtf[") for reason in r["grids"][reference]["failures"])]
    for size in PUPIL_GRIDS:
        def difference(row):
            a, b = row["grids"][size], row["grids"][reference]
            return max(abs(x-y) for x, y in zip(a["mtf"], b["mtf"], strict=True))

        disagreements = [dict(task=r["task"], candidate=r["candidate"]) for r in rows
                         if r["grids"][size]["nominal"] != r["grids"][reference]["nominal"]]
        comparisons[size] = dict(nominal_disagreements=disagreements, maximum_mtf_delta=max(map(difference, rows)),
                                 median_max_mtf_delta=statistics.median(map(difference, rows)),
                                 max_boundary_mtf_delta=max(map(difference, boundary), default=None),
                                 median_cpu_seconds=statistics.median(r["grids"][size]["cpu_seconds"] for r in rows))
    return dict(reference_grid=reference, cases=len(rows), boundary_cases=[
        dict(task=r["task"], candidate=r["candidate"]) for r in boundary], comparisons=comparisons)


def budget_worker(args):
    task, candidate, search_grid = args
    brief = Requirements.model_validate(task["brief"])
    starter = Lens.model_validate(candidate["prescription"])
    frozen_protocol = Protocol.model_validate(task["protocol"])
    protocol = frozen_protocol.model_copy(update=dict(evaluations=BUDGETS[-1],
                                                 pupil_samples=search_grid or frozen_protocol.pupil_samples))
    start = process_time()
    result = refine(starter, brief, protocol)
    fit_cpu = process_time()-start
    outcomes = {}
    for budget in BUDGETS:
        # Budgets are odd, so their prefixes end after both arms of a trial pair.
        best = min((r for r in result["trajectory"][:budget] if r["measured"] is not None),
                   key=lambda r: (bool(r["failures"]), r["score"]))
        final = Lens.model_validate(best["prescription"])
        outcomes[budget] = dict(search=metrics(best["measured"], brief), checked=metrics(measurements(final, brief, 128), brief),
                                prescription_sha256=digest(best["prescription"]))
    frozen = task["outcomes"][str(candidate["id"])]
    if (protocol.pupil_samples == frozen_protocol.pupil_samples and frozen_protocol.evaluations in outcomes
            and outcomes[frozen_protocol.evaluations]["search"]["objective"] != frozen["objective"]):
        raise ValueError("optimizer-prefix objective differs from frozen result")
    return dict(task=task["task_id"], candidate=candidate["id"], search_grid=protocol.pupil_samples,
                check_grid=128, fit_cpu_seconds=fit_cpu, budgets=outcomes)


def summarize_budgets(rows):
    result = {}
    for budget in BUDGETS:
        scores = {r["candidate"]: r["budgets"][budget]["checked"] for r in rows}
        prior = {r["candidate"]: r["budgets"][25]["checked"] for r in rows}
        keys = sorted(scores)
        ordering = lambda results, a, b: np.sign(results[a]["objective"]-results[b]["objective"])
        agreeing = sum(bool(ordering(scores,a,b)==ordering(prior,a,b)) for a,b in combinations(keys,2))
        result[budget] = dict(checked_nominal_ids=[k for k in keys if scores[k]["nominal"]], agreeing_pairs=agreeing,
                              total_pairs=math.comb(len(keys),2),
                              search_nominal_ids=sorted(r["candidate"] for r in rows if r["budgets"][budget]["search"]["nominal"]),
                              changed_prescriptions=sum(r["budgets"][budget]["prescription_sha256"] !=
                                                        r["budgets"][25]["prescription_sha256"] for r in rows))
    return result


def factory_worker(args):
    task, candidate, first, stop, size, phase = args
    outcome = task["outcomes"][str(candidate)]
    brief = Requirements.model_validate(task["brief"])
    design = Lens.model_validate(outcome["refined_prescription"])
    profile = Profile(**{f.name: task["profile"][f.name] for f in fields(Profile)})
    prefix = namespace(280221, task["group"], phase, "definitive")
    start = process_time()
    rows = [simulate_build(design, brief, size, profile, [*prefix, i]) for i in range(first, stop)]
    return dict(candidate=candidate, size=size, rows=rows, cpu_seconds=process_time()-start)


def factory(tasks, pool):
    name = "train-cohort000-contrast-spec006-standard-adjusted"
    task = next(t for t in tasks if t["task_id"] == name)
    pair = (6, 4)
    brief = Requirements.model_validate(task["brief"])
    check = {c: {size: metrics(measurements(Lens.model_validate(task["outcomes"][str(c)]["refined_prescription"]),
                                            brief, size), brief) for size in (32, 128, 256)} for c in pair}
    phase = "v22-independent-confirmation"
    jobs = [(task, c, first, first+32, size, phase) for c in pair
            for size, count in ((128, 128), (32, 32), (256, 32)) for first in range(0, count, 32)]
    batches = list(pool.map(factory_worker, jobs))
    def select(candidate, size):
        return [row for batch in batches if (batch["candidate"], batch["size"]) == (candidate, size) for row in batch["rows"]]

    primary = {c: select(c,128) for c in pair}
    by_size = {size: {c: sum(r["passed"] for r in select(c,size)[:32]) for c in pair} for size in (32,128,256)}
    checks = {size: {c: paired_yield(select(c,128)[:32], select(c,size)) for c in pair} for size in (32,256)}
    return dict(task=name, task_sha256=digest(task), pair=pair, nominal=check, primary_pupil_grid=128,
                yields={c: yield_record(primary[c], task["yield_target"], "definitive") for c in pair},
                paired=paired_yield(*(primary[c] for c in pair)), same_draw_first_32_by_grid=by_size,
                grid_discordance=checks,
                failure_counts={c: dict(Counter(r["constraint"] for b in primary[c] for r in b["failures"])) for c in pair},
                batches=batches)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", choices=("grids", "budgets", "factory"))
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--task", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--search-pupil", type=int, choices=PUPIL_GRIDS, help="override the refinement grid for the budget study")
    parser.add_argument("--candidate", action="append", default=[], metavar="TASK:ID", help="limit a budget study to specified candidates")
    args = parser.parse_args()
    if args.workers<1 or args.output.exists():
        raise ValueError("positive worker count and unused output path required")
    tasks = [t for t in load_tasks(args.manifest) if t["task_id"] in args.task]
    if set(args.task) != {t["task_id"] for t in tasks}:
        raise ValueError("a requested study task was absent")
    selected = {(task_id, int(candidate)) for task_id, candidate in (v.rsplit(":",1) for v in args.candidate)}
    available = {(t["task_id"], c["id"]) for t in tasks for c in t["candidates"]}
    if not selected <= available or ((selected or args.search_pupil) and args.study != "budgets"):
        raise ValueError("candidate and refinement overrides must reference requested budget-study cases")
    started = perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        if args.study == "grids":
            jobs = [(t,c,1 if t["task_id"] == "development-cohort000-spec003-standard-adjusted" and c["id"] == 3 else None)
                    for t in tasks for c in t["candidates"]]
            rows = list(pool.map(grid_worker, jobs))
            data = dict(rows=rows, summary=summarize_grids(rows))
        elif args.study == "budgets":
            rows = list(pool.map(budget_worker, ((t,c,args.search_pupil) for t in tasks for c in t["candidates"]
                                                 if not selected or (t["task_id"],c["id"]) in selected)))
            data = dict(rows=rows, summary={t["task_id"]: summarize_budgets([r for r in rows if r["task"] == t["task_id"]])
                                            for t in tasks if any(r["task"]==t["task_id"] for r in rows)})
        else:
            data = factory(tasks, pool)
    data.update(wall_seconds=perf_counter()-started, workers=args.workers, provenance=provenance())
    # Normalize numeric grid and candidate keys to their JSON representation
    # before digesting. Lexical and integer key ordering otherwise differ.
    write_envelope(args.output, json.loads(json.dumps(data, allow_nan=False)), args.study)
    print(f"{args.study}: {data['wall_seconds']:.2f} s, {args.output}", flush=True)


if __name__ == "__main__":
    main()

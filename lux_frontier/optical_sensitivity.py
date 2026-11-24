"""Offline paired refocus/ablation study; original optical protocol is immutable."""

import argparse
import copy
import json
import multiprocessing
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from uuid import uuid4

import numpy as np

from lux_frontier.contracts import NominalMetrics
from lux_frontier.optical_diagnostic import sha_file, source_manifest, write_json
from lux_frontier.optical_engine import BuildErrors, InvalidBuild, constraint_failures, measure
from lux_frontier.optical_grading import OpticalCache, OpticalGrader, build_draw, load_recorded_case
from lux_frontier.optical_prescriptions import ArtifactStore, Prescription, digest

STUDY_PROTOCOL = "cooke-refocus-sensitivity-v1"
DIAGNOSTIC = Path("outputs/optical_diagnostics/20261123T194942Z-29c360f3/run.json")
FAMILIES = ("radius_fraction", "center_thickness_mm", "air_gap_mm", "decenter_mm", "tilt_deg")
CONFIRMATION_SEED = 2026120101  # reserved development stream; not a final study test
MATERIAL_GAIN = 10


def nested_focus_grid(bound: float, count: int) -> list[float]:
    if count not in (9, 33):
        raise ValueError("sensitivity protocol supports only original 9 or nested 33 positions")
    original = np.linspace(-bound, bound, 9).tolist()
    positions = np.linspace(-bound, bound, count).tolist()
    if count == 33:
        positions[::4] = original  # retain exact original floating-point positions
    return sorted(set(positions), key=lambda value: (abs(value), value))


def intervene(draw: dict, family: str) -> dict:
    if family not in (*FAMILIES, "all"):
        raise ValueError("unsupported diagnostic intervention")
    changed = copy.deepcopy(draw)
    zeros = BuildErrors()
    for name in FAMILIES if family == "all" else (family,):
        changed["errors"][name] = json.loads(json.dumps(getattr(zeros, name)))
    changed["diagnostic_intervention"] = family
    changed["source_draw_sha256"] = digest(draw)
    return changed


@lru_cache(maxsize=4)
def load_study_source(diagnostic_path: Path):
    diagnostic = json.loads(diagnostic_path.read_text())
    source, case, profile, settings = load_recorded_case(Path(diagnostic["source_record"]))
    if (settings.refocus_positions, settings.refinement_evaluations, settings.evaluation_builds,
            case.brief.minimum_simulated_yield_fraction) != (9, 37, 100, 0.9):
        raise ValueError("study requires the preserved 9-position, 37-evaluation, 100-build, 90% development protocol")
    row = next(r for r in diagnostic["starters"] if r["candidate_id"] == 7)
    p = Prescription.model_validate(row["refinement"]["refined_prescription"])
    if row["evaluation"]["builds_passed"] != 51 or p.reference.model_dump(mode="json") != source["result"]["shipped_prescription_ref"]:
        raise ValueError("diagnostic does not contain the original starter-7 result")
    return source, case, profile, settings, row, p


def evaluate_focus(p: Prescription, brief, profile, settings, draw: dict, positions: list[float],
                   cache: OpticalCache, *, saved_build: dict | None = None) -> dict:
    """Every trial measures all fields at ONE physical sensor plane.

    A finer-grid replay may reuse all original focus scores and the selected
    original measurement. No optical state is inferred for an unevaluated point.
    """
    if saved_build is not None and digest(saved_build["draw"]) != digest(draw):
        raise ValueError("saved focus trials do not belong to the requested realized errors")
    if saved_build is not None and saved_build["prescription_ref"] != p.reference.model_dump(mode="json"):
        raise ValueError("saved focus trials belong to a different prescription")
    if (saved_build is not None and saved_build.get("details") is not None
            and saved_build["details"]["pupil_samples"] != settings.pupil_samples):
        raise ValueError("saved focus trials use a different pupil sampling protocol")
    original_trials = {t["sensor_shift_mm"]: t for t in saved_build["refocus_trials"]} if saved_build else {}
    trials, candidates = [], []
    # Preserve the entire original winner; a strict higher minimum-MTF score is
    # required to replace it unless the specified focus-order tie-break wins.
    if saved_build is not None and saved_build["metrics"] is not None:
        m = NominalMetrics.model_validate(saved_build["metrics"])
        score = min(a.mtf_fraction / b.minimum_mtf_fraction for a, b in zip(m.mtf, brief.mtf_thresholds, strict=True))
        candidates.append((score, saved_build["details"]["sensor_shift_mm"],
                           dict(metrics=saved_build["metrics"], details=saved_build["details"], failures=saved_build["failures"])))
    metric_misses = 0
    for shift in positions:
        if shift in original_trials:
            trials.append(dict(original_trials[shift], reused_original_trial=True))
            continue
        inputs = dict(study_protocol=STUDY_PROTOCOL, optical_protocol=p.protocol,
                      prescription=p.model_dump(mode="json"), brief=brief.model_dump(mode="json"),
                      profile=profile.model_dump(mode="json"), settings=settings.model_dump(mode="json"),
                      realized_errors=draw["errors"], source_seed=draw["seed"], sensor_shift_mm=shift,
                      numpy_version=np.__version__, optiland_version="0.6.3")

        def compute():
            try:
                metrics, details = measure(p, brief, pupil_samples=settings.pupil_samples,
                                           errors=BuildErrors(**draw["errors"]), sensor_shift_mm=shift)
                score = min(a.mtf_fraction / b.minimum_mtf_fraction for a, b in zip(metrics.mtf, brief.mtf_thresholds, strict=True))
                return dict(status="measured", metrics=metrics.model_dump(mode="json"), details=details,
                            minimum_mtf_ratio=score, failures=constraint_failures(metrics, brief))
            except InvalidBuild as exc:
                return dict(status="physically_invalid", failure=str(exc))

        value, hit = cache.compute("focus_measurement", inputs, compute)
        metric_misses += not hit
        if value["status"] == "measured":
            score = value["minimum_mtf_ratio"]
            trials.append(dict(sensor_shift_mm=shift, minimum_mtf_ratio=score, reused_original_trial=False))
            candidates.append((score, shift, value))
        else:
            trials.append(dict(sensor_shift_mm=shift, physically_invalid=value["failure"], reused_original_trial=False))
    if not candidates:
        result = dict(status="physically_invalid", metrics=None, details=None,
                      failures=[dict(constraint=next(t["physically_invalid"] for t in trials))])
    else:
        _, _, selected = min(candidates, key=lambda c: (-c[0], abs(c[1]), c[1]))
        result = dict(status="measured", metrics=selected["metrics"], details=selected["details"], failures=selected["failures"])
    return dict(result, passed=not result["failures"], draw=draw, refocus_trials=trials,
                prescription_ref=p.reference.model_dump(mode="json"),
                work=dict(focus_positions=len(positions), original_trials_reused=sum(t["reused_original_trial"] for t in trials),
                          new_metric_evaluations=metric_misses))


def study_worker(payload: dict) -> dict:
    source, case, profile, settings, row, p = load_study_source(Path(payload["diagnostic"]))
    index, experiment = payload["index"], payload["experiment"]
    saved = row["evaluation"]["builds"][index]
    cache = OpticalCache(Path(payload["cache_directory"]))
    if experiment == "fine33":
        draw, saved_build, count = saved["draw"], saved, 33
    else:
        draw, saved_build, count = intervene(saved["draw"], experiment), None, 9
    result = evaluate_focus(p, case.brief, profile, settings, draw, nested_focus_grid(0.2, count), cache, saved_build=saved_build)
    path = Path(payload["record_path"])
    write_json(path, result)
    return dict(experiment=experiment, index=index, path=str(path), passed=result["passed"])


def build_summary(builds: list[dict]) -> dict:
    return dict(builds_passed=sum(b["passed"] for b in builds), builds_tested=len(builds),
                failures_by_constraint=dict(Counter(f["constraint"] for b in builds for f in b["failures"])),
                physically_invalid=sum(b["status"] == "physically_invalid" for b in builds),
                lower_travel_limit=sum(b.get("details") is not None and abs(b["details"]["sensor_shift_mm"] + 0.2) < 1e-12 for b in builds),
                upper_travel_limit=sum(b.get("details") is not None and abs(b["details"]["sensor_shift_mm"] - 0.2) < 1e-12 for b in builds))


def compare_builds(before: list[dict], after: list[dict]) -> dict:
    switches = Counter()
    repaired, introduced = Counter(), Counter()
    per_build = []
    for i, (a, b) in enumerate(zip(before, after, strict=True)):
        label = "both_passed" if a["passed"] and b["passed"] else "fail_to_pass" if b["passed"] else "pass_to_fail" if a["passed"] else "both_failed"
        switches[label] += 1
        a_keys, b_keys = {f["constraint"] for f in a["failures"]}, {f["constraint"] for f in b["failures"]}
        repaired.update(a_keys - b_keys)
        introduced.update(b_keys - a_keys)
        per_build.append(dict(index=i, transition=label, repaired_constraints=sorted(a_keys - b_keys),
                              introduced_constraints=sorted(b_keys - a_keys)))
    return dict(transitions={k: switches[k] for k in ("both_passed", "fail_to_pass", "pass_to_fail", "both_failed")},
                constraint_repairs=dict(repaired), constraint_introductions=dict(introduced), builds=per_build)


def execute_jobs(payloads: list[dict], workers: int):
    if workers == 1:
        for payload in payloads:
            yield study_worker(payload)
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as executor:
            for future in as_completed([executor.submit(study_worker, p) for p in payloads]):
                yield future.result()


def diagnose(diagnostic: Path, output_root: Path, workers: int) -> tuple[Path, dict]:
    started = time.perf_counter()
    source, case, profile, settings, row, p = load_study_source(diagnostic)
    original_root = Path(source["cache_directory"]).parent
    manifest = source_manifest(original_root)
    directory = output_root / f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-diagnosis-{uuid4().hex[:8]}"
    directory.mkdir(parents=True)
    cache_directory = output_root / "cache"
    reproduced = OpticalGrader(ArtifactStore(Path(source["artifact_directory"])), OpticalCache(Path(source["cache_directory"])),
                               settings, profile).simulate(p, case.brief, "evaluation", 100)
    if reproduced["builds_passed"] != 51:
        raise RuntimeError("original 51/100 result did not reproduce")
    for a, b in zip(reproduced["builds"], row["evaluation"]["builds"], strict=True):
        if any(a[key] != b[key] for key in ("metrics", "failures", "draw", "refocus_trials")):
            raise RuntimeError("original build replay changed")
    write_json(directory / "original-reproduction.json", reproduced)
    write_json(directory / "study-plan.json", dict(protocol=STUDY_PROTOCOL, source_diagnostic=str(diagnostic),
               source_sha256=sha_file(diagnostic), original_positions=nested_focus_grid(0.2, 9),
               nested_positions=nested_focus_grid(0.2, 33), material_gain_builds=MATERIAL_GAIN,
               families=list(FAMILIES), confirmation_seed_reserved=CONFIRMATION_SEED,
               final_study_test_set_accessed=False, requirements_and_profile_unchanged=True))
    print(json.dumps(dict(reproduced="51/100", directory=str(directory))), flush=True)
    experiments = ("fine33", *FAMILIES, "all")
    payloads = [dict(diagnostic=str(diagnostic), cache_directory=str(cache_directory), experiment=experiment, index=i,
                     record_path=str(directory / experiment / f"{i:03d}.json")) for experiment in experiments for i in range(100)]
    completed = Counter()
    for item in execute_jobs(payloads, workers):
        completed[item["experiment"]] += 1
        if completed[item["experiment"]] % 20 == 0:
            print(json.dumps(dict(experiment=item["experiment"], completed=completed[item["experiment"]])), flush=True)
    results = {name: [json.loads((directory / name / f"{i:03d}.json").read_text()) for i in range(100)] for name in experiments}
    zero_metrics, zero_details = measure(p, case.brief, pupil_samples=settings.pupil_samples, errors=BuildErrors(), sensor_shift_mm=0)
    if zero_metrics.model_dump(mode="json") != row["refinement"]["after"]["metrics"]:
        raise RuntimeError("zero errors at nominal sensor failed to reproduce nominal performance")
    if any(sha_file(Path(path)) != expected for path, expected in manifest.items()):
        raise RuntimeError("preserved optical records changed")
    summaries = {name: build_summary(builds) for name, builds in results.items()}
    result = dict(protocol=STUDY_PROTOCOL, kind="development_diagnosis", diagnostic=str(diagnostic),
                  source_record=source, prescription=p.model_dump(mode="json"), settings=settings.model_dump(mode="json"),
                  profile=profile.model_dump(mode="json"), original=build_summary(reproduced["builds"]),
                  experiments=summaries, fine33_comparison=compare_builds(reproduced["builds"], results["fine33"]),
                  zero_error_control=dict(reproduces_nominal=True, metrics=zero_metrics.model_dump(mode="json"), details=zero_details),
                  recommended_alternative="fine33" if summaries["fine33"]["builds_passed"] - 51 >= MATERIAL_GAIN else "margin-v1",
                  cache_directory=str(cache_directory), original_files_verified_unchanged=len(manifest),
                  final_study_test_set_accessed=False, elapsed_seconds=time.perf_counter() - started)
    path = directory / "run.json"
    write_json(path, result)
    print(json.dumps(dict(record=str(path), original=result["original"], experiments=summaries,
                          recommended_alternative=result["recommended_alternative"], elapsed_seconds=result["elapsed_seconds"]), indent=2), flush=True)
    return path, result


class MarginGrader(OpticalGrader):
    """One predeclared alternative merit; same starting lens, bounds and 37 calls."""

    def context(self, p, brief):
        return dict(super().context(p, brief), refinement_merit_protocol="cooke-extra-margin-v1",
                    merit="original merit + sum(max(0,1.25-MTF/required_minimum)^2)")

    def nominal_objective(self, metrics, brief):
        return super().nominal_objective(metrics, brief) + sum(
            max(0, 1.25 - m.mtf_fraction / t.minimum_mtf_fraction) ** 2
            for m, t in zip(metrics.mtf, brief.mtf_thresholds, strict=True))


def confirmation_worker(payload: dict) -> dict:
    _, case, profile, original_settings, _, baseline = load_study_source(Path(payload["diagnostic"]))
    settings = original_settings.model_copy(update={"evaluation_seed": payload["seed"]})
    draw = build_draw(profile, settings, "evaluation", payload["index"])
    cache = OpticalCache(Path(payload["cache_directory"]))
    before = evaluate_focus(baseline, case.brief, profile, settings, draw, nested_focus_grid(0.2, 9), cache)
    alternative = Prescription.model_validate(payload["alternative_prescription"])
    fine = payload["alternative"] == "fine33"
    after = evaluate_focus(alternative, case.brief, profile, settings, draw,
                           nested_focus_grid(0.2, 33 if fine else 9), cache,
                           saved_build=before if fine else None)
    result = dict(index=payload["index"], role="new_development_confirmation", draw=draw,
                  original9=before, alternative=after)
    write_json(Path(payload["record_path"]), result)
    return result


def confirm(diagnosis_path: Path, output_root: Path, workers: int) -> tuple[Path, dict]:
    started = time.perf_counter()
    diagnosis = json.loads(diagnosis_path.read_text())
    diagnostic = Path(diagnosis["diagnostic"])
    source, case, profile, settings, row, baseline = load_study_source(diagnostic)
    alternative_name = diagnosis["recommended_alternative"]
    directory = output_root / f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-confirmation-{uuid4().hex[:8]}"
    directory.mkdir(parents=True)
    cache_directory = output_root / "cache"
    if alternative_name == "fine33":
        reason = (f"The nested grid gained {diagnosis['experiments']['fine33']['builds_passed'] - 51} passes on identical development errors. "
                  "Test the same refined starter with 12.5 um axial focus steps over unchanged +/-0.2 mm travel; "
                  "review whether assembly refocus supports that placement resolution and repeatability.")
    else:
        reason = ("Grid refinement gained fewer than ten passes. Test one nominal merit rewarding 25% extra MTF margin, "
                  "using the original initial starter 7, identical parameter bounds and exactly 37 objective evaluations.")
    proposal = dict(study_protocol=STUDY_PROTOCOL, alternative=alternative_name, reason=reason,
                    fixed_target=0.9, profile_sha256=digest(profile), requirements_sha256=digest(case.brief),
                    original_optimization_budget=37, alternative_optimization_budget=37,
                    confirmation_seed=CONFIRMATION_SEED, diagnosis_sha256=sha_file(diagnosis_path),
                    final_study_test_set_accessed=False)
    write_json(directory / "choice-proposal.json", proposal)
    print(json.dumps(proposal, indent=2), flush=True)
    refinement = None
    if alternative_name == "fine33":
        alternative = baseline
    elif alternative_name == "margin-v1":
        store = ArtifactStore(Path(source["artifact_directory"]))
        initial = store.resolve(next(c.prescription_ref for c in case.candidates if c.candidate_id == 7))
        refinement = MarginGrader(store, OpticalCache(cache_directory), settings, profile).refine(initial, case.brief)
        alternative = Prescription.model_validate(refinement["refined_prescription"])
    else:
        raise ValueError("unsupported predeclared alternative")
    nominal, nominal_details = measure(alternative, case.brief, pupil_samples=settings.pupil_samples)
    plan = dict(proposal, frozen_before_confirmation_draws=True,
                original_prescription=baseline.model_dump(mode="json"), alternative_prescription=alternative.model_dump(mode="json"),
                alternative_reference=alternative.reference.model_dump(mode="json"),
                original_focus_positions=nested_focus_grid(0.2, 9),
                alternative_focus_positions=nested_focus_grid(0.2, 33 if alternative_name == "fine33" else 9),
                alternative_nominal_metrics=nominal.model_dump(mode="json"), alternative_nominal_details=nominal_details,
                alternative_nominal_failures=constraint_failures(nominal, case.brief), refinement=refinement,
                refinement_source=("exact original 37-evaluation refinement reused; no new optimization" if alternative_name == "fine33"
                                   else "original initial starter independently refined with the same 37-evaluation budget"),
                role="new_development_confirmation", final_study_test_set_accessed=False)
    write_json(directory / "confirmation-plan.json", plan)
    # Only after the exact design, protocol and seed are frozen do we generate
    # any confirmation draws. This namespace is explicitly development data.
    confirm_settings = settings.model_copy(update={"evaluation_seed": CONFIRMATION_SEED})
    draws = [build_draw(profile, confirm_settings, "evaluation", i) for i in range(100)]
    old_streams = {tuple(b["draw"]["stream_key"]) for b in row["evaluation"]["builds"]}
    if old_streams & {tuple(d["stream_key"]) for d in draws} or CONFIRMATION_SEED == settings.screening_seed:
        raise RuntimeError("confirmation streams overlap diagnostic development draws")
    write_json(directory / "development-draws.json", dict(role="new_development_confirmation", draws=draws))
    payloads = [dict(diagnostic=str(diagnostic), cache_directory=str(cache_directory), seed=CONFIRMATION_SEED,
                     index=i, alternative=alternative_name, alternative_prescription=alternative.model_dump(mode="json"),
                     record_path=str(directory / "builds" / f"{i:03d}.json")) for i in range(100)]
    results = []
    if workers == 1:
        for payload in payloads:
            results.append(confirmation_worker(payload))
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as executor:
            for future in as_completed([executor.submit(confirmation_worker, p) for p in payloads]):
                results.append(future.result())
                if len(results) % 20 == 0:
                    print(json.dumps(dict(confirmation_builds_completed=len(results))), flush=True)
    results.sort(key=lambda r: r["index"])
    original_builds, alternatives = [r["original9"] for r in results], [r["alternative"] for r in results]
    for expected, before, after in zip(draws, original_builds, alternatives, strict=True):
        if expected != before["draw"] or expected != after["draw"]:
            raise RuntimeError("confirmation protocols used different realized errors")
    original_summary, alternative_summary = build_summary(original_builds), build_summary(alternatives)
    result = dict(protocol=STUDY_PROTOCOL, kind="new_development_confirmation", diagnosis=str(diagnosis_path),
                  plan=plan, original9=original_summary, alternative=alternative_summary,
                  paired=compare_builds(original_builds, alternatives),
                  original_reaches_90=original_summary["builds_passed"] >= 90,
                  alternative_reaches_90=alternative_summary["builds_passed"] >= 90 and not plan["alternative_nominal_failures"],
                  independent_development_streams=True, final_study_test_set_accessed=False,
                  elapsed_seconds=time.perf_counter() - started)
    path = directory / "run.json"
    write_json(path, result)
    print(json.dumps(dict(record=str(path), original9=original_summary, alternative=alternative_summary,
                          transitions=result["paired"]["transitions"], elapsed_seconds=result["elapsed_seconds"]), indent=2), flush=True)
    return path, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("diagnose", "confirm", "run"))
    parser.add_argument("--diagnostic", type=Path, default=DIAGNOSTIC)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/optical_sensitivity"))
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=4)
    parser.add_argument("--diagnosis-record", type=Path)
    args = parser.parse_args()
    if not args.output_dir.resolve().is_relative_to((Path.cwd() / "outputs").resolve()):
        parser.error("study output must be inside ignored outputs/")
    diagnosis_path = args.diagnosis_record
    if args.mode in ("diagnose", "run"):
        diagnosis_path, _ = diagnose(args.diagnostic, args.output_dir, args.workers)
    if args.mode in ("confirm", "run"):
        if diagnosis_path is None:
            parser.error("confirm requires --diagnosis-record")
        confirm(diagnosis_path, args.output_dir, args.workers)


if __name__ == "__main__":
    main()

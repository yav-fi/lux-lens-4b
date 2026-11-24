"""Frozen, resumable, offline eight-brief optical development batch."""

import argparse
import importlib.metadata
import json
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from uuid import uuid4

import numpy as np

from lux_frontier.contracts import Brief, BriefCase, Candidate, GraderResult, NominalMetrics
from lux_frontier.optical_diagnostic import HEURISTIC_RULE, heuristic_ranking, sha_file, summarize, verify_common_draws, write_json
from lux_frontier.optical_engine import cooke_prescription, geometry_summary, measure
from lux_frontier.optical_fine_diagnostic import freeze_selectors
from lux_frontier.optical_grading import OpticalCache, OpticalGrader, OpticalSettings, build_draw
from lux_frontier.optical_prescriptions import ArtifactStore, DevelopmentProfile, Prescription, development_profile, digest, verify_profile
from lux_frontier.optical_sensitivity import build_summary, evaluate_focus, nested_focus_grid
from lux_frontier.prompting import render_prompt

PROTOCOL = "development-eight-briefs-v1"
DEFAULT_DIRECTORY = Path("outputs/optical_batches/development-eight-v1")
REFERENCE = Path("outputs/optical_diagnostics/20261123T222848Z-refocus33-6845de14/run.json")
CACHE = Path("outputs/optical/20261123T170131Z-development-daea8868/cache")
FOCUS_CACHE = Path("outputs/optical_sensitivity/cache")
ARTIFACTS = Path("outputs/optical/artifacts")
NUMERICAL_FILES = ("optical_engine.py", "optical_grading.py", "optical_prescriptions.py",
                   "optical_sensitivity.py", "optical_diagnostic.py", "optical_fine_diagnostic.py",
                   "contracts.py", "prompting.py")

# Camera requirements are engineering hypotheses, fixed before new measurements.
# All use the existing 50 mm mounting family because the current optimizer has
# absolute Cooke-scale bounds. The MTF constraints are monochromatic +Y slices.
SCENARIOS = (
    ("label-inspection", "Bright, stopped-down label inspection over a narrow field", 8.0, .25, 60.9, .55,
     ((0, "sagittal", 30, .60), (10, "sagittal", 30, .45), (10, "tangential", 30, .35)),
     "Narrow 10 deg half-field and f/8 favor depth of field; 30 lp/mm label contrast has generous packaging allowance."),
    ("conveyor-vision", "General conveyor machine vision with moderate field coverage", 5.6, .18, 60.65, .55,
     ((0, "sagittal", 30, .65), (16, "sagittal", 30, .35), (16, "tangential", 40, .20)),
     "A 16 deg half-field covers the belt; f/5.6 balances light and aberrations, with higher central than edge contrast."),
    ("warehouse-low-light", "Low-light warehouse navigation with moderate wide coverage", 4.5, .18, 60.7, .55,
     ((0, "sagittal", 20, .55), (18, "sagittal", 20, .30), (18, "tangential", 30, .18)),
     "f/4.5 prioritizes throughput; 18 deg coverage and lower 20/30 lp/mm contrast favor navigation over fine inspection."),
    ("daylight-wide-survey", "Stopped-down daylight scene survey with wider coverage", 8.0, .25, 60.8, .55,
     ((0, "sagittal", 30, .65), (22, "sagittal", 30, .25), (22, "tangential", 30, .20)),
     "A 22 deg half-field is the broadest case; f/8 and relaxed edge contrast trade illumination for coverage."),
    ("compact-inspection", "Compact housing for moderate-field industrial inspection", 6.3, .20, 60.15, .55,
     ((0, "sagittal", 40, .55), (14, "sagittal", 40, .30), (14, "tangential", 40, .22)),
     "The 60.15 mm housing cap is an explicit packaging challenge; f/6.3 and 14 deg coverage support 40 lp/mm inspection."),
    ("fine-barcode", "High-contrast fine barcode inspection over a narrow field", 5.0, .15, 60.65, .55,
     ((0, "sagittal", 60, .60), (12, "sagittal", 60, .35), (12, "tangential", 60, .30)),
     "60 lp/mm and stronger contrast represent finer markings; a 12 deg field and f/5 limit the competing field/throughput demand."),
    ("robot-wide-detail", "Robot scene imaging with central detail and broad peripheral coverage", 5.6, .18, 60.5, .55,
     ((0, "sagittal", 30, .65), (12, "tangential", 20, .40),
      (20, "sagittal", 30, .35), (20, "tangential", 60, .14)),
     "20 deg peripheral coverage, a mid-field check and a weak 60 lp/mm edge requirement combine navigation and object detail at f/5.6."),
    ("red-alignment", "Red-illuminated alignment camera in an industrial cell", 6.3, .20, 60.6, .65,
     ((0, "sagittal", 30, .65), (16, "sagittal", 30, .40), (16, "tangential", 40, .25)),
     "A monochromatic 0.65 um red source is explicitly traced; f/6.3, 16 deg coverage and 30/40 lp/mm contrast target alignment features."),
)

# name, bending amount, three center-thickness offsets, two air-gap offsets,
# sensor offset, fractional stop offset. All dimensions below are mm except
# the dimensionless bending/stop parameters. No glasses or topology are changed.
VARIANTS = (
    ("centered-parent", 0., (0., 0., 0.), (0., 0.), -.06, 0.),
    ("positive-bend", .012, (.03, -.02, .03), (-.09, .08), -.10, -.012),
    ("negative-bend", -.012, (-.03, .02, -.03), (.09, -.08), .08, .012),
    ("thicker-front", .006, (.09, .025, 0.), (-.10, .02), 0., -.006),
    ("thicker-rear", -.006, (0., -.02, .09), (.10, -.02), .12, .006),
    ("thin-positive", .016, (-.06, .025, -.06), (-.04, .12), -.12, .018),
    ("thicker-negative", -.016, (.02, .04, .02), (.04, -.12), .04, -.018),
    ("redistributed-stack", .004, (.04, -.03, -.04), (-.12, -.10), .16, 0.),
)


def versions():
    return {p: importlib.metadata.version(p) for p in ("optiland", "numpy", "scipy", "pydantic")}


def save_once(path: Path, value) -> None:
    """An existing frozen artifact must agree; never silently replace it."""
    normalized = json.loads(json.dumps(value))
    if path.exists():
        if json.loads(path.read_text()) != normalized:
            raise ValueError(f"frozen artifact differs: {path}")
    else:
        write_json(path, normalized)


def candidate_prescriptions(base: Prescription, brief: Brief, presentation_seed: int) -> list[dict]:
    fields = tuple(dict.fromkeys(((0., 0.), *(t.field_angle_deg for t in brief.mtf_thresholds))))
    designs = []
    coefficients = (1, -.3, .6, -.5, .2, -.8)
    rng = np.random.Generator(np.random.PCG64(presentation_seed))
    ids = rng.permutation(np.arange(1, 9)).tolist()
    for local_id, (name, bend, thickness, gaps, sensor, stop) in zip(ids, VARIANTS, strict=True):
        spacings = list(base.spacings_mm)
        for e in range(3):
            spacings[2 * e] += thickness[e]
        for e in range(2):
            spacings[2 * e + 1] += gaps[e]
        p = Prescription.model_validate(dict(base.model_dump(mode="json"),
            radii_mm=tuple(r * (1 + bend * c) for r, c in zip(base.radii_mm, coefficients, strict=True)),
            spacings_mm=spacings, sensor_z_mm=base.sensor_z_mm + sensor,
            stop_diameter_mm=base.stop_diameter_mm * 5 / brief.target_f_number * (1 + stop),
            fields_deg=fields, primary_wavelength_um=brief.mtf_thresholds[0].wavelength_um))
        geometry_summary(p)  # geometric validity only; no optical measurement
        designs.append(dict(candidate_id=local_id, lineage=name, parent_prescription_ref=base.reference.model_dump(mode="json"),
                            prescription_ref=p.reference.model_dump(mode="json"), prescription=p.model_dump(mode="json"),
                            modifications=dict(bending_amount=bend, radius_coefficients=coefficients,
                                               center_thickness_offsets_mm=thickness, air_gap_offsets_mm=gaps,
                                               sensor_offset_mm=sensor, stop_fractional_offset=stop)))
    return [designs[i] for i in rng.permutation(8)]


def make_manifest(base: Prescription, profile: DevelopmentProfile) -> dict:
    """Pure recipe construction: do not measure, refine, screen or evaluate."""
    entries = []
    for i, (name, use_case, fno, ftol, length, wave, targets, rationale) in enumerate(SCENARIOS):
        settings = OpticalSettings(refocus_positions=33, screening_seed=2027010101 + i * 2,
                                   evaluation_seed=2027010102 + i * 2)
        brief = Brief(brief_id=f"development-eight-v1-{i + 1:02d}-{name}",
                      target_focal_length_mm=50, focal_length_tolerance_mm=.5,
                      target_f_number=fno, f_number_tolerance=ftol, max_first_surface_to_sensor_mm=length,
                      mtf_thresholds=tuple(dict(field_angle_deg=(0, field), orientation=orientation,
                                               spatial_frequency_lp_per_mm=frequency, wavelength_um=wave,
                                               minimum_mtf_fraction=minimum) for field, orientation, frequency, minimum in targets),
                      minimum_simulated_yield_fraction=.9, manufacturing_tolerance_profile_ref=profile.reference)
        entries.append(dict(brief=brief.model_dump(mode="json"), use_case=use_case,
            requirement_rationale=dict(
                focal_length="50 +/-0.5 mm preserves the supported Cooke mounting scale and common optimizer bounds.",
                aperture=f"Target f/{fno:g} +/-{ftol:g}: " + rationale,
                field_coverage=rationale + " Field coordinates are object-space degrees on the +Y meridian.",
                packaging=f"First-surface-to-sensor housing limit {length:g} mm; the sensor travel must also fit this limit.",
                mtf=rationale + " Thresholds are provisional task requirements, not claims derived from a sensor datasheet.",
                wavelength=f"Explicit monochromatic illumination at {wave:g} um; no implied broadband average.",
                manufacturing="Unchanged versioned provisional development profile for every brief.",
                yield_acceptance="Fixed development acceptance: at least 90 of 100 fresh simulated builds."),
            settings=settings.model_dump(mode="json"), presentation_seed=2027011101 + i,
            candidates=candidate_prescriptions(base, brief, 2027011101 + i),
            draw_streams={phase: [build_draw(profile, settings, phase, k) for k in range(count)]
                          for phase, count in (("screening", 20), ("evaluation", 100))}))
    return dict(protocol=PROTOCOL, development=True, requirements_provisional=True,
                refocus_assumption_provisional=True, final_study_test_set_accessed=False,
                profile=profile.model_dump(mode="json"), parent_prescription=base.model_dump(mode="json"),
                parent_prescription_ref=base.reference.model_dump(mode="json"), briefs=entries,
                focus_positions_mm=nested_focus_grid(.2, 33), heuristic_rule=HEURISTIC_RULE,
                refinement=dict(optimizer="existing bounded coordinate pattern search", objective_evaluations=37,
                    parameters=["sensor_z_mm", "air_gap_1_mm", "air_gap_2_mm"],
                    bounds_mm=[[59.9, 60.45], [5.8, 6.2], [4.55, 4.95]], initial_steps_mm=[.08, .04, .04]),
                diversity_limits="All 64 starters share Optiland 0.6.3 CookeTriplet ancestry, six spherical surfaces, three air-spaced elements and SK16/F2/SK16 glasses. Geometry, stop, fields and primary wavelength vary; no new topology, asphere, glass search, arbitrary azimuth or broadband MTF is claimed.")


def freeze_batch(directory: Path) -> Path:
    path = directory / "manifest.json"
    if path.exists():
        load_manifest(path)
        return path
    profile, base = development_profile(), cooke_prescription()
    manifest = make_manifest(base, profile)
    manifest.update(created_utc=datetime.now(timezone.utc).isoformat(), versions=versions(),
                    numerical_source_sha256={name: sha_file(Path(__file__).with_name(name)) for name in NUMERICAL_FILES},
                    orchestration_source_sha256=sha_file(Path(__file__)),
                    cooke_reference=dict(path=str(REFERENCE), sha256=sha_file(REFERENCE), included_in_batch=False),
                    cache_directory=str(CACHE), focus_cache_directory=str(FOCUS_CACHE), artifact_directory=str(ARTIFACTS))
    # The complete batch is durable BEFORE storing artifacts or any initial MTF.
    save_once(path, dict(manifest_sha256=digest(manifest), manifest=manifest))
    load_manifest.cache_clear()
    load_manifest(path)
    return path


@lru_cache(maxsize=4)
def load_manifest(path: Path) -> tuple[dict, str]:
    envelope = json.loads(path.read_text())
    m, key = envelope["manifest"], envelope["manifest_sha256"]
    if digest(m) != key or m["protocol"] != PROTOCOL or len(m["briefs"]) != 8:
        raise ValueError("invalid or modified frozen batch manifest")
    profile = DevelopmentProfile.model_validate(m["profile"])
    verify_profile(profile)
    if profile != development_profile() or m["focus_positions_mm"] != nested_focus_grid(.2, 33):
        raise ValueError("batch differs from fixed development profile/focus protocol")
    if m.get("versions") != versions():
        raise ValueError("installed dependency versions differ from frozen manifest")
    for name, sha in m.get("numerical_source_sha256", {}).items():
        if sha_file(Path(__file__).with_name(name)) != sha:
            raise ValueError(f"numerical implementation differs from frozen manifest: {name}")
    streams = set()
    for entry in m["briefs"]:
        b, s = Brief.model_validate(entry["brief"]), OpticalSettings.model_validate(entry["settings"])
        if (s.refinement_evaluations, s.refocus_positions, s.screening_builds, s.evaluation_builds,
                s.pupil_samples, b.minimum_simulated_yield_fraction) != (37, 33, 20, 100, 64, .9):
            raise ValueError("batch changed its fixed workload or acceptance threshold")
        if b.manufacturing_tolerance_profile_ref != profile.reference:
            raise ValueError("brief profile differs from frozen profile")
        candidates = entry["candidates"]
        if len(candidates) != 8 or len({c["candidate_id"] for c in candidates}) != 8:
            raise ValueError("require eight distinct local candidate IDs")
        refs = set()
        for c in candidates:
            p = Prescription.model_validate(c["prescription"])
            if p.reference.model_dump(mode="json") != c["prescription_ref"]:
                raise ValueError("frozen prescription does not match its immutable reference")
            refs.add(p.reference.uri)
        if len(refs) != 8:
            raise ValueError("brief repeats a prescription")
        for phase, count in (("screening", 20), ("evaluation", 100)):
            expected = [json.loads(json.dumps(build_draw(profile, s, phase, k))) for k in range(count)]
            if entry["draw_streams"][phase] != expected:
                raise ValueError("saved draws do not reproduce the frozen seeds/profile")
            for draw in expected:
                stream = tuple(draw["stream_key"])
                if stream in streams:
                    raise ValueError("batch seed streams overlap")
                streams.add(stream)
    ref = m.get("cooke_reference")
    if ref and sha_file(Path(ref["path"])) != ref["sha256"]:
        raise ValueError("separate Cooke reference record changed")
    return m, key


def record_value(path: Path, key: str):
    saved = json.loads(path.read_text())
    if saved["input_sha256"] != key or saved["value_sha256"] != digest(saved["value"]):
        raise ValueError(f"resume record integrity check failed: {path}")
    return saved["value"]


def checkpoint(path: Path, key: str, operation):
    if path.exists():
        return record_value(path, key), True
    value = operation()  # do not turn software errors into optical failures
    save_once(path, dict(input_sha256=key, value_sha256=digest(value), value=value))
    return value, False


def case_root(path: Path, entry: dict) -> Path:
    return path.parent / "briefs" / entry["brief"]["brief_id"]


def job_path(path: Path, entry: dict, candidate_id: int, phase: str, index: int | None = None) -> Path:
    stem = f"{candidate_id}" if index is None else f"{candidate_id}/{index:03d}"
    return case_root(path, entry) / phase / (stem + ".json")


def job_key(manifest_key, entry, candidate_id, phase, index=None):
    return digest(dict(manifest_sha256=manifest_key, brief_id=entry["brief"]["brief_id"],
                       candidate_id=candidate_id, phase=phase, build_index=index))


def read_job(path, entry, manifest_key, candidate_id, phase, index=None):
    return record_value(job_path(path, entry, candidate_id, phase, index),
                        job_key(manifest_key, entry, candidate_id, phase, index))


def batch_worker(payload: dict) -> dict:
    path = Path(payload["manifest"])
    m, key = load_manifest(path)
    entry = m["briefs"][payload["brief_index"]]
    candidate_id, phase, index = payload["candidate_id"], payload["phase"], payload.get("index")
    target = job_path(path, entry, candidate_id, phase, index)
    candidate = next(c for c in entry["candidates"] if c["candidate_id"] == candidate_id)
    p, b = Prescription.model_validate(candidate["prescription"]), Brief.model_validate(entry["brief"])
    profile, settings = DevelopmentProfile.model_validate(m["profile"]), OpticalSettings.model_validate(entry["settings"])
    store, cache = ArtifactStore(Path(m["artifact_directory"])), OpticalCache(Path(m["cache_directory"]))
    store.put(p)

    def execute():
        if phase == "initial":
            inputs = OpticalGrader(store, cache, settings, profile).context(p, b)
            value, _ = cache.compute("fixture_nominal", inputs, lambda: measure(p, b, pupil_samples=settings.pupil_samples)[0].model_dump(mode="json"))
            return value
        if phase == "refinement":
            return OpticalGrader(store, cache, settings, profile).refine(p, b)
        if phase not in ("screening", "evaluation"):
            raise ValueError("unsupported batch phase")
        refined = read_job(path, entry, key, candidate_id, "refinement")
        design = Prescription.model_validate(refined["refined_prescription"])
        draw = entry["draw_streams"][phase][index]
        return evaluate_focus(design, b, profile, settings, draw, m["focus_positions_mm"],
                              OpticalCache(Path(m["focus_cache_directory"])))

    try:
        value, resumed = checkpoint(target, job_key(key, entry, candidate_id, phase, index), execute)
    except Exception as exc:
        write_json(path.parent / "software-errors" / f"{uuid4().hex}.json",
                   dict(manifest_sha256=key, job=payload, type=type(exc).__name__, message=str(exc),
                        prescription_ref=p.reference.model_dump(mode="json"),
                        draw=entry["draw_streams"][phase][index] if phase in ("screening", "evaluation") else None))
        raise
    work = 0 if resumed else value.get("work", {}).get("new_metric_evaluations", value.get("work_this_run", 1))
    return dict(brief_index=payload["brief_index"], candidate_id=candidate_id, phase=phase,
                index=index, resumed=resumed, new_metric_evaluations=work)


def execute_jobs(jobs: list[dict], workers: int):
    if workers == 1:
        for payload in jobs:
            yield batch_worker(payload)
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            for future in as_completed([pool.submit(batch_worker, p) for p in jobs]):
                yield future.result()


def initial_case(path, m, key, entry) -> BriefCase:
    candidates = []
    for c in entry["candidates"]:
        p = Prescription.model_validate(c["prescription"])
        candidates.append(Candidate(candidate_id=c["candidate_id"], prescription_ref=p.reference,
            layout="Cooke-derived spherical positive / negative / positive triplet; stop at rear of element 2.",
            glass_types=tuple(f"{catalog}:{name}" for name, catalog in p.glasses), geometry=geometry_summary(p),
            nominal_metrics=NominalMetrics.model_validate(read_job(path, entry, key, c["candidate_id"], "initial"))))
    return BriefCase(brief=Brief.model_validate(entry["brief"]), candidates=candidates)


def aggregate(builds):
    s = build_summary(builds)
    return dict(s, status="completed", yield_fraction=s["builds_passed"] / len(builds), builds=builds)


def nominal_rows(path, entry, key):
    rows = []
    for c in entry["candidates"]:
        refined = read_job(path, entry, key, c["candidate_id"], "refinement")
        failures = refined["after"].get("constraint_failures", [{"constraint": "physically_invalid"}])
        rows.append(dict(candidate_id=c["candidate_id"], lineage=c["lineage"], refinement=refined,
                         initial_metrics=read_job(path, entry, key, c["candidate_id"], "initial"),
                         nominal_pass="metrics" in refined["after"] and not failures,
                         nominal_failures=failures, refined_prescription_sha256=digest(refined["refined_prescription"])))
    return rows


def pair_contract(case, pair, by_id):
    submission = case.parse_submission(json.dumps(dict(starter_ids=pair["candidate_ids"])))
    if pair["failed_submission"]:
        result = GraderResult(brief_id=case.brief.brief_id, starter_ids=submission.starter_ids,
                              constraint_failures=("neither starter passes all nominal constraints",), final_pass=False, reward=0)
    else:
        row = by_id[pair["shipped_candidate_id"]]
        result = GraderResult(brief_id=case.brief.brief_id, starter_ids=submission.starter_ids,
            shipped_prescription_ref=row["refinement"]["refined_prescription_ref"], nominal_metrics=row["refinement"]["after"]["metrics"],
            simulation=dict(status="completed", builds_passed=pair["builds_passed"], builds_tested=100, yield_fraction=pair["reward"]),
            constraint_failures=() if pair["final_pass"] else ("simulated_yield_fraction < 0.9",),
            final_pass=pair["final_pass"], reward=pair["reward"])
    return case.parse_result(result.model_dump_json()).model_dump(mode="json")


def summarize_brief(rows, heuristic_ids, frozen_plan, case):
    result = summarize(sorted(rows, key=lambda r: r["candidate_id"]), heuristic_ids, .9)
    for actual, frozen in zip(result["pairs"], frozen_plan["pairs"], strict=True):
        if actual["shipped_candidate_id"] != frozen["shipped_candidate_id"] or actual["candidate_ids"] != frozen["candidate_ids"]:
            raise RuntimeError("evaluation changed a frozen pair selection")
    for key, name in (("heuristic", "closest_to_spec"), ("try_all_eight", "try_all_eight")):
        if frozen_plan[key]["shipped_candidate_id"] != result["baselines"][name]["shipped_candidate_id"]:
            raise RuntimeError("evaluation changed a frozen baseline selection")
    by_id = {r["candidate_id"]: r for r in rows}
    for pair in result["pairs"]:
        pair["grader_result"] = pair_contract(case, pair, by_id)
    best = max(p["reward"] for p in result["pairs"])
    result["hindsight_best_achievable_pair"] = dict(deployable_selector=False, shipped_yield=best,
        candidate_pairs=[p["candidate_ids"] for p in result["pairs"] if p["reward"] == best],
        shipped_candidate_ids=sorted({p["shipped_candidate_id"] for p in result["pairs"] if p["reward"] == best and p["shipped_candidate_id"] is not None}))
    result["pairs_meeting_target"] = sum(p["final_pass"] for p in result["pairs"])
    return result


def run_batch(path: Path, workers: int, stop_after: str | None = None):
    started = time.perf_counter()
    load_manifest.cache_clear()
    m, key = load_manifest(path)
    root, session = path.parent, dict(started_utc=datetime.now(timezone.utc).isoformat(), workers=workers, stages={})
    profile = DevelopmentProfile.model_validate(m["profile"])

    def jobs(phase, eligibility=None):
        for i, entry in enumerate(m["briefs"]):
            for c in entry["candidates"]:
                if eligibility is not None and c["candidate_id"] not in eligibility[i]:
                    continue
                count = 20 if phase == "screening" else 100 if phase == "evaluation" else 1
                for index in range(count):
                    yield dict(manifest=str(path), brief_index=i, candidate_id=c["candidate_id"], phase=phase,
                               **(dict(index=index) if phase in ("screening", "evaluation") else {}))

    def stage(phase, eligibility=None):
        payloads, completed, reused, work = list(jobs(phase, eligibility)), 0, 0, 0
        for outcome in execute_jobs(payloads, workers):
            completed += 1
            reused += outcome["resumed"]
            work += outcome["new_metric_evaluations"]
            if completed % (64 if phase in ("initial", "refinement") else 100) == 0 or completed == len(payloads):
                print(json.dumps(dict(phase=phase, completed=completed, total=len(payloads), resumed=reused,
                                      elapsed_seconds=round(time.perf_counter() - started, 2))), flush=True)
        session["stages"][phase] = dict(jobs=completed, resumed=reused, new_metric_evaluations=work)

    stage("initial")
    cases, precommits = [], []
    for entry in m["briefs"]:
        case = initial_case(path, m, key, entry)
        ranking = heuristic_ranking(case, profile)
        precommit = dict(rule=HEURISTIC_RULE, ranking=ranking, selected_ids=[r["candidate_id"] for r in ranking[:2]],
                         manifest_sha256=key, frozen_before_refinement_and_screening=True)
        save_once(case_root(path, entry) / "case.json", case.model_dump(mode="json"))
        save_once(case_root(path, entry) / "prompt.json", dict(prompt=render_prompt(case, profile, provenance="development_optical")))
        save_once(case_root(path, entry) / "heuristic-precommit.json", precommit)
        cases.append(case)
        precommits.append(precommit)
    if stop_after == "prepare":
        return dict(stopped_after="prepare", manifest_sha256=key)
    stage("refinement")
    rows_by_brief = [nominal_rows(path, e, key) for e in m["briefs"]]
    eligibility = [{r["candidate_id"] for r in rows if r["nominal_pass"]} for rows in rows_by_brief]
    print(json.dumps(dict(nominally_passing_starters=[len(x) for x in eligibility])), flush=True)
    stage("screening", eligibility)
    plans = []
    for entry, rows, precommit in zip(m["briefs"], rows_by_brief, precommits, strict=True):
        for row in rows:
            row["screening"] = aggregate([read_job(path, entry, key, row["candidate_id"], "screening", k) for k in range(20)]) if row["nominal_pass"] else None
        plan = freeze_selectors(rows, precommit["selected_ids"])
        plan.update(manifest_sha256=key, frozen_before_any_fresh_evaluation=True)
        save_once(case_root(path, entry) / "selection-plan.json", plan)
        plans.append(plan)
    # All eight briefs' 28-pair and baseline choices are durable before any
    # fresh evaluation job is scheduled, including on a partial-run resume.
    if stop_after == "screening":
        return dict(stopped_after="screening", manifest_sha256=key)
    stage("evaluation", eligibility)
    summaries = []
    for entry, rows, case, precommit, plan in zip(m["briefs"], rows_by_brief, cases, precommits, plans, strict=True):
        for row in rows:
            row["evaluation"] = aggregate([read_job(path, entry, key, row["candidate_id"], "evaluation", k) for k in range(100)]) if row["nominal_pass"] else None
            row["final_pass"] = row["nominal_pass"] and row["evaluation"]["builds_passed"] >= 90
        result = summarize_brief(rows, precommit["selected_ids"], plan, case)
        result.update(brief_id=case.brief.brief_id, use_case=entry["use_case"], manifest_sha256=key,
                      starters=rows, selection_plan=plan, heuristic_precommit=precommit,
                      draw_verification=verify_common_draws(rows, OpticalSettings.model_validate(entry["settings"])))
        result_path = case_root(path, entry) / "run.json"
        save_once(result_path, result)
        summaries.append(dict(brief_id=case.brief.brief_id, use_case=entry["use_case"], record=str(result_path),
            heuristic_yield=result["baselines"]["closest_to_spec"]["reward"],
            expected_random_yield=result["baselines"]["uniform_random_pair"]["expected_shipped_yield"],
            exact_expected_random_yield=result["baselines"]["uniform_random_pair"]["exact_expected_shipped_yield"],
            try_all_eight_yield=result["baselines"]["try_all_eight"]["reward"],
            highest_observed_achievable_pair_yield=result["hindsight_best_achievable_pair"]["shipped_yield"],
            pairs_meeting_target=result["pairs_meeting_target"], nominally_passing_starters=sum(r["nominal_pass"] for r in rows)))
    counts = dict(any_pair_passes=sum(r["pairs_meeting_target"] > 0 for r in summaries),
                  heuristic_fails_another_pair_passes=sum(r["heuristic_yield"] < .9 and r["pairs_meeting_target"] > 0 for r in summaries),
                  heuristic_achieves_highest_observed_yield=sum(r["heuristic_yield"] == r["highest_observed_achievable_pair_yield"] for r in summaries), denominator=8)
    result = dict(protocol=PROTOCOL, manifest=str(path), manifest_sha256=key, briefs=summaries, counts=counts,
                  cooke_reference=m["cooke_reference"], target_yield=.9, development=True,
                  final_study_test_set_accessed=False, tinker_calls=0)
    save_once(root / "run.json", result)
    if sha_file(Path(m["cooke_reference"]["path"])) != m["cooke_reference"]["sha256"]:
        raise RuntimeError("separate Cooke reference changed")
    session.update(completed_utc=datetime.now(timezone.utc).isoformat(), manifest_sha256=key,
                   elapsed_seconds=time.perf_counter() - started, result=str(root / "run.json"))
    write_json(root / "sessions" / f"{uuid4().hex}.json", session)
    print(json.dumps(dict(record=str(root / "run.json"), counts=counts, briefs=summaries, session=session), indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--output-dir", type=Path, default=DEFAULT_DIRECTORY)
    run = sub.add_parser("run")
    run.add_argument("--manifest", type=Path, default=DEFAULT_DIRECTORY / "manifest.json")
    run.add_argument("--workers", type=int, choices=range(1, 13), default=8)
    run.add_argument("--stop-after", choices=("prepare", "screening"))
    args = parser.parse_args()
    target = args.output_dir if args.command == "freeze" else args.manifest.parent
    if not target.resolve().is_relative_to((Path.cwd() / "outputs").resolve()):
        parser.error("batch records must be inside ignored outputs/")
    if args.command == "freeze":
        print(freeze_batch(args.output_dir))
    else:
        run_batch(args.manifest, args.workers, args.stop_after)


if __name__ == "__main__":
    main()

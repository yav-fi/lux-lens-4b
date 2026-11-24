"""Offline Cooke development grading with measured MTF and manufacturing draws."""

# ruff: noqa: E402 -- capture cold process time before importing the optical stack

import time

_PROCESS_START = time.perf_counter()

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist
from typing import Annotated, Self
from uuid import uuid4

import numpy as np
from pydantic import Field, model_validator

from lux_frontier.contracts import Brief, BriefCase, Candidate, Contract, GraderResult, NominalMetrics, Submission
from lux_frontier.optical_engine import BuildErrors, InvalidBuild, constraint_failures, cooke_prescription, geometry_summary, measure
from lux_frontier.optical_prescriptions import ArtifactStore, DevelopmentProfile, Prescription, PROTOCOL, development_profile, digest, verify_profile
from lux_frontier.prompting import render_prompt


class OpticalSettings(Contract):
    pupil_samples: Annotated[int, Field(ge=16, le=256)] = 64
    refinement_evaluations: Annotated[int, Field(ge=1, le=193)] = 37
    screening_builds: Annotated[int, Field(ge=1, le=1000)] = 20
    evaluation_builds: Annotated[int, Field(ge=1, le=1000)] = 100
    refocus_positions: Annotated[int, Field(ge=3, le=41)] = 9
    screening_seed: Annotated[int, Field(ge=0)] = 20261123
    evaluation_seed: Annotated[int, Field(ge=0)] = 20261124

    @model_validator(mode="after")
    def numerical_protocol(self) -> Self:
        if self.pupil_samples & (self.pupil_samples - 1):
            raise ValueError("pupil_samples must be a power of two")
        if self.refocus_positions % 2 != 1:
            raise ValueError("refocus_positions must be odd, including the nominal sensor")
        if self.screening_seed == self.evaluation_seed:
            raise ValueError("screening and evaluation seeds must be disjoint")
        return self


def development_brief(profile: DevelopmentProfile) -> Brief:
    targets = [(0, "sagittal", 30, 0.65), (14, "tangential", 20, 0.35),
               (20, "sagittal", 30, 0.20), (20, "tangential", 60, 0.12)]
    return Brief(brief_id="cooke-development-v1-" + profile.label,
                 target_focal_length_mm=50, focal_length_tolerance_mm=0.5,
                 max_first_surface_to_sensor_mm=60.6, target_f_number=5, f_number_tolerance=0.15,
                 mtf_thresholds=tuple(dict(field_angle_deg=(0, field), orientation=orientation,
                                          spatial_frequency_lp_per_mm=frequency, wavelength_um=0.55,
                                          minimum_mtf_fraction=minimum) for field, orientation, frequency, minimum in targets),
                 minimum_simulated_yield_fraction=0.9, manufacturing_tolerance_profile_ref=profile.reference)


class OpticalCache:
    def __init__(self, directory: Path):
        self.directory = directory
        self.hits = Counter()
        self.misses = Counter()

    def compute(self, kind: str, inputs: dict, operation):
        inputs = _json(inputs)
        key = digest(inputs)
        path = self.directory / kind / (key + ".json")
        if path.exists():
            payload = json.loads(path.read_text())
            if payload["key_inputs"] != inputs or payload["value_digest"] != digest(payload["value"]):
                raise ValueError("optical cache integrity check failed")
            self.hits[kind] += 1
            return payload["value"], True
        self.misses[kind] += 1
        value = _json(operation())  # software errors are deliberately never cached
        payload = dict(key_inputs=inputs, value=value, value_digest=digest(value))
        path.parent.mkdir(parents=True, exist_ok=True)
        # Independent workers can legitimately compute an identical physical
        # control state. Give each writer its own atomic replacement file.
        temporary = path.with_name(path.stem + "." + uuid4().hex + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
        temporary.replace(path)
        return value, False


def _json(value):
    return json.loads(json.dumps(value, default=lambda x: x.model_dump(mode="json")))


def make_development_case(store: ArtifactStore, cache: OpticalCache, settings: OpticalSettings,
                          profile: DevelopmentProfile) -> BriefCase:
    base = cooke_prescription()
    brief = development_brief(profile)
    candidates = []
    coefficients = (1, -0.3, 0.6, -0.5, 0.2, -0.8)
    for index in range(8):
        amount = (index - 3.5) * 0.00015
        p = base.model_copy(update={
            "radii_mm": tuple(r * (1 + amount * c) for r, c in zip(base.radii_mm, coefficients, strict=True)),
            "sensor_z_mm": base.sensor_z_mm + (index - 3.5) * 0.04,
        })
        ref = store.put(p)
        inputs = _json(dict(protocol=PROTOCOL, prescription=p, brief=brief,
                            profile=profile, settings=settings, numpy_version=np.__version__))
        metrics, _ = cache.compute("fixture_nominal", inputs,
                                   lambda p=p: measure(p, brief, pupil_samples=settings.pupil_samples)[0].model_dump(mode="json"))
        candidates.append(Candidate(candidate_id=index + 1, prescription_ref=ref,
                                    layout=f"Cooke triplet controlled variant {index + 1}: spherical positive / negative / positive; stop on rear surface of element 2.",
                                    glass_types=tuple(f"{catalog}:{name}" for name, catalog in p.glasses),
                                    nominal_metrics=NominalMetrics.model_validate(metrics), geometry=geometry_summary(p)))
    # Presentation order is unrelated to design identity or grader tie-breaking.
    order = (5, 1, 7, 3, 0, 6, 2, 4)
    return BriefCase(brief=brief, candidates=tuple(candidates[i] for i in order))


def build_draw(profile: DevelopmentProfile, settings: OpticalSettings, phase: str, index: int) -> dict:
    if phase not in ("screening", "evaluation"):
        raise ValueError("unsupported draw phase")
    if index < 0:
        raise ValueError("build index must be nonnegative")
    seed = settings.screening_seed if phase == "screening" else settings.evaluation_seed
    phase_code = 100 if phase == "screening" else 200
    sequence = np.random.SeedSequence([seed, phase_code, index])
    rng = np.random.Generator(np.random.PCG64(sequence))
    shapes = {"radius_error": (6,), "thickness_error": (3,), "air_gap_error": (2,), "decenter": (3, 2), "tilt": (3, 2)}
    uniforms, values = {}, {}
    # Canonical parameter order prevents profile list ordering changing the draws.
    by_name = {e.parameter: e for e in profile.errors}
    for parameter in shapes:
        error = by_name[parameter]
        u = rng.random(shapes[parameter])
        uniforms[parameter] = u.tolist()
        if error.distribution == "uniform":
            value = error.lower_bound + u * (error.upper_bound - error.lower_bound)
        elif error.distribution == "truncated_normal":
            normal = NormalDist(error.mean, error.standard_deviation)
            lower, upper = normal.cdf(error.lower_bound), normal.cdf(error.upper_bound)
            value = np.array([normal.inv_cdf(float(lower + v * (upper - lower))) for v in u.ravel()]).reshape(u.shape)
        else:
            raise ValueError("unsupported manufacturing error distribution")
        if not np.all((value >= error.lower_bound) & (value <= error.upper_bound)):
            raise RuntimeError("manufacturing draw escaped profile bounds")
        values[parameter] = value.tolist()
    errors = BuildErrors(tuple(values["radius_error"]), tuple(values["thickness_error"]), tuple(values["air_gap_error"]),
                         tuple(tuple(v) for v in values["decenter"]), tuple(tuple(v) for v in values["tilt"]))
    return dict(seed=seed, phase=phase, index=index, stream_key=[seed, phase_code, index],
                bit_generator="PCG64", underlying_uniform_draws=uniforms, errors=asdict(errors))


def objective(metrics: NominalMetrics, brief: Brief) -> float:
    """Same smooth constrained nominal merit and same budget for both starters."""
    focal = abs(metrics.effective_focal_length_mm - brief.target_focal_length_mm) / max(brief.focal_length_tolerance_mm, 1e-9)
    fno = abs(metrics.f_number - brief.target_f_number) / max(brief.f_number_tolerance, 1e-9)
    track = max(0, metrics.first_surface_to_sensor_mm - brief.max_first_surface_to_sensor_mm)
    shortfalls = [max(0, t.minimum_mtf_fraction - m.mtf_fraction) / max(t.minimum_mtf_fraction, 1e-9)
                  for m, t in zip(metrics.mtf, brief.mtf_thresholds, strict=True)]
    violations = max(0, focal - 1) ** 2 + max(0, fno - 1) ** 2 + track ** 2 + sum(v ** 2 for v in shortfalls)
    return 100 * violations + 0.01 * (focal ** 2 + fno ** 2) + 0.1 * sum((1 - m.mtf_fraction) ** 2 for m in metrics.mtf)


@dataclass(frozen=True)
class ScreenedCandidate:
    """Only nominal/screening information; no fresh-evaluation outcome allowed."""

    candidate_id: int
    nominal_pass: bool
    screening_passed: int | None
    refined_prescription_sha256: str


def select_shipped(candidates: list[ScreenedCandidate]) -> ScreenedCandidate | None:
    eligible = [c for c in candidates if c.nominal_pass]
    if any(c.screening_passed is None for c in eligible):
        raise ValueError("a nominally passing candidate requires completed screening")
    return min(eligible, key=lambda c: (-c.screening_passed, c.refined_prescription_sha256)) if eligible else None


def load_recorded_case(source_path: Path):
    """One verified loader for optical diagnostics and training task sources."""
    source = json.loads(source_path.read_text())
    if source["protocol"] != PROTOCOL:
        raise ValueError("unsupported source optical protocol")
    case = BriefCase.model_validate_json((source_path.parent / "brief.json").read_text())
    profile = DevelopmentProfile.model_validate(source["profile"])
    settings = OpticalSettings.model_validate(source["settings"])
    verify_profile(profile)
    if case.brief.model_dump(mode="json") != source["brief"]:
        raise ValueError("saved case does not match the original brief")
    if case.brief.manufacturing_tolerance_profile_ref != profile.reference:
        raise ValueError("source profile does not match the brief")
    return source, case, profile, settings


class OpticalGrader:
    """Callable replacement for the mock reward, plus detailed contract result."""

    def __init__(self, store: ArtifactStore, cache: OpticalCache, settings: OpticalSettings, profile: DevelopmentProfile):
        verify_profile(profile)
        self.store, self.cache, self.settings, self.profile = store, cache, settings, profile
        self.last_run = None

    def context(self, p: Prescription, brief: Brief) -> dict:
        return _json(dict(protocol=PROTOCOL, prescription=p, brief=brief, profile=self.profile,
                          settings=self.settings, numpy_version=np.__version__, optiland_version="0.6.3"))

    def nominal_objective(self, metrics: NominalMetrics, brief: Brief) -> float:
        return objective(metrics, brief)

    def refine(self, p: Prescription, brief: Brief) -> dict:
        inputs = self.context(p, brief)

        def execute():
            started = time.perf_counter()
            bounds = [(59.9, 60.45), (5.8, 6.2), (4.55, 4.95)]
            initial_steps = np.array([0.08, 0.04, 0.04])
            x = np.array([p.sensor_z_mm, p.spacings_mm[1], p.spacings_mm[3]])
            if any(not lo <= v <= hi for v, (lo, hi) in zip(x, bounds, strict=True)):
                raise ValueError("starter parameters lie outside common refinement bounds")
            history = []

            def evaluate(vector):
                spacings = list(p.spacings_mm)
                spacings[1], spacings[3] = float(vector[1]), float(vector[2])
                design = p.model_copy(update={"sensor_z_mm": float(vector[0]), "spacings_mm": tuple(spacings)})
                try:
                    metrics, details = measure(design, brief, pupil_samples=self.settings.pupil_samples)
                    value = self.nominal_objective(metrics, brief)
                    entry = dict(parameters=vector.tolist(), objective=value, metrics=metrics.model_dump(mode="json"), details=details,
                                 constraint_failures=constraint_failures(metrics, brief))
                except InvalidBuild as exc:
                    value = 1e9
                    entry = dict(parameters=vector.tolist(), objective=value, physically_invalid=str(exc))
                history.append(entry)
                return value, design, entry

            best, winner, best_entry = evaluate(x)
            before = best_entry
            steps = initial_steps.copy()
            for evaluation in range(1, self.settings.refinement_evaluations):
                coordinate = ((evaluation - 1) // 2) % 3
                sign = -1 if evaluation % 2 else 1
                trial = x.copy()
                lo, hi = bounds[coordinate]
                trial[coordinate] = np.clip(trial[coordinate] + sign * steps[coordinate], lo, hi)
                score, design, entry = evaluate(trial)
                if score < best:
                    x, best, winner, best_entry = trial, score, design, entry
                if evaluation % 6 == 0:
                    steps *= 0.65
            return dict(before=before, after=best_entry, refined_prescription=winner.model_dump(mode="json"),
                        refined_prescription_ref=self.store.put(winner).model_dump(mode="json"),
                        optimizer="deterministic bounded coordinate pattern search", parameter_names=["sensor_z_mm", "air_gap_1_mm", "air_gap_2_mm"],
                        bounds=bounds, initial_steps=initial_steps.tolist(), history=history,
                        work=dict(objective_evaluations=len(history), metric_evaluations=len(history)),
                        elapsed_seconds=time.perf_counter() - started)

        value, hit = self.cache.compute("refinement", inputs, execute)
        self.store.put(Prescription.model_validate(value["refined_prescription"]))
        return dict(value, cache_hit=hit, work_this_run=0 if hit else value["work"]["metric_evaluations"])

    def simulate_build(self, p: Prescription, brief: Brief, draw: dict) -> dict:
        inputs = dict(self.context(p, brief), draw=draw)

        def execute():
            errors = BuildErrors(**draw["errors"])
            refocus = self.profile.refocus
            shifts = np.linspace(-refocus.maximum_sensor_shift_mm, refocus.maximum_sensor_shift_mm,
                                 self.settings.refocus_positions).tolist() if refocus.allowed else [0.0]
            shifts = sorted(set(shifts), key=lambda v: (abs(v), v))
            trials, best, selected = [], -float("inf"), None
            for shift in shifts:
                try:
                    metrics, details = measure(p, brief, pupil_samples=self.settings.pupil_samples,
                                               errors=errors, sensor_shift_mm=shift)
                    score = min(m.mtf_fraction / max(t.minimum_mtf_fraction, 1e-9)
                                for m, t in zip(metrics.mtf, brief.mtf_thresholds, strict=True))
                    trials.append(dict(sensor_shift_mm=shift, minimum_mtf_ratio=score))
                    if score > best:
                        best = score
                        selected = dict(status="measured", metrics=metrics.model_dump(mode="json"), details=details,
                                        failures=constraint_failures(metrics, brief))
                except InvalidBuild as exc:
                    trials.append(dict(sensor_shift_mm=shift, physically_invalid=str(exc)))
            if selected is None:
                selected = dict(status="physically_invalid", metrics=None, failures=[dict(constraint=trials[0]["physically_invalid"])])
            selected.update(passed=not selected["failures"], draw=draw, refocus_trials=trials,
                            prescription_ref=p.reference.model_dump(mode="json"),
                            work=dict(metric_evaluations=len(trials)))
            return selected

        value, hit = self.cache.compute("build", inputs, execute)
        return dict(value, cache_hit=hit, work_this_run=0 if hit else value["work"]["metric_evaluations"])

    def simulate(self, p: Prescription, brief: Brief, phase: str, count: int) -> dict:
        started = time.perf_counter()
        builds = []
        for index in range(count):
            draw = build_draw(self.profile, self.settings, phase, index)
            try:
                builds.append(self.simulate_build(p, brief, draw))
            except Exception as exc:
                # Never convert a programming/library/configuration error into
                # an optical failure or a measured zero yield.
                return dict(status="software_error", builds=builds,
                            software_error=dict(type=type(exc).__name__, message=str(exc), draw=draw,
                                                prescription_ref=p.reference.model_dump(mode="json")),
                            elapsed_seconds=time.perf_counter() - started)
        passed = sum(b["passed"] for b in builds)
        return dict(status="completed", builds_passed=passed, builds_tested=count, yield_fraction=passed / count,
                    failures_by_constraint=dict(Counter(f["constraint"] for b in builds for f in b["failures"])),
                    builds=builds, elapsed_seconds=time.perf_counter() - started,
                    work_this_run=sum(b["work_this_run"] for b in builds))

    def grade(self, case: BriefCase, submission: Submission) -> dict:
        started = time.perf_counter()
        case.parse_submission(submission.model_dump_json())
        if case.brief.manufacturing_tolerance_profile_ref != self.profile.reference:
            raise ValueError("required optical tolerance profile cannot be resolved")
        by_id = {c.candidate_id: c for c in case.candidates}
        run = dict(protocol=PROTOCOL, development=True, provisional_tolerances=self.profile.provisional,
                   brief=case.brief.model_dump(mode="json"), submission=submission.model_dump(mode="json"), starters=[])
        eligible = []
        for local_id in submission.starter_ids:
            p = self.store.resolve(by_id[local_id].prescription_ref)
            try:
                refined = self.refine(p, case.brief)
            except Exception as exc:
                run["software_error"] = dict(stage="refinement", type=type(exc).__name__, message=str(exc),
                                             prescription_ref=p.reference.model_dump(mode="json"))
                result = GraderResult(brief_id=case.brief.brief_id, starter_ids=submission.starter_ids,
                                     simulation=dict(status="failed", error="software error during refinement"))
                run.update(result=result.model_dump(mode="json"), elapsed_seconds=time.perf_counter() - started)
                self.last_run = run
                return run
            entry = dict(candidate_id=local_id, initial_prescription_ref=p.reference.model_dump(mode="json"), refinement=refined)
            run["starters"].append(entry)
            if "metrics" in refined["after"] and not refined["after"]["constraint_failures"]:
                design = Prescription.model_validate(refined["refined_prescription"])
                screening = self.simulate(design, case.brief, "screening", self.settings.screening_builds)
                entry["screening"] = screening
                if screening["status"] == "software_error":
                    result = GraderResult(brief_id=case.brief.brief_id, starter_ids=submission.starter_ids,
                                         simulation=dict(status="failed", error="software error during screening"))
                    run["result"] = result.model_dump(mode="json")
                    run["elapsed_seconds"] = time.perf_counter() - started
                    self.last_run = run
                    return run
                eligible.append((entry, design))
        if not eligible:
            result = GraderResult(brief_id=case.brief.brief_id, starter_ids=submission.starter_ids,
                                 constraint_failures=("neither starter passes all nominal constraints",), final_pass=False, reward=0)
        else:
            chosen = select_shipped([ScreenedCandidate(e["candidate_id"], True, e["screening"]["builds_passed"], digest(p))
                                     for e, p in eligible])
            entry, design = next((e, p) for e, p in eligible if e["candidate_id"] == chosen.candidate_id)
            run["selection"] = dict(frozen_before_evaluation=True, shipped_candidate_id=entry["candidate_id"],
                                    prescription_ref=design.reference.model_dump(mode="json"),
                                    screening_yield=entry["screening"]["yield_fraction"],
                                    tie_break="lexicographically smallest SHA-256 of exact refined prescription",
                                    evaluation_seed=self.settings.evaluation_seed)
            evaluation = self.simulate(design, case.brief, "evaluation", self.settings.evaluation_builds)
            run["evaluation"] = evaluation
            nominal = NominalMetrics.model_validate(entry["refinement"]["after"]["metrics"])
            if evaluation["status"] == "software_error":
                result = GraderResult(brief_id=case.brief.brief_id, starter_ids=submission.starter_ids,
                                     shipped_prescription_ref=design.reference, nominal_metrics=nominal,
                                     simulation=dict(status="failed", error="software error during fresh evaluation"))
            else:
                passed = evaluation["yield_fraction"] >= case.brief.minimum_simulated_yield_fraction
                result = GraderResult(brief_id=case.brief.brief_id, starter_ids=submission.starter_ids,
                                     shipped_prescription_ref=design.reference, nominal_metrics=nominal,
                                     simulation=dict(status="completed", builds_passed=evaluation["builds_passed"],
                                                     builds_tested=evaluation["builds_tested"], yield_fraction=evaluation["yield_fraction"]),
                                     constraint_failures=() if passed else (f"simulated_yield_fraction < {case.brief.minimum_simulated_yield_fraction:g}",),
                                     final_pass=passed, reward=evaluation["yield_fraction"])
        case.parse_result(result.model_dump_json())
        run.update(result=result.model_dump(mode="json"), elapsed_seconds=time.perf_counter() - started)
        self.last_run = run
        return run

    def __call__(self, environment, submission: Submission) -> float:
        if environment.tolerance_profile != self.profile:
            raise ValueError("environment profile does not match the optical grader")
        result = self.grade(environment.case, submission)["result"]
        if result["simulation"]["status"] == "failed":
            raise RuntimeError(result["simulation"]["error"])
        return result["reward"]


def replay_evaluation(source_path: Path, build_index: int, output_root: Path) -> tuple[Path, dict]:
    """Rebuild one recorded realization from self-contained prescription + draws."""
    source = json.loads(source_path.read_text())
    if source["protocol"] != PROTOCOL:
        raise ValueError("unsupported replay protocol")
    profile = DevelopmentProfile.model_validate(source["profile"])
    settings = OpticalSettings.model_validate(source["settings"])
    brief = Brief.model_validate(source["brief"])
    builds = source.get("evaluation", {}).get("builds", [])
    if not 0 <= build_index < len(builds):
        raise ValueError("replay index is outside the recorded fresh evaluation builds")
    original = builds[build_index]
    entry = next(s for s in source["starters"]
                 if s["refinement"]["refined_prescription_ref"] == original["prescription_ref"])
    p = Prescription.model_validate(entry["refinement"]["refined_prescription"])
    if p.reference.model_dump(mode="json") != original["prescription_ref"]:
        raise ValueError("replay prescription does not match its immutable reference")
    expected_draw = build_draw(profile, settings, "evaluation", build_index)
    if _json(expected_draw) != original["draw"]:
        raise ValueError("recorded draw does not reproduce from the stated seed and protocol")
    directory = output_root / f"replay-{uuid4().hex[:8]}"
    store = ArtifactStore(directory / "artifacts")
    store.put(p)
    grader = OpticalGrader(store, OpticalCache(directory / "cache"), settings, profile)
    reproduced = grader.simulate_build(p, brief, expected_draw)
    matched = all(reproduced.get(key) == original.get(key)
                  for key in ("metrics", "details", "failures", "status", "refocus_trials"))
    result = dict(protocol=PROTOCOL, source_record=str(source_path), build_index=build_index,
                  exact_reproduction=matched, build=reproduced)
    path = directory / "run.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return path, result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--brief", type=Path, help="existing BriefCase JSON; default generates the measured development case")
    parser.add_argument("--profile", type=Path, help="resolved DevelopmentProfile JSON; default uses the versioned development profile")
    parser.add_argument("--stress-profile", action="store_true")
    parser.add_argument("--submission", default='{"starter_ids":[3,7]}')
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/optical"))
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--cold", action="store_true", help="use a fresh per-run cache without removing existing work")
    parser.add_argument("--replay-record", type=Path, help="reproduce one saved fresh-evaluation build")
    parser.add_argument("--build-index", type=int, default=0, help="zero-based evaluation build for --replay-record")
    for name, field in OpticalSettings.model_fields.items():
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=field.default)
    args = vars(parser.parse_args())
    brief_path, profile_path = args.pop("brief"), args.pop("profile")
    stress, submission_raw = args.pop("stress_profile"), args.pop("submission")
    root, cache_path, artifact_path, cold = args.pop("output_dir"), args.pop("cache_dir"), args.pop("artifact_dir"), args.pop("cold")
    replay_path, build_index = args.pop("replay_record"), args.pop("build_index")
    if not root.resolve().is_relative_to((Path.cwd() / "outputs").resolve()):
        parser.error("optical run records must be inside ignored outputs/")
    if replay_path:
        path, result = replay_evaluation(replay_path, build_index, root)
        print(json.dumps(dict(record=str(path), exact_reproduction=result["exact_reproduction"],
                              failures=result["build"]["failures"]), indent=2))
        if not result["exact_reproduction"]:
            raise SystemExit(1)
        return
    if cold and cache_path:
        parser.error("--cold and --cache-dir are mutually exclusive")
    if stress and profile_path:
        parser.error("--stress-profile and --profile are mutually exclusive")
    settings = OpticalSettings(**args)
    profile = DevelopmentProfile.model_validate_json(profile_path.read_text()) if profile_path else development_profile(stress=stress)
    verify_profile(profile)
    run_dir = root / f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{profile.label}-{uuid4().hex[:8]}"
    run_dir.mkdir(parents=True)
    store = ArtifactStore(artifact_path or root / "artifacts")
    cache = OpticalCache(cache_path or (run_dir / "cache" if cold else root / "cache"))
    case = BriefCase.model_validate_json(brief_path.read_text()) if brief_path else make_development_case(store, cache, settings, profile)
    submission = case.parse_submission(submission_raw)
    (run_dir / "brief.json").write_text(case.model_dump_json(indent=2) + "\n")
    (run_dir / "profile.json").write_text(profile.model_dump_json(indent=2) + "\n")
    prompt = render_prompt(case, profile, provenance="development_optical")
    grader = OpticalGrader(store, cache, settings, profile)
    run = grader.grade(case, submission)
    run.update(settings=settings.model_dump(mode="json"), profile=profile.model_dump(mode="json"), prompt=prompt,
               cache_directory=str(cache.directory), artifact_directory=str(store.directory),
               cache_hits=dict(cache.hits), cache_misses=dict(cache.misses), requested_cold=cold,
               process_elapsed_seconds=time.perf_counter() - _PROCESS_START,
               dependencies=dict(optiland="0.6.3", numpy=np.__version__),
               assumptions=["PROVISIONAL development brief and tolerance bounds; no manufacturing review",
                            f"Scalar incoherent FFT MTF, {settings.pupil_samples}x{settings.pupil_samples} pupil samples, 4x zero padding; linear interpolation in image-space lp/mm",
                            "+Y meridional fields only; tangential=pupil Y, sagittal=pupil X; monochromatic per threshold",
                            "Real-ray differential on-axis EFL (mean of X/Y focal axes); real-ray on-axis working f-number",
                            "Optiland scalar pupil references ignore small tilts/decenters; actual ray intersections and wavefronts use rigid geometry",
                            "First-vertex housing datum; fixed physical stop; one physical sensor plane for all fields per build",
                            "Point-estimate simulated yield; no confidence-bound acceptance or production-yield claim"])
    path = run_dir / "run.json"
    path.write_text(json.dumps(run, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(record=str(path), result=run["result"], selection=run.get("selection"),
                          cache_hits=run["cache_hits"], cache_misses=run["cache_misses"],
                          process_elapsed_seconds=run["process_elapsed_seconds"]), indent=2))
    if run["result"]["simulation"]["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

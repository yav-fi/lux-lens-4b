"""Reproducible SYNTHETIC cases and an invented grader, never optical evidence."""

import hashlib
import random
from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite
from typing import Literal

from lux_frontier.contracts import ArtifactRef, BriefCase, Candidate, Submission
from lux_frontier.prompting import ResolvedToleranceProfile, resolve_profile
from lux_frontier.synthetic_fixture import synthetic_fixture


@dataclass(frozen=True)
class LensEnvironment:
    case: BriefCase  # retains immutable full-prescription references for a real evaluator
    tolerance_profile: ResolvedToleranceProfile


def synthetic_profile(ref: ArtifactRef) -> ResolvedToleranceProfile:
    return ResolvedToleranceProfile.model_validate({
        "reference": ref,
        "errors": [
            dict(parameter="radius_error", applies_to="each spherical surface, fractional radius change",
                 distribution="truncated_normal", units="dimensionless",
                 lower_bound=-0.003, upper_bound=0.003, standard_deviation=0.001),
            dict(parameter="thickness_error", applies_to="each element center thickness and each air gap",
                 distribution="uniform", units="mm", lower_bound=-0.03, upper_bound=0.03),
            dict(parameter="decenter", applies_to="each element, independently on x and y",
                 distribution="truncated_normal", units="mm", lower_bound=-0.06,
                 upper_bound=0.06, standard_deviation=0.02),
            dict(parameter="tilt", applies_to="each element, independently about x and y",
                 distribution="uniform", units="deg", lower_bound=-0.1, upper_bound=0.1),
            dict(parameter="index_error", applies_to="each element, common shift across wavelengths",
                 distribution="uniform", units="dimensionless", lower_bound=-0.0002, upper_bound=0.0002),
        ],
        "correlations": "Independent across parameters and elements; index shift shared across wavelengths within an element.",
        "refocus": dict(allowed=True, maximum_sensor_shift_mm=0.2,
                        objective="maximize the minimum required MTF using one axial sensor shift per build"),
    })


def make_environment(seed: int = 17, split: Literal["train", "eval"] = "train", index: int = 0) -> LensEnvironment:
    """Eight distinct tradeoffs; separate deterministic random streams for train/eval.

    Candidate order and local IDs are independently shuffled. Identity remains the
    prescription reference, so scores cannot depend on position or local labels.
    """
    if split not in ("train", "eval") or index < 0:
        raise ValueError("expected train/eval split and nonnegative case index")
    digest = hashlib.sha256(f"lux-synthetic-v1:{seed}:{split}:{index}".encode()).digest()
    rng = random.Random(int.from_bytes(digest, "big"))
    original, _ = synthetic_fixture()
    payload = original.model_dump()
    brief = payload["brief"]
    brief_id = f"synthetic-v1-{seed}-{split}-{index}"
    focal = rng.choice([35.0, 45.0, 50.0, 65.0, 75.0])
    fnum = rng.choice([3.5, 4.0, 5.0, 5.6, 7.0])
    track = round(focal * rng.uniform(1.05, 1.3), 3)
    brief.update(brief_id=brief_id, target_focal_length_mm=focal,
                 focal_length_tolerance_mm=round(focal * 0.02, 3),
                 max_first_surface_to_sensor_mm=track, target_f_number=fnum,
                 f_number_tolerance=0.15, minimum_simulated_yield_fraction=rng.choice([0.8, 0.85, 0.9]))
    thresholds = brief["mtf_thresholds"]
    thresholds[0]["minimum_mtf_fraction"] = round(rng.uniform(0.55, 0.75), 3)
    thresholds[1]["minimum_mtf_fraction"] = round(rng.uniform(0.25, 0.45), 3)
    layouts = [
        ("Three separated spherical elements: positive / negative / positive; stop after element 2.", 3),
        ("Four spherical elements: positive / negative / negative / positive; central stop.", 4),
        ("Two spherical elements: positive / negative; front stop.", 2),
        ("Four spherical elements: negative / positive / positive / negative; central stop.", 4),
        ("Three spherical elements: negative / positive / positive; stop after element 1.", 3),
        ("Five spherical elements: positive / negative / positive / negative / positive; central stop.", 5),
        ("Four spherical elements: positive / positive / negative / positive; rear stop.", 4),
        ("Three spherical elements: positive / positive / negative; front stop.", 3),
    ]
    candidates = []
    for design, (layout, count) in enumerate(layouts):
        metrics = {
            "effective_focal_length_mm": round(focal + rng.uniform(-1.8, 1.8) * brief["focal_length_tolerance_mm"], 3),
            "first_surface_to_sensor_mm": round(track * rng.uniform(0.82, 1.1), 3),
            "f_number": round(fnum + rng.uniform(-0.3, 0.3), 3),
            "mtf": [dict({k: v for k, v in t.items() if k != "minimum_mtf_fraction"},
                         mtf_fraction=round(max(0, min(1, t["minimum_mtf_fraction"] + rng.uniform(-0.18, 0.18))), 3))
                    for t in thresholds],
        }
        candidates.append(dict(
            candidate_id=design,
            prescription_ref=dict(uri=f"synthetic://prescriptions/{brief_id}/{design}", revision="v1"),
            layout=layout,
            glass_types=[rng.choice(["schott:N-BK7", "schott:F2", "hikari:SK16", "schott:N-LAK22"]) for _ in range(count)],
            nominal_metrics=metrics,
            geometry=dict(element_count=count, minimum_edge_thickness_mm=round(rng.uniform(0.4, 2.5), 3),
                          minimum_air_gap_mm=round(rng.uniform(0.08, 1.5), 3),
                          maximum_clear_aperture_mm=round(focal / fnum * rng.uniform(1.1, 1.6), 3),
                          maximum_absolute_curvature_per_mm=round(rng.uniform(0.025, 0.15), 4)),
        ))
    rng.shuffle(candidates)
    labels = list(range(1, 9))
    rng.shuffle(labels)
    for candidate, label in zip(candidates, labels, strict=True):
        candidate["candidate_id"] = label
    payload["candidates"] = candidates
    case = BriefCase.model_validate(payload)
    profile = synthetic_profile(case.brief.manufacturing_tolerance_profile_ref)
    # Even the fixture resolves by the entire immutable key, never by a URI prefix.
    profile = resolve_profile(profile.reference, {(profile.reference.uri, profile.reference.revision): profile})
    return LensEnvironment(case, profile)


def _mtf_key(point) -> tuple:
    return (point.field_angle_deg, point.spatial_frequency_lp_per_mm, point.wavelength_um, point.orientation)


def candidate_score(environment: LensEnvironment, candidate: Candidate) -> float:
    """Invented smooth utility in (0, 1], NOT ray tracing or measured yield.

    Penalty = normalized focal error + normalized f/# error + track excess
    + twice mean relative MTF shortfall + twice relative toy-yield shortfall.
    Toy yield = 1 / (1 + geometry stress), where stress increases with curvature,
    thin edges, narrow gaps, and extra elements, scaled by tolerance bounds.
    Final refocus reduces stress by 20% when allowed. Utility = 1/(1+penalty).
    A pair receives mean utility: both starters matter; no hidden answer key.
    """
    b, m, g = environment.case.brief, candidate.nominal_metrics, candidate.geometry
    if g is None:
        raise ValueError("synthetic grader requires geometry summaries")
    errors = {e.parameter: e for e in environment.tolerance_profile.errors}
    required = {"radius_error", "thickness_error", "decenter", "tilt", "index_error"}
    if not required <= errors.keys():
        raise ValueError("synthetic grader requires all five error parameters")
    def bound(name: str) -> float:
        return max(abs(errors[name].lower_bound), abs(errors[name].upper_bound))

    stress = (
        20 * bound("radius_error") * g.maximum_absolute_curvature_per_mm / 0.1
        + bound("thickness_error") / g.minimum_edge_thickness_mm
        + bound("decenter") / max(g.minimum_air_gap_mm, 0.02)
        + 0.1 * bound("tilt") * g.element_count
        + 20 * bound("index_error") * g.element_count
    )
    if (environment.tolerance_profile.refocus.allowed
            and environment.tolerance_profile.refocus.maximum_sensor_shift_mm > 0):
        stress *= 0.8
    toy_yield = 1 / (1 + stress)
    measured = {_mtf_key(p): p.mtf_fraction for p in m.mtf}
    mtf_shortfall = sum(max(0, t.minimum_mtf_fraction - measured.get(_mtf_key(t), 0)) /
                        max(t.minimum_mtf_fraction, 1e-6) for t in b.mtf_thresholds) / len(b.mtf_thresholds)
    penalty = (
        abs(m.effective_focal_length_mm - b.target_focal_length_mm) / max(b.focal_length_tolerance_mm, 1e-6)
        + abs(m.f_number - b.target_f_number) / max(b.f_number_tolerance, 1e-6)
        + max(0, m.first_surface_to_sensor_mm / b.max_first_surface_to_sensor_mm - 1) * 10
        + 2 * mtf_shortfall
        + 2 * max(0, b.minimum_simulated_yield_fraction - toy_yield) / max(b.minimum_simulated_yield_fraction, 1e-6)
    )
    return 1 / (1 + penalty)


Grader = Callable[[LensEnvironment, Submission], float]


def synthetic_grader(environment: LensEnvironment, submission: Submission) -> float:
    by_id = {c.candidate_id: c for c in environment.case.candidates}
    return sum(candidate_score(environment, by_id[i]) for i in submission.starter_ids) / 2


@dataclass(frozen=True)
class Grade:
    raw_response: str
    parsed_choices: tuple[int, int] | None
    reward: float
    error: str | None


def grade_response(environment: LensEnvironment, response: str, *, truncated: bool = False,
                   grader: Grader = synthetic_grader) -> Grade:
    """Keep the original response; never repair JSON or infer a partial choice."""
    try:
        submission = environment.case.parse_submission(response)
    except ValueError as exc:
        return Grade(response, None, 0, f"invalid submission: {exc}")
    if truncated:
        return Grade(response, submission.starter_ids, 0, "truncated or unclean model response")
    reward = float(grader(environment, submission))
    if not isfinite(reward):
        raise ValueError("grader returned a non-finite reward")
    return Grade(response, submission.starter_ids, reward, None)

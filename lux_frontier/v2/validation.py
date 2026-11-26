"""Small local checks for optimizer budgets, focus quantization, and first-order optics."""

from dataclasses import replace
import statistics
import time

import numpy as np

from lux_frontier.contracts import NominalMetrics
from lux_frontier.optical_engine import sag
from lux_frontier.v2.benchmark import Protocol, merit, refine, requirements
from lux_frontier.v2.optics import (
    PROFILES, REFERENCE_CLASSES, Lens, Requirements, build_optic, failures, measurements,
    paraxial_matrix, reference, simulate_build,
)


def independent_sensor_intersection(p, optic, origin, direction):
    """Trace a centered spherical stack with explicit sphere intersections and Snell's law."""
    point = np.array(origin, dtype=float)
    direction = np.array(direction, dtype=float)
    z, before = 0., 1.
    for k, radius in enumerate(p.radii_mm):
        center = np.array([0., 0., z+radius])
        offset = point-center
        b = float(np.dot(offset, direction))
        delta = b*b - (float(np.dot(offset, offset))-radius*radius)
        if delta < 0:
            raise ValueError("independent ray missed a spherical surface")
        distance = -b - np.copysign(np.sqrt(delta), radius)
        point += distance*direction
        height = float(np.linalg.norm(point[:2]))
        if height >= min(p.clear_semidiameters_mm[k//2], p.stop_diameter_mm/2 if k+1 == p.stop_surface else np.inf):
            raise ValueError("independent validation ray was clipped")
        if distance < -1e-8 or abs(point[2]-z-sag(radius, height)) > 1e-8:
            raise ValueError("independent ray is on the wrong spherical cap")
        normal = (point-center)/radius
        cosine = -float(np.dot(direction, normal))
        after = float(np.asarray(optic.surfaces[k+1].material_post.n(p.primary_wavelength_um)).item())
        ratio = before/after
        transmitted = np.sqrt(1-ratio*ratio*(1-cosine*cosine))
        direction = ratio*direction + (ratio*cosine-transmitted)*normal
        before = after
        if k < len(p.spacings_mm):
            z += p.spacings_mm[k]
    return point + (p.sensor_z_mm-point[2])/direction[2]*direction


def compare_sensor_traces(p, lens):
    from optiland.rays.real_rays import RealRays

    rows = []
    for origin, slope in (((.21, -.13, -1.), (0., 0.)), ((-.32, .71, -1.), (0., .012)),
                          ((.12, .18, -1.), (.01, -.009))):
        vector = np.array([*slope, 1.])
        direction = vector/np.linalg.norm(vector)
        rays = RealRays(*origin, *direction, intensity=1., wavelength=p.primary_wavelength_um)
        lens.surfaces.trace(rays, skip=1)
        actual = np.array([rays.x[0], rays.y[0], rays.z[0]])
        separate = independent_sensor_intersection(p, lens, origin, direction)
        delta = float(np.linalg.norm(actual-separate))
        if not np.isfinite(delta) or delta > 1e-8 or rays.i[0] <= 0:
            raise ValueError(f"sensor intersection disagreement: {delta} mm")
        rows.append(dict(origin_mm=origin, direction=direction.tolist(), independent_mm=separate.tolist(),
                         optiland_mm=actual.tolist(), delta_mm=delta))
    return rows


def validate_prescription(p, pupil_samples=32):
    profile = PROFILES["precision-fixed"]
    brief = requirements(p, profile, (1, 0., .02, 0., .12, .02, .65), p.ancestry)
    optic, geometry = build_optic(p)
    matrix = paraxial_matrix(p, optic)
    measured = measurements(p, brief, pupil_samples)
    real = NominalMetrics.model_validate(measured["metrics"]).effective_focal_length_mm
    delta = abs(real-matrix["efl_mm"])
    if delta >= 1e-4 or not all(value > 0 for value in geometry.values()):
        raise ValueError(f"reference failed focal length or clearance validation: {p.ancestry}, {delta}")
    return dict(reference=p.ancestry, geometry=geometry, abcd=matrix, real_ray_efl_mm=real,
                         absolute_efl_error_mm=delta, physical_failures=failures(measured, brief),
                         independent_sensor_intersections=compare_sensor_traces(p, optic))


def validate_references(pupil_samples=32):
    return [validate_prescription(reference(cls.__name__), pupil_samples) for cls in REFERENCE_CLASSES]


def pair_order_similarity(values_a, values_b):
    def ordering(values, i, j):
        return np.sign(values[i]-values[j])
    pairs = [(i, j) for i in range(len(values_a)) for j in range(i)]
    return float(np.mean([ordering(values_a, i, j) == ordering(values_b, i, j) for i, j in pairs]))


def study(task, *, builds=16):
    started = time.perf_counter()
    protocol = Protocol.model_validate(task["protocol"])
    brief = Requirements.model_validate(task["brief"])
    budgets = sorted({7, protocol.evaluations, 49})
    optimization = {}
    for budget in budgets:
        start = time.perf_counter()
        outcomes = []
        for c in task["candidates"]:
            final = task["outcomes"][str(c["id"])]
            fit = refine(Lens.model_validate(c["prescription"]), brief, protocol.model_copy(update={"evaluations": budget}))["final"]
            score, failure = fit["score"], bool(fit["failures"])
            if budget == protocol.evaluations and (score != final["objective"] or failure == final["nominal_pass"]):
                raise ValueError("repeat optical refinement differs from frozen task")
            outcomes.append(dict(starter_id=c["id"], merit=score, passes=not failure))
        optimization[str(budget)] = dict(candidates=outcomes, seconds=time.perf_counter()-start)
    canonical = optimization[str(protocol.evaluations)]["candidates"]
    for result in optimization.values():
        result["agreement_with_pilot_pair_order"] = pair_order_similarity(
            [r["merit"] for r in canonical], [r["merit"] for r in result["candidates"]])
    for result in optimization.values():
        result["nominal_passes"] = sum(x["passes"] for x in result["candidates"])

    feasible = [o for o in task["outcomes"].values() if o["nominal_pass"]]
    if not feasible:
        raise ValueError("focus study needs a nominally feasible design")
    p = Lens.model_validate(min(feasible, key=lambda row: row["objective"])["refined_prescription"])
    profile = PROFILES["standard-adjusted"]
    focus_results = {}
    for increment in (0., .06, .03, .015, .0075, .00075):
        check = replace(profile, focus_increment_mm=increment or profile.focus_increment_mm,
                        focus_travel_mm=profile.focus_travel_mm if increment else 0.)
        rows = [simulate_build(p, brief, protocol.pupil_samples, check, [protocol.seed, 5991, b]) for b in range(builds)]
        focus_results[str(increment)] = dict(passed=sum(r["passed"] for r in rows), tested=builds,
            focus_mm=[r["sensor_shift_mm"] for r in rows],
            best_unconstrained_focus_mm=[r["unconstrained_best_rms_shift_mm"] for r in rows])
    convergence = {}
    for count in (16, 32, 64, 128):
        measured = measurements(p, brief, count)
        convergence[str(count)] = dict(mtf=[m["mtf_fraction"] for m in measured["metrics"]["mtf"]],
                                       nominal_pass=not failures(measured, brief), merit=merit(measured, brief)[0])
    return dict(task_id=task["task_id"], optimizer=optimization, focus_quantization=focus_results,
                pupil_samples=convergence, references=validate_references(protocol.pupil_samples), runtime_seconds=time.perf_counter()-started)


def ranking_convergence(task, pupil_grids=(64, 128)):
    """Compare all frozen final nominal designs, then refit at shorter and longer budgets."""
    started = time.perf_counter()
    protocol = Protocol.model_validate(task["protocol"])
    brief = Requirements.model_validate(task["brief"])
    outcomes = [task["outcomes"][str(c["id"])] for c in task["candidates"]]
    canonical = [r["objective"] for r in outcomes]
    fixed = {}
    for size in pupil_grids:
        measurements_at_size = [measurements(Lens.model_validate(r["refined_prescription"]), brief, size) for r in outcomes]
        scores = [merit(m, brief)[0] for m in measurements_at_size]
        nominal = [not failures(m, brief) for m in measurements_at_size]
        fixed[size] = dict(pair_order_agreement=pair_order_similarity(scores, canonical), nominal=nominal,
                           nominal_flips=sum(x != r["nominal_pass"] for x, r in zip(nominal, outcomes, strict=True)),
                           maximum_mtf_delta=max(abs(a["mtf_fraction"]-b["mtf_fraction"])
                                                 for m, r in zip(measurements_at_size, outcomes, strict=True)
                                                 for a, b in zip(m["metrics"]["mtf"], r["final"]["metrics"]["mtf"], strict=True)))
    refits = {}
    for budget in (7, 49):
        fitted = [refine(Lens.model_validate(c["prescription"]), brief,
                         protocol.model_copy(update={"evaluations": budget}))["final"] for c in task["candidates"]]
        nominal = [not item["failures"] for item in fitted]
        refits[budget] = dict(pair_order_agreement=pair_order_similarity([r["score"] for r in fitted], canonical),
                              nominal=nominal, nominal_flips=sum(x != r["nominal_pass"] for x, r in zip(nominal, outcomes, strict=True)))
    return dict(task_id=task["task_id"], candidate_order=[c["id"] for c in task["candidates"]],
                canonical_pupil_grid=protocol.pupil_samples, canonical_budget=protocol.evaluations,
                canonical_nominal=[r["nominal_pass"] for r in outcomes], fixed_design_resolution=fixed, refits=refits,
                wall_seconds=time.perf_counter()-started)

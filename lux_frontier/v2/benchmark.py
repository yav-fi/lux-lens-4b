"""Deterministic task recipes, exhaustive optical grading, and curriculum certification."""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import importlib.metadata
import json
import math
from pathlib import Path
import random
import subprocess
import time
from uuid import uuid4

import numpy as np
from pydantic import Field

from lux_frontier.contracts import Contract, NominalMetrics
from lux_frontier.optical_engine import InvalidBuild
from lux_frontier.optical_grading import OpticalCache
from lux_frontier.v2.grading import PHASES, VERSION, digest, grade, load_envelope, submit
from lux_frontier.v2.optics import (
    OPTICS_VERSION, PROFILES, Lens, Requirements, build_optic, clearance, failures,
    measurements, paraxial_matrix, reference, simulate_build,
)

SPLITS = {"train": "CookeTriplet", "development": "TripletTelescopeObjective", "test": "TelescopeDoublet"}
FAMILIES = ("A", "B", "C", "D", "E")
REPRESENTATIONS = ("summary", "full", "physics")
SCENARIOS = (
    # field degrees, focal target bias, half-width, F/# bias, F/# half-width, track slack, MTF severity
    (1, 0., .028, 0., .12, .028, .68),
    (3, 0., .030, 0., .13, .014, .62),
    (7, 0., .045, 0., .17, .024, .58),
    (1, -.018, .023, -.07, .06, .005, .73),
    (3, .018, .023, .07, .075, -.003, .73),
    (7, 0., .021, 0., .08, .003, .83),
    (14, .06, .008, -.09, .05, -.021, .88),
)
CONTRAST_SCENARIOS = (
    (3, 0., .013, 0., .11, .014, .91),
    (7, 0., .015, 0., .10, .020, .93),
    (14, 0., .018, 0., .10, .020, 1.04),
    (1, -.009, .012, -.035, .085, .007, .96),
    (7, .007, .015, .06, .10, .008, .92),
    (14, 0., .018, .055, .115, .008, .96),
    (20, 0., .02, 0., .11, .015, .78),
    (3, 0., .016, -.01, .09, .003, 1.13),
)


class Protocol(Contract):
    seed: int = Field(default=17291, ge=0)
    evaluations: int = Field(default=25, ge=1, le=301)
    pupil_samples: int = Field(default=32, ge=16, le=256)
    builds_per_phase: int = Field(default=8, ge=2, le=10000)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def write_envelope(path, value, key):
    write_json(path, {key: value, "sha256": digest(value)})


def load_replay_details(manifest, task):
    details = load_envelope(Path(manifest).parent/"replay-data"/(task["task_id"]+".json"), "details")
    expected = task.get("replay_details_sha256")
    if expected is not None:
        if digest(details) != expected:
            raise ValueError("replay details differ from frozen task")
        return details
    # Early V2 pilots predate a direct link. Corroborate their independently
    # checksummed replay with frozen optimizer objectives and build counts.
    for candidate in task["candidates"]:
        key = str(candidate["id"])
        outcome, replay = task["outcomes"][key], details[key]
        trial = min((r for r in replay["trajectory"] if r["measured"] is not None),
                    key=lambda r: (bool(r["failures"]), r["score"]))
        if (trial["prescription"] != outcome["refined_prescription"] or trial["score"] != outcome["objective"]
                or replay["trajectory"][0]["prescription"] != candidate["prescription"]):
            raise ValueError("legacy replay does not match frozen optimizer result")
        for phase, counts in outcome["simulations"].items():
            rows = replay["builds"][phase]
            if (sum(b["passed"] for b in rows), len(rows)) != (counts["passed"], counts["tested"]):
                raise ValueError("legacy replay does not match frozen manufacturing result")
    return details


def provenance():
    root = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL, text=True).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        commit = None
    research_modules = [p for p in Path(__file__).parent.glob("*.py") if p.name != "serve.py"]
    modules = [*research_modules, root/"lux_frontier"/"optical_engine.py",
               root/"lux_frontier"/"optical_grading.py", root/"lux_frontier"/"contracts.py",
               root/"lux_frontier"/"grpo.py", root/"lux_frontier"/"tinker_backend.py", root/"pyproject.toml", root/"uv.lock"]
    return dict(source_commit=commit, source_hashes={str(p.relative_to(root)): digest(p.read_text()) for p in sorted(modules) if p.is_file()},
                dependencies={d.metadata["Name"]: d.version for d in importlib.metadata.distributions()})


def requirements(base, profile, scenario, task_id):
    f, bias, width, fno_bias, fno_width, track, severity = scenario
    # Use common application frequencies for both candidates and all phases.
    optic, _ = build_optic(base)
    focal = float(paraxial_matrix(base, optic)["efl_mm"])
    from optiland.utils import get_working_FNO
    fno = float(np.asarray(get_working_FNO(optic, (0, 0), base.primary_wavelength_um)).item())
    points = [(0, .55, "sagittal", 30, .68), (f, .55, "tangential", 30, .37),
              (f, .55, "sagittal", 30, .44), (f, .65, "tangential", 20, .32)]
    return Requirements(brief_id=task_id, target_focal_length_mm=focal*(1+bias), focal_length_tolerance_mm=focal*width,
                        max_first_surface_to_sensor_mm=base.sensor_z_mm*(1+track),
                        target_f_number=fno*(1+fno_bias), f_number_tolerance=fno*fno_width,
                        mtf_thresholds=[dict(field_angle_deg=(0, field), wavelength_um=wave, orientation=direction,
                                             spatial_frequency_lp_per_mm=freq, minimum_mtf_fraction=level*severity)
                                        for field, wave, direction, freq, level in points],
                        minimum_simulated_yield_fraction=.625, minimum_pupil_throughput=.50,
                        manufacturing_tolerance_profile_ref=dict(uri="local://lux/v2/profiles/"+digest(profile.record()), revision=OPTICS_VERSION))


def candidate_pool(base, seed, pool_index, *, contrast=False):
    """Controlled design variants of a documented reference, with explicit provenance."""
    rng = random.Random(digest([seed, base.ancestry, pool_index, "geometry"]))
    candidates = []
    for index in range(8):
        accepted = None
        for _ in range(60):
            if index == 0:
                accepted = base
                break
            # Radius perturbations vary the power distribution, while common
            # axial scaling and f/# variation keep some candidates near target.
            radius_width, spacing_width = (.006, .04) if contrast else (.032, .10)
            radius = [r*(1+rng.uniform(-radius_width, radius_width)) for r in base.radii_mm]
            spacing = [t*(1+rng.uniform(-spacing_width, spacing_width)) for t in base.spacings_mm]
            sensor = base.sensor_z_mm + rng.choice((-1, 1))*rng.uniform(.018, .26)
            stop = base.stop_diameter_mm*(1+rng.uniform(-.07, .07))
            try:
                accepted = Lens.model_validate(dict(base.model_dump(), radii_mm=radius, spacings_mm=spacing,
                                                    sensor_z_mm=sensor, stop_diameter_mm=stop))
            except ValueError:
                continue
            break
        if accepted is None:
            raise ValueError("candidate geometry rejection budget exhausted")
        candidates.append(accepted)
    return candidates


def spec_terms(measured, brief):
    metric = measured["metrics"]
    focal = abs(metric["effective_focal_length_mm"]-brief.target_focal_length_mm)/max(brief.focal_length_tolerance_mm, 1e-8)
    fno = abs(metric["f_number"]-brief.target_f_number)/max(brief.f_number_tolerance, 1e-8)
    track = max(0., metric["first_surface_to_sensor_mm"]-brief.max_first_surface_to_sensor_mm)/.25
    mtf = [max(0., 1-m["mtf_fraction"]/max(t.minimum_mtf_fraction, 1e-8))
           for m, t in zip(metric["mtf"], brief.mtf_thresholds, strict=True)]
    pupil = [max(0., brief.minimum_pupil_throughput-r["pupil_fraction"]) for r in measured["throughput"]]
    return focal, fno, track, mtf, pupil


def merit(measured, brief):
    focal, fno, track, mtf, pupil = spec_terms(measured, brief)
    violation = sum(v*v for v in [max(0, focal-1), max(0, fno-1), track, *mtf, *pupil])
    quality = sum((1-v["mtf_fraction"])**2 for v in measured["metrics"]["mtf"])
    return 100*violation + .004*(focal*focal + fno*fno) + .1*quality, violation


def shape(p, vector):
    sizes = (len(p.radii_mm), len(p.spacings_mm))
    n, m = sizes
    return p.model_copy(update=dict(sensor_z_mm=float(vector[0]),
                                   radii_mm=tuple(r*(1+x) for r, x in zip(p.radii_mm, vector[1:n+1], strict=True)),
                                   spacings_mm=tuple(t*(1+x) for t, x in zip(p.spacings_mm, vector[n+1:n+m+1], strict=True))))


def refine(p, brief, protocol):
    """Bounded greedy coordinate search with coupled sensor/curvature trials."""
    n, m = len(p.radii_mm), len(p.spacings_mm)
    origin = np.array([p.sensor_z_mm, *([0.]*(n+m))])
    bounds = np.array([.50, *([.040]*n), *([.12]*m)])
    steps = np.array([.085, *([.009]*n), *([.04]*m)])
    # Include both signs about the same incumbent, then pick the best. Interleave
    # paired surface, glass thickness, and air-gap moves early in the budget.
    coordinates = [0, 1, 1+n, 2, 2+n, n, *[k for k in range(3, n) if k != n], *range(n+3, n+m+1)]
    history = []

    def evaluate(x, axis, coupled_focus=False):
        design = shape(p, x)
        if coupled_focus:
            lens, _ = build_optic(design)
            bfl = paraxial_matrix(design, lens)["paraxial_bfl_mm"] + sum(design.spacings_mm)
            x[0] = float(np.clip(x[0]+bfl-best_paraxial, origin[0]-.5, origin[0]+.5))
            design = shape(p, x)
        row = dict(parameters=x.tolist(), prescription=design.model_dump(mode="json"), axis=axis)
        try:
            value = measurements(design, brief, protocol.pupil_samples)
            score, violation = merit(value, brief)
            reasons = failures(value, brief)
            row.update(measured=value, score=score, violation=violation, failures=reasons)
        except InvalidBuild as exc:
            row.update(measured=None, score=None, violation=None, failures=[dict(constraint=str(exc))])
        history.append(row)
        return row

    best = evaluate(origin.copy(), "initial")
    initial = best
    if best["measured"] is None:
        raise InvalidBuild("starter:" + best["failures"][0]["constraint"])
    for i in range(0, protocol.evaluations-1, 2):
        k = coordinates[(i//2) % len(coordinates)]
        scale = .55**(i//(2*len(coordinates)))
        center = np.array(best["parameters"])
        if 0 < k <= n:
            inc = shape(p, center)
            incumbent, _ = build_optic(inc)
            best_paraxial = paraxial_matrix(inc, incumbent)["paraxial_bfl_mm"]+sum(inc.spacings_mm)
        trials = []
        for sign in (-1, 1)[:min(2, protocol.evaluations-1-i)]:
            x = center.copy()
            x[k] = np.clip(x[k]+sign*steps[k]*scale, origin[k]-bounds[k], origin[k]+bounds[k])
            try:
                trials.append(evaluate(x, k, 0 < k <= n))
            except InvalidBuild as exc:
                history.append(dict(parameters=x.tolist(), prescription=shape(p, x).model_dump(mode="json"),
                                    axis=k, measured=None, score=None, violation=None, failures=[dict(constraint=str(exc))]))
        best = min([best, *[x for x in trials if x["measured"]]],
                   key=lambda x: (bool(x["failures"]), x["score"]))
    return dict(initial=initial, final=best, trajectory=history)


def sensitivity(p, brief, protocol, initial):
    # Standardized one-factor probes independent of tasks' manufacturing draws.
    from lux_frontier.v2.optics import zero_errors
    from dataclasses import replace
    zero = zero_errors(p)
    probes = {"sensor_plus_50um": (zero, .05),
              "front_radius_plus_0p1pct": (replace(zero, radius_fraction=(.001, *zero.radius_fraction[1:])), 0.),
              "front_decenter_x_plus_10um": (replace(zero, decenter_mm=((.01, 0.), *zero.decenter_mm[1:])), 0.)}
    result = {}
    baseline = [row["mtf_fraction"] for row in initial["metrics"]["mtf"]]
    for name, (errors, focus) in probes.items():
        try:
            measured = measurements(p, brief, protocol.pupil_samples, errors=errors, focus=focus)
            changed = [row["mtf_fraction"] for row in measured["metrics"]["mtf"]]
            result[name] = dict(mtf_delta=[a-b for a, b in zip(changed, baseline, strict=True)],
                                focal_delta_mm=measured["metrics"]["effective_focal_length_mm"]-initial["metrics"]["effective_focal_length_mm"])
        except InvalidBuild:
            result[name] = None
    return result


def render(task, representation="physics", *, format_example=False):
    if representation not in REPRESENTATIONS:
        raise ValueError("unknown prompt representation")
    brief = task["brief"]
    profile = task["profile"]
    lines = ["Select exactly one independent lens starter for the camera brief.",
             "Each candidate is refined individually. Prefer a finished design that meets the optical, physical and manufacturing requirements.",
             f"EFL {brief['target_focal_length_mm']:.4g} ± {brief['focal_length_tolerance_mm']:.4g} mm; "
             f"working f/# {brief['target_f_number']:.4g} ± {brief['f_number_tolerance']:.4g}; "
             f"first vertex to sensor ≤ {brief['max_first_surface_to_sensor_mm']:.4g} mm.",
             f"Each listed field/wavelength must transmit at least {brief['minimum_pupil_throughput']:.3g} of sampled pupil rays; "
             f"manufacturing yield target {task['yield_target']:.3g}. Object at infinity.",
             "MTF targets in table order (field Y deg, wavelength µm, slice, lp/mm, minimum):"]
    for point in brief["mtf_thresholds"]:
        lines.append(f"{point['field_angle_deg'][1]:g} {point['wavelength_um']:g} {point['orientation']} "
                     f"{point['spatial_frequency_lp_per_mm']:g} {point['minimum_mtf_fraction']:.3g}")
    lines.extend([
        f"Refinement uses {task['protocol']['evaluations']} objective trials, including the initial lens. "
        "Bounds from each starter: sensor ±0.50 mm, each radius ±4%, each center thickness and air gap ±12%. "
        "Glass, clear apertures and stop are fixed; physically invalid updates are rejected.",
        f"Provisional manufacturing profile {profile['name']}. {profile['description']}",
        f"Radius σ {profile['radius_sigma']:g} fractional; thickness ±{profile['thickness_bound_mm']:g} mm; "
        f"gap ±{profile['gap_bound_mm']:g} mm; decenter σ {profile['decenter_sigma_mm']:g} mm per axis; "
        f"tilt ±{profile['tilt_bound_deg']:g}° per axis; sensor ±{profile['focus_travel_mm']:g} mm in "
        f"{profile['focus_increment_mm']:g} mm increments.",
        "Initial data per candidate: number of elements; EFL mm, f/#, track mm; minimum glass and air clearance mm; "
        "MTFs and pupil throughput in target order (throughput listed once per field/wavelength).",
    ])
    def numbers(values):
        return " ".join(f"{v:.5g}" for v in values)
    for c in task["candidates"]:
        p, first = c["prescription"], c["initial"]
        m = first["metrics"]
        lines.append(f"Candidate {c['id']} | {len(p['glasses'])} | "
                     + numbers([m["effective_focal_length_mm"], m["f_number"], m["first_surface_to_sensor_mm"]]) + " | "
                     + numbers([first["minimum_glass_thickness_mm"], first["minimum_air_clearance_mm"]]) + " | "
                     + numbers([x["mtf_fraction"] for x in m["mtf"]]) + " | "
                     + numbers([x["pupil_fraction"] for x in first["throughput"]]))
        if representation != "summary":
            lines.append("Object to sensor: glasses " + " ".join(f"{g[0]}/{g[1]}" for g in p["glasses"])
                         + "; signed radii mm " + numbers(p["radii_mm"])
                         + "; alternating center thickness/air gap mm " + numbers(p["spacings_mm"])
                         + "; clear semidiameters mm " + numbers(p["clear_semidiameters_mm"])
                         + f"; stop on surface {p['stop_surface']} diameter {p['stop_diameter_mm']:.5g} mm.")
        if representation == "physics":
            lines.append("Probe changes (EFL mm then MTF): " + "; ".join(
                f"{name} " + (numbers([probe["focal_delta_mm"], *probe["mtf_delta"]]) if probe else "invalid")
                for name, probe in c["sensitivity"].items()))
    lines.append('Return only a JSON object with a single integer field named "starter_id".')
    if format_example:
        lines.append('Example of JSON syntax using an unavailable ID: {"starter_id":0}. Replace 0 with your selected candidate ID.')
    return "\n".join(lines)


def heuristic(candidate, brief, *, physics=False):
    focal, fno, track, mtf, pupil = spec_terms(candidate["initial"], brief)
    score = focal*focal + fno*fno + .02*track*track + np.mean(np.square(mtf))
    if physics:
        probes = [1. if row is None else np.mean(np.abs(row["mtf_delta"])) for row in candidate["sensitivity"].values()]
        # Robust proxy penalizes sensitivity and low-throughput candidates,
        # while placing greater weight on initial margins than raw proximity.
        score = (max(0, focal-.5)**2 + max(0, fno-.5)**2 + .08*track*track + 4*np.mean(np.square(mtf))
                 + 4*np.mean(np.square(pupil)) + 2*np.mean(probes))
    return float(score)


def characterize(task):
    brief = Requirements.model_validate(task["brief"])
    grades = {c["id"]: grade(task, submit(c["id"]), phase="construction") for c in task["candidates"]}
    rewards = {i: g.reward for i, g in grades.items()}
    ranked = sorted(task["candidates"], key=lambda c: (heuristic(c, brief), c["identity"]))
    strong = min(task["candidates"], key=lambda c: (heuristic(c, brief, physics=True), c["identity"]))
    ids = sorted(rewards)
    best = max(rewards.values())
    winners = [i for i in ids if rewards[i] == best]
    eligible = [i for i in ids if grades[i].nominal_pass]
    row = task["outcomes"]
    first_mtf = {c["id"]: np.mean([v["mtf_fraction"] for v in c["initial"]["metrics"]["mtf"]]) for c in task["candidates"]}
    final_mtf = {i: np.mean([v["mtf_fraction"] for v in row[str(i)]["final"]["metrics"]["mtf"]]) for i in eligible}
    reversal_pairs = [(i, j) for i in eligible for j in eligible if first_mtf[i]-first_mtf[j] >= .02
                      and final_mtf[j]-final_mtf[i] >= .02]
    yield_pairs = [(i, j) for i in eligible for j in eligible if final_mtf[i] >= final_mtf[j]
                   and grades[j].yield_fraction-grades[i].yield_fraction >= .25]
    qualifying = [i for i in ids if grades[i].final_pass]
    families = []
    if 0 < len(eligible) < len(ids):
        families.append("A")
    if reversal_pairs:
        families.append("B")
    if yield_pairs:
        families.append("C")
    if qualifying and best-min(rewards.values()) >= .12:
        families.append("E")
    rank_reversal = any(heuristic(a, brief) < heuristic(b, brief) and
                        row[str(a["id"])]["objective"] > row[str(b["id"])]["objective"]
                        for a in task["candidates"] for b in task["candidates"])
    sharpest = max(task["candidates"], key=lambda c: (first_mtf[c["id"]], c["identity"]))["id"]
    return dict(families=families, nominally_feasible=len(eligible), yield_target_passes=len(qualifying),
                best_reward=best, random_expected_reward=float(np.mean(list(rewards.values()))),
                spread=best-min(rewards.values()), best_ids=winners, target_pass_ids=qualifying,
                heuristic_id=ranked[0]["id"], heuristic_reward=rewards[ranked[0]["id"]],
                heuristic_regret=best-rewards[ranked[0]["id"]], physics_id=strong["id"],
                physics_heuristic_reward=rewards[strong["id"]], physics_heuristic_regret=best-rewards[strong["id"]],
                initial_sharpest_id=sharpest, initial_sharpest_regret=best-rewards[sharpest], optimization_rank_reversal=rank_reversal,
                mtf_reversals=reversal_pairs, manufacturing_reversals=yield_pairs,
                nominal_quality_order=sorted(eligible, key=lambda i: (-final_mtf[i], i)),
                construction_rewards=rewards, base_model=None)


def prepare_group(args):
    split, index, pool_index, scenario, profile_names, protocol, root, source_key, contrast, lineage = args
    base = Lens.model_validate(lineage["prescription"]) if lineage else reference(SPLITS[split])
    pool = candidate_pool(base, protocol.seed, pool_index, contrast=contrast)
    prefix = f"{split}-{digest(base.ancestry)[:12]}" if lineage else split
    cohort = f"{prefix}-cohort{pool_index:03d}" + ("-contrast" if contrast else "")
    group_id = f"{cohort}-spec{index:03d}"
    cache = OpticalCache(root/"cache")
    forms = {}
    rejected = []
    tasks = []
    for profile_name in profile_names:
        profile = PROFILES[profile_name]
        task_id = f"{group_id}-{profile.name}"
        brief = requirements(base, profile, scenario, task_id)
        physical = brief.model_dump(mode="json", exclude={"brief_id", "manufacturing_tolerance_profile_ref"})
        if not forms:
            for p in pool:
                key = dict(version=OPTICS_VERSION, source_key=source_key, prescription=p, constraints=physical,
                           evaluations=protocol.evaluations, pupil_samples=protocol.pupil_samples)
                try:
                    def compute():
                        fitted = refine(p, brief, protocol)
                        probes = sensitivity(p, brief, protocol, fitted["initial"]["measured"])
                        return {"fit": fitted, "sensitivity": probes}
                    forms[digest(p.model_dump(mode="json"))], _ = cache.compute("fit", key, compute)
                except InvalidBuild as exc:
                    rejected.append(dict(group=group_id, candidate=p.model_dump(mode="json"), reason=str(exc)))
            if rejected:
                return [], rejected
        identities = list(forms)
        ids = random.Random(digest([protocol.seed, task_id, "local-id"])).sample(range(1, 9), 8)
        order = list(range(8))
        random.Random(digest([protocol.seed, task_id, "display-order"])).shuffle(order)
        candidates, outcomes, replay_details = [], {}, {}
        for k in order:
            p, local_id = pool[k], ids[k]
            identity = identities[k]
            form = forms[identity]
            fitted = form["fit"]
            initial, final = fitted["initial"]["measured"], fitted["final"]
            candidate = dict(id=local_id, identity=identity, prescription=p.model_dump(mode="json"),
                             initial=initial, sensitivity=form["sensitivity"])
            candidates.append(candidate)
            outcome = dict(nominal_pass=not final["failures"], objective=final["score"],
                           near_miss=.5/(1+12*math.sqrt(final["violation"])), final=final["measured"],
                           refined_prescription=final["prescription"], failures=final["failures"], simulations={})
            records = {}
            for phase_index, phase in enumerate(PHASES):
                if outcome["nominal_pass"]:
                    stream_id = int(digest([group_id, "errors"])[:8], 16)
                    design = Lens.model_validate(final["prescription"])
                    # Profiles and candidates within a counterfactual group share
                    # random quantiles. The four purposes have disjoint streams.
                    settings = dict(version=OPTICS_VERSION, source_key=source_key, prescription=design,
                                    constraints=physical, profile=profile.record(), phase=phase,
                                    stream_id=stream_id, seed=protocol.seed, count=protocol.builds_per_phase,
                                    pupil_samples=protocol.pupil_samples)
                    def compute_builds():
                        return [simulate_build(design, brief, protocol.pupil_samples, profile,
                                               [protocol.seed, stream_id, phase_index+1, b])
                                for b in range(protocol.builds_per_phase)]
                    builds, _ = cache.compute("tolerancing", settings, compute_builds)
                    records[phase] = builds
                    outcome["simulations"][phase] = dict(passed=sum(b["passed"] for b in builds), tested=len(builds))
            outcomes[str(local_id)] = outcome
            replay_details[str(local_id)] = dict(trajectory=fitted["trajectory"], builds=records)
        task = dict(version=VERSION, optics_version=OPTICS_VERSION, task_id=task_id, split=split,
                    group=group_id, cohort=cohort, brief=brief.model_dump(mode="json"),
                    profile=profile.record(), protocol=protocol.model_dump(mode="json"),
                    yield_target=brief.minimum_simulated_yield_fraction, candidates=candidates, outcomes=outcomes)
        task["construction"] = characterize(task)
        task["replay_details_sha256"] = digest(replay_details)
        write_envelope(root/"replay-data"/(task_id+".json"), replay_details, "details")
        tasks.append(task)
    for task in tasks:
        current = task["construction"]
        best = {c["identity"] for c in task["candidates"] if c["id"] in current["best_ids"]}
        for other in tasks:
            alternate = other["construction"]
            other_best = {c["identity"] for c in other["candidates"] if c["id"] in alternate["best_ids"]}
            if current["yield_target_passes"] and alternate["yield_target_passes"] and best.isdisjoint(other_best):
                current.setdefault("profile_switches", []).append(other["task_id"])
        if current.get("profile_switches"):
            current["families"].insert(-1 if "E" in current["families"] else len(current["families"]), "D")
        task["capability"] = "E" in current["families"]
        task["representative"] = True
        task["capability_rejection"] = None if task["capability"] else (
            "no_nominal_solution" if not current["nominally_feasible"] else
            "no_yield_target_solution" if not current["yield_target_passes"] else "insufficient_reward_spread")
    return tasks, []


def validate_splits(tasks):
    from collections import defaultdict
    owners = defaultdict(set)
    task_ids = set()
    for task in tasks:
        if task["task_id"] in task_ids:
            raise ValueError("duplicate task identity")
        task_ids.add(task["task_id"])
        keys = [task["group"], task["cohort"]]
        for c in task["candidates"]:
            p = c["prescription"]
            scale = p["sensor_z_mm"]
            geometry = [round(v/scale, 2) for v in [*p["radii_mm"], *p["spacings_mm"], p["stop_diameter_mm"]]]
            keys.extend((c["identity"], p["ancestry"], digest([geometry, p["stop_surface"]])))
        for key in keys:
            owners[key].add(task["split"])
    collisions = [key for key, splits in owners.items() if len(splits) > 1]
    if collisions:
        raise ValueError(f"split leakage in {len(collisions)} identities, lineages, or groups")


def generate(root, protocol, *, workers=1, scenarios=7, pools=1, profiles=None, contrast_parent=None, lineages=None):
    if (root/"manifest.json").exists():
        raise ValueError("manifest already published; use a new output directory")
    if min(workers, pools, scenarios) <= 0 or protocol.pupil_samples & (protocol.pupil_samples-1):
        raise ValueError("positive workers/pools/scenarios and a power-of-two pupil grid are required")
    profiles = tuple(profiles or ("precision-fixed", "standard-adjusted"))
    if len(set(profiles)) != len(profiles) or any(name not in PROFILES for name in profiles):
        raise ValueError("unknown or duplicate manufacturing profiles")
    started = time.perf_counter()
    previous = None
    prior_tasks = []
    if lineages:
        if contrast_parent is not None:
            raise ValueError("use a separate manifest for an externally sourced lineage panel")
        from lux_frontier.v2.validation import validate_prescription
        if len({row["prescription"]["ancestry"] for row in lineages}) != len(lineages):
            raise ValueError("each supplied independent lineage must be unique")
        for row in lineages:
            if row["split"] not in SPLITS or not row.get("source"):
                raise ValueError("each lineage needs a split and documented prescription source")
            row["validation"] = validate_prescription(Lens.model_validate(row["prescription"]), protocol.pupil_samples)
    if contrast_parent is not None:
        previous = load_envelope(contrast_parent, "manifest")
        if previous["protocol"] != protocol.model_dump(mode="json"):
            raise ValueError("contrast extension requires identical physical budgets and seeds")
        prior_tasks = load_tasks(contrast_parent)
    source = provenance()
    source_key = digest(dict(source=source["source_hashes"], dependencies=source["dependencies"]))
    jobs = []
    contrasts = contrast_parent is not None
    split_names = ("train",) if contrasts else tuple(SPLITS)
    recipes = CONTRAST_SCENARIOS if contrasts else SCENARIOS
    if contrasts and profiles == ("precision-fixed", "standard-adjusted"):
        profiles = ("standard-fixed", "standard-adjusted")
    entries = [(row["split"], row) for row in lineages] if lineages else [(s, None) for s in split_names]
    for split_index, (split, lineage) in enumerate(entries):
        for pool in range(pools):
            for index in range(scenarios):
                # Train, development and test use different focal/f-number
                # centers and fields, even before accounting for the ancestry.
                scenario = list(recipes[index % len(recipes)])
                if index >= len(recipes):
                    rng = random.Random(digest([protocol.seed, "crossed" if lineage else split, index, "brief"]))
                    scenario[1] += rng.uniform(-.012, .012)
                    scenario[6] *= rng.uniform(.85, 1.15)
                if not lineage:
                    scenario[1] += (.0, -.002, .004)[split_index]
                    scenario[3] += (.0, .013, -.008)[split_index]
                selected_profiles = profiles if lineage or split != "test" or "economy-adjusted" in profiles else (
                    profiles[0], "economy-adjusted")
                jobs.append((split, index, pool, scenario, selected_profiles, protocol, root, source_key, contrasts, lineage))
    tasks, rejected = [], []
    def collect(results):
        for batch, skipped in results:
            tasks.extend(batch)
            rejected.extend(skipped)
    if workers == 1:
        collect(map(prepare_group, jobs))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            collect(pool.map(prepare_group, jobs))
    validate_splits(tasks)
    # Certify preference changes under a new optical brief at the same process.
    comparisons = {}
    for task in tasks:
        comparisons.setdefault((task["cohort"], task["profile"]["name"]), []).append(task)
    for cohort in comparisons.values():
        winners = {t["task_id"]: {c["identity"] for c in t["candidates"] if c["id"] in t["construction"]["best_ids"]}
                   for t in cohort if t["construction"]["yield_target_passes"]}
        for task in cohort:
            key = task["task_id"]
            if key in winners:
                switches = [other for other, best in winners.items() if key != other and winners[key].isdisjoint(best)]
                if switches:
                    task["construction"]["brief_switches"] = switches
                    families = task["construction"]["families"]
                    if "D" not in families:
                        families.append("D")
                        families.sort()
    for task in tasks:
        write_envelope(root/"tasks"/(task["task_id"]+".json"), task, "task")
    parent_digest = None
    if contrasts:
        parent_digest = digest(previous)
        validate_splits([*prior_tasks, *tasks])
        for task in prior_tasks:
            item = task["task_id"]+".json"
            for folder in ("tasks", "replay-data"):
                target = root/folder/item
                key = "task" if folder == "tasks" else "details"
                parent_payload = load_envelope(contrast_parent.parent/folder/item, key)
                if target.exists() and load_envelope(target, key) != parent_payload:
                    raise ValueError(f"contrast target collides with parent: {item}")
                if not target.exists():
                    write_envelope(target, parent_payload, key)
        tasks = [*prior_tasks, *tasks]
    manifest = dict(version=VERSION, optics_version=OPTICS_VERSION, protocol=protocol.model_dump(mode="json"),
                    configuration=dict(scenarios=scenarios, pools=pools, profiles=profiles,
                                       training_contrast_parent_sha256=parent_digest, supplied_lineages=lineages), provenance=source,
                    runtime_seconds=time.perf_counter()-started, attempted_groups=len(jobs),
                    rejection_details=rejected, task_index=[dict(id=t["task_id"], split=t["split"], group=t["group"],
                        capability=t["capability"], reason=t["capability_rejection"], families=t["construction"]["families"],
                        sha256=digest(t)) for t in tasks])
    write_envelope(root/"manifest.json", manifest, "manifest")
    return manifest


def load_tasks(manifest_path, *, split=None, distribution="representative"):
    manifest = load_envelope(manifest_path, "manifest")
    if manifest["version"] != VERSION or distribution not in ("representative", "capability"):
        raise ValueError("unsupported benchmark version or distribution")
    tasks = []
    for record in manifest["task_index"]:
        if (split is not None and record["split"] != split) or (distribution == "capability" and not record["capability"]):
            continue
        task = load_envelope(Path(manifest_path).parent/"tasks"/(record["id"]+".json"), "task")
        if digest(task) != record["sha256"]:
            raise ValueError("task differs from frozen manifest")
        ids = [c["id"] for c in task["candidates"]]
        if sorted(ids) != list(range(1, 9)) or {str(i) for i in ids} != set(task["outcomes"]):
            raise ValueError("task does not contain every action exactly once")
        tasks.append(task)
    validate_splits(tasks)
    return tasks

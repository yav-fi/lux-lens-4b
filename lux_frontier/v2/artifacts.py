"""Versioned replay and Harbor exports from the same frozen optical task."""

from dataclasses import asdict
import gzip
import json
from pathlib import Path
import shlex

import numpy as np

from lux_frontier.optical_engine import BuildErrors, sag
from lux_frontier.v2 import grading
from lux_frontier.v2.benchmark import load_envelope, render, write_json
from lux_frontier.v2.grading import digest, grade, submit
from lux_frontier.v2.optics import Lens, Requirements, build_optic, measurements, trace_field

REPLAY_VERSION = "lux-optical-replay-1"


def ray_record(lens, p, field, wave):
    rays = trace_field(lens, p, field, wave, count=49)

    def coordinates(x, y, z, intensity):
        return [[round(float(a), 8) for a in point] if good and np.isfinite(point).all() else None
                for point, good in zip(zip(x, y, z, strict=True), intensity > 0, strict=True)]

    surfaces = [coordinates(s.x, s.y, s.z, s.intensity) for s in list(lens.surfaces)[1:]]
    return dict(field_deg=field, wavelength_um=wave, surface_nodes_mm=surfaces,
                sensor_intersections_mm=coordinates(rays.x, rays.y, rays.z, rays.i))


def snapshot(p, brief, samples, *, errors=None, focus=0., curves=False, full_clearance=False):
    lens, geometry = build_optic(p, errors, focus)
    surfaces = []
    for index in range(1, len(lens.surfaces)-1):
        s = lens.surfaces[index]
        cs = s.geometry.cs
        rim = float(s.aperture.r_max)
        glass_rim = p.clear_semidiameters_mm[(index-1)//2] if full_clearance else rim
        radius = float(s.geometry.radius)
        surfaces.append(dict(index=index, radius_mm=radius, aperture_radius_mm=rim, is_stop=s.is_stop,
                             vertex_mm=[float(cs.x), float(cs.y), float(cs.z)],
                             tilt_rad=[float(cs.rx), float(cs.ry)],
                             radial_profile_mm=[[float(r), sag(radius, abs(float(r)))]
                                                for r in np.linspace(-glass_rim, glass_rim, 41)]))
    targets = list(dict.fromkeys((t.field_angle_deg, t.wavelength_um) for t in brief.mtf_thresholds))
    analyzed = measurements(p, brief, samples, optic=(lens, geometry), focus=focus, curves=True) if curves else None
    return dict(surface_geometry=surfaces, traces=[ray_record(lens, p, f, w) for f, w in targets],
                sensor_z_mm=float(lens.image_surface.geometry.cs.z), measured_with_curves=analyzed)


def export_replay(task, manifest, output, *, model_records=None, phase="development", representation="physics"):
    details = load_envelope(manifest.parent/"replay-data"/(task["task_id"]+".json"), "details")
    if digest(details) != task.get("replay_details_sha256", digest(details)):
        raise ValueError("replay details differ from frozen task")
    brief = Requirements.model_validate(task["brief"])
    samples = task["protocol"]["pupil_samples"]
    nominal = task["outcomes"]
    candidates = {}
    for c in task["candidates"]:
        key = str(c["id"])
        result = nominal[key]
        original = Lens.model_validate(c["prescription"])
        refined = Lens.model_validate(result["refined_prescription"])
        trajectory = []
        for step in details[key]["trajectory"]:
            trial = Lens.model_validate(step["prescription"]) if step["measured"] else None
            trajectory.append(dict(**step, optical=snapshot(trial, brief, samples) if trial else None))
        builds = details[key]["builds"]
        # Preserve every manufacturing draw and physical outcome. Attach traced
        # geometry for one passing and one measurable failing example in the
        # display phase where such examples exist.
        targets = [next((b for b in builds.get(phase, []) if b["passed"] == flag and b["measurements"]), None)
                   for flag in (True, False)]
        for build in filter(None, targets):
            build["optical"] = snapshot(refined, brief, samples, errors=BuildErrors(**build["errors"]),
                                        focus=build["sensor_shift_mm"], curves=True)
        candidates[key] = dict(candidate=c, result=result, initial=snapshot(original, brief, samples, curves=True),
                               refined=snapshot(refined, brief, samples, curves=True), trajectory=trajectory, builds=builds)

    from lux_frontier.v2.evaluation import response_map
    comparisons = {}
    for label, record in (model_records or {}).items():
        responses = response_map(record, [task], name=label).get(task["task_id"])
        if not responses:
            raise ValueError(f"{label} has no matching task")
        comparisons[label] = [dict(r, model=record["model"], checkpoint=record.get("checkpoint"),
                                   prompt=render(task, r["representation"], format_example=r.get("format_example", False)),
                                   grade=asdict(grade(task, r["response"], clean=r["clean"], phase=phase))) for r in responses]
    if not comparisons:
        closest = task["construction"]["heuristic_id"]
        best = max(task["candidates"], key=lambda c: (grade(task, submit(c["id"]), phase=phase).reward, c["identity"]))["id"]
        comparisons = {name: [dict(response=submit(i), prompt=render(task, representation), model=None,
                                   grade=asdict(grade(task, submit(i), phase=phase)))]
                       for name, i in (("closest_to_spec", closest), ("exhaustive_oracle", best))}
    value = dict(version=REPLAY_VERSION, grading_version=task["version"], task_id=task["task_id"],
                 task_sha256=digest(task), phase=phase, brief=task["brief"], profile=task["profile"], protocol=task["protocol"],
                 candidate_order=[c["id"] for c in task["candidates"]], candidates=candidates, comparisons=comparisons,
                 coordinates="Global XYZ in mm; null means the ray was clipped, invalid, or unavailable at that surface.",
                 analysis="Monochromatic FFT MTF slices and geometric ray intersections from Optiland 0.6.3.")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(gzip.compress(json.dumps(value, separators=(",", ":"), allow_nan=False).encode(), mtime=0))
    return value


def plot_replay(source, output, candidate=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from lux_frontier.optical_engine import rotation_matrix

    replay = json.loads(gzip.decompress(source.read_bytes()))
    if replay["version"] != REPLAY_VERSION:
        raise ValueError("unsupported replay version")
    if candidate is None:
        candidate = next((r["grade"]["starter_id"] for rows in reversed(list(replay["comparisons"].values()))
                          for r in rows if r["grade"]["starter_id"] is not None), replay["candidate_order"][0])
    item = replay["candidates"][str(candidate)]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    for axis, label in zip(axes[0], ("initial", "refined"), strict=True):
        opt = item[label]
        for surface in opt["surface_geometry"]:
            line = np.array([[0, y, z] for y, z in surface["radial_profile_mm"]])
            orient = rotation_matrix(np.rad2deg(surface["tilt_rad"]))
            line = line @ orient.T + surface["vertex_mm"]
            axis.plot(line[:, 2], line[:, 1], color="navy", linewidth=1)
        trace = opt["traces"][-1]
        for ray in zip(*trace["surface_nodes_mm"], strict=True):
            ray = [p for p in ray if p is not None]
            if ray:
                axis.plot([p[2] for p in ray], [p[1] for p in ray], alpha=.35, linewidth=.7)
        axis.set(title=f"{label.capitalize()} · {trace['field_deg'][1]:g}° · {trace['wavelength_um']:g} µm",
                 xlabel="Z (mm)", ylabel="Y (mm)")
    lines, spots = axes[1]
    for label in ("initial", "refined"):
        opt = item[label]
        available = opt["measured_with_curves"]["trace"]["field_mapping"]
        curve = available[-1]["curve"]
        lines.plot(curve["frequency_lp_per_mm"], curve["mtf"], label=label)
        xy = np.array([p[:2] for p in opt["traces"][-1]["sensor_intersections_mm"] if p is not None])
        if xy.size:
            xy = (xy-xy.mean(axis=0))*1000
            spots.scatter(xy[:, 0], xy[:, 1], s=6, alpha=.7, label=label)
    lines.set(title="Off-axis tangential MTF", xlabel="Line pairs/mm", ylabel="MTF", xlim=(0, 120), ylim=(0, 1.02))
    spots.set(title="Off-axis spot around centroid", xlabel="X (µm)", ylabel="Y (µm)", aspect="equal")
    for axis in (lines, spots):
        axis.legend()
    fig.suptitle(f"{replay['task_id']} · candidate {candidate}")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=130)
    plt.close(fig)


def export_harbor(tasks, root, *, phase, representation="physics"):
    if not tasks:
        raise ValueError("no tasks to export")
    if root.exists():
        raise ValueError("choose an unused Harbor output path")
    index = []
    for task in tasks:
        folder = root/task["task_id"]
        for name in ("environment", "tests", "solution"):
            (folder/name).mkdir(parents=True)
        prompt = render(task, representation)
        (folder/"instruction.md").write_text(prompt+"\n\nSave the JSON to `/app/submission.json`.\n")
        (folder/"task.toml").write_text('version = "1.0"\n\n[metadata]\ndomain = "optical-engineering"\n'
                                        '\n[verifier]\ntimeout_sec = 60\n\n[agent]\ntimeout_sec = 300\n'
                                        '\n[environment]\nbuild_timeout_sec = 600\ncpus = 1\nmemory_mb = 1024\nstorage_mb = 1024\n')
        (folder/"environment"/"Dockerfile").write_text("FROM python:3.12-slim\nWORKDIR /app\n")
        (folder/"tests"/"grading.py").write_bytes(Path(grading.__file__).read_bytes())
        # Only hidden files contain rewards. The verifier needs Python's standard library.
        cache = {key: task[key] for key in ("version", "yield_target", "outcomes")}
        cache["outcomes"] = {k: {f: row[f] for f in ("nominal_pass", "near_miss", "simulations")}
                             for k, row in cache["outcomes"].items()}
        for row in cache["outcomes"].values():
            row["simulations"] = {phase: row["simulations"][phase]} if row["nominal_pass"] else {}
        write_json(folder/"tests"/"cache.json", dict(task=cache, sha256=digest(cache), phase=phase))
        test = folder/"tests"/"test.sh"
        test.write_text('#!/bin/sh\nset -eu\ncd "$(dirname "$0")"\n'
                        'mkdir -p "${LUX_VERIFIER_DIR:-/logs/verifier}"\n'
                        'python3 grading.py cache.json "${LUX_SUBMISSION:-/app/submission.json}" '
                        '"${LUX_VERIFIER_DIR:-/logs/verifier}/reward.txt"\n')
        best = max(task["candidates"], key=lambda c: (grade(task, submit(c["id"]), phase=phase).reward, c["identity"]))
        solve = folder/"solution"/"solve.sh"
        solve.write_text("#!/bin/sh\nset -eu\nprintf '%s\\n' "+shlex.quote(submit(best["id"]))+
                         ' > "${LUX_SUBMISSION:-/app/submission.json}"\n')
        for f in (test, solve):
            f.chmod(0o755)
        index.append(dict(task_id=task["task_id"], task_sha256=digest(task), phase=phase,
                          prompt_sha256=digest(prompt), oracle_reward=grade(task, submit(best["id"]), phase=phase).reward,
                          files={str(path.relative_to(folder)): digest(path.read_text())
                                 for path in sorted(folder.glob("**/*")) if path.is_file()}))
    write_json(root/"manifest.json", dict(grading_version=grading.VERSION, tasks=index))
    return index

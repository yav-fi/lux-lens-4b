"""Compact browser assets backed by frozen task and ray-trace records."""

from dataclasses import asdict
import gzip
import json

from lux_frontier.optical_engine import BuildErrors
from lux_frontier.v2 import BROWSER_REPLAY_VERSION
from lux_frontier.v2.artifacts import snapshot
from lux_frontier.v2.benchmark import load_replay_details, render
from lux_frontier.v2.curriculum import difficulty
from lux_frontier.v2.evaluation import response_map
from lux_frontier.v2.grading import digest, grade, submit
from lux_frontier.v2.optics import Lens, Requirements

def accepted_indices(trajectory):
    """Recover incumbent changes from V2's same-incumbent pairs of objective trials."""
    if not trajectory or trajectory[0]["axis"] != "initial" or trajectory[0]["measured"] is None:
        raise ValueError("missing initial optimizer measurement")
    accepted = [0]
    for start in range(1, len(trajectory), 2):
        candidates = [accepted[-1], *range(start, min(start+2, len(trajectory)))]
        best = min((i for i in candidates if trajectory[i]["measured"] is not None),
                   key=lambda i: (bool(trajectory[i]["failures"]), trajectory[i]["score"]))
        if best != accepted[-1]:
            accepted.append(best)
    return accepted


def slim_measure(measure):
    if measure is None:
        return None
    measured = {k: v for k, v in measure.items() if k != "trace"}
    curves = []
    for row in measure["trace"]["field_mapping"]:
        if "curve" not in row:
            continue
        curve = row["curve"]
        points = [[round(x, 4), round(y, 6)] for x, y in zip(curve["frequency_lp_per_mm"], curve["mtf"], strict=True)
                  if x <= 160]
        curves.append(dict(field_deg=row["field_deg"], wavelength_um=row["wavelength_um"],
                           orientation=curve["orientation"], points=points))
    if curves:
        measured["curves"] = curves
    return measured


def export_browser(task, manifest, output, *, phase, records=None):
    details = load_replay_details(manifest, task)
    brief = Requirements.model_validate(task["brief"])
    samples = task["protocol"]["pupil_samples"]
    states, candidates, prompts = {}, {}, {}

    def optical(p, errors=None, focus=0.):
        key = digest([p, errors, focus])[:24]
        if key not in states:
            optic = snapshot(Lens.model_validate(p), brief, samples, errors=BuildErrors(**errors) if errors else None,
                             focus=focus, curves=errors is None, full_clearance=True)
            optic["measurements"] = slim_measure(optic.pop("measured_with_curves"))
            if errors is None:
                optic["prescription"] = p
            states[key] = optic
        return key

    for item in task["candidates"]:
        key = str(item["id"])
        outcome, detail = task["outcomes"][key], details[key]
        trajectory = detail["trajectory"]
        accepted = accepted_indices(trajectory)
        if trajectory[accepted[-1]]["prescription"] != outcome["refined_prescription"]:
            raise ValueError("accepted trajectory does not end at the frozen refined prescription")
        timeline = [dict(trial=i, state=optical(trajectory[i]["prescription"]), objective=trajectory[i]["score"],
                         axis=trajectory[i]["axis"], measurements=slim_measure(trajectory[i]["measured"]),
                         failures=trajectory[i]["failures"]) for i in accepted]
        rejected = [dict(trial=i, axis=step["axis"], objective=step["score"], failures=step["failures"],
                         valid=step["measured"] is not None) for i, step in enumerate(trajectory) if i not in accepted]
        builds = []
        for build in detail["builds"].get(phase, []):
            state = (optical(outcome["refined_prescription"], build["errors"], build["sensor_shift_mm"])
                     if build["measurements"] else None)
            builds.append(dict(**{k: v for k, v in build.items() if k not in ("optical", "measurements")}, state=state,
                               measurements=slim_measure(build["measurements"])))
        candidates[key] = dict(id=item["id"], identity=item["identity"], ancestry=item["prescription"]["ancestry"],
                               timeline=timeline, rejected=rejected, builds=builds,
                               result=asdict(grade(task, submit(item["id"]), phase=phase)),
                               simulation=outcome["simulations"].get(phase))
    comparisons = {}
    for label, record in (records or {}).items():
        responses = response_map(record, [task], name=label).get(task["task_id"])
        if not responses:
            continue
        payloads = []
        for r in responses:
            prompt = render(task, r["representation"], format_example=r.get("format_example", False))
            prompt_ref = digest(prompt)
            prompts[prompt_ref] = prompt
            payloads.append(dict(response=r["response"], prompt_ref=prompt_ref, model=record["model"],
                                 checkpoint=record.get("checkpoint"), source=record.get("execution_backend", "unrecorded"),
                                 grade=asdict(grade(task, r["response"], clean=r["clean"], phase=phase))))
        comparisons[label] = payloads
    if not comparisons:
        keys = ("closest_to_spec", "exhaustive_oracle")
        ids = [task["construction"]["heuristic_id"], max(candidates.values(), key=lambda c: (
            c["result"]["reward"], c["identity"]))["id"]]
        comparisons = {label: [dict(response=submit(i), source="offline_control", grade=candidates[str(i)]["result"])]
                       for label, i in zip(keys, ids, strict=True)}
    payload = dict(version=BROWSER_REPLAY_VERSION, task_id=task["task_id"], task_sha256=digest(task),
                   replay_details_sha256=digest(details), optics_version=task["optics_version"],
                   simulator="Optiland 0.6.3", phase=phase, split=task["split"], families=task["construction"]["families"],
                   difficulty=difficulty(task), brief=task["brief"], profile=task["profile"], protocol=task["protocol"],
                   candidate_order=[c["id"] for c in task["candidates"]], candidates=candidates, states=states,
                   comparisons=comparisons, prompts=prompts)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(gzip.compress(json.dumps(payload, separators=(",", ":"), allow_nan=False).encode(), mtime=0))
    return payload

"""Independent, paired manufacturing streams and yield uncertainty."""

from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from dataclasses import fields
import math

from lux_frontier.optical_grading import OpticalCache
from lux_frontier.v2.benchmark import load_envelope, load_replay_details, load_tasks, provenance, write_envelope
from lux_frontier.v2.grading import PHASES, digest
from lux_frontier.v2.optics import Lens, Profile, Requirements, simulate_build

MINIMUM_BUILDS = dict(screen=8, investigation=32, definitive=100)
STREAM_VERSION = "lux-v21-factory-v1"


def wilson(passed, tested):
    if not 0 <= passed <= tested or not tested:
        raise ValueError("invalid manufacturing counts")
    z = 1.959963984540054
    point = passed/tested
    denominator = 1+z*z/tested
    center = (point+z*z/(2*tested))/denominator
    radius = z*math.sqrt(point*(1-point)/tested+z*z/(4*tested*tested))/denominator
    return [max(0., center-radius), min(1., center+radius)]


def yield_record(builds, target, fidelity):
    passed, tested = sum(b["passed"] for b in builds), len(builds)
    interval = wilson(passed, tested)
    return dict(passed=passed, tested=tested, yield_fraction=passed/tested, interval95=interval,
                threshold_status=("above" if interval[0] >= target else "below" if interval[1] < target else "unresolved"),
                fidelity=fidelity, evidence="confirmed_simulated" if fidelity == "definitive" else "provisional_simulated",
                stream_prefix=builds[0]["stream"][:-1])


def paired_yield(first, second):
    mismatches = [0, 0]
    for left, right in zip(first, second, strict=True):
        if left["stream"] != right["stream"]:
            raise ValueError("paired yield comparison requires identical draw indices")
        if left["passed"] != right["passed"]:
            mismatches[int(right["passed"])] += 1
    only_first, only_second = mismatches
    discordant = sum(mismatches)
    p = 2*sum(math.comb(discordant, i)/2**discordant for i in range(min(mismatches)+1))
    return dict(only_first=only_first, only_second=only_second, builds=len(first),
                difference_second_minus_first=(only_second-only_first)/len(first), exact_mcnemar_p=min(1., p))


def physical_brief(brief):
    return {k: v for k, v in brief.items() if k not in ("brief_id", "manufacturing_tolerance_profile_ref")}


def namespace(seed, physical_group, phase, fidelity):
    # Same physical error channels and draw index across designs and process counterfactuals.
    value = digest([STREAM_VERSION, seed, physical_group, phase, fidelity])
    return [0x4C580201, *[int(value[i:i+8], 16) for i in range(0, 32, 8)]]


def cached_builds(cache, task, outcome, prefix, count, source_key):
    prescription = Lens.model_validate(outcome["refined_prescription"])
    brief = Requirements.model_validate(task["brief"])
    profile = Profile(**{f.name: task["profile"][f.name] for f in fields(Profile)})
    samples = task["protocol"]["pupil_samples"]
    key = dict(source=source_key, prescription=prescription, brief=physical_brief(task["brief"]),
               profile=profile.record(), pupil_samples=samples, stream=prefix, count=count)
    return cache.compute("paired-builds-v21", key, lambda: [
        simulate_build(prescription, brief, samples, profile, [*prefix, i]) for i in range(count)
    ])[0]


def grade_manufacturing(job):
    task, source, root, phase, fidelity, count, seed, source_key = job
    result = deepcopy(task)
    details = load_replay_details(source, task)
    cache = OpticalCache(root/"cache")
    prefix = namespace(seed, task["group"], phase, fidelity)
    for local_id, outcome in result["outcomes"].items():
        if not outcome["nominal_pass"]:
            continue
        builds = cached_builds(cache, task, outcome, prefix, count, source_key)
        metrics = yield_record(builds, task["yield_target"], fidelity)
        outcome.setdefault("manufacturing_fidelities", {}).setdefault(phase, {})[fidelity] = metrics
        outcome["simulations"][phase] = metrics
        details[local_id]["builds"][phase] = builds
    result["replay_details_sha256"] = digest(details)
    return result, details


def extend(source, root, *, phase, fidelity, seed, count=None, task_ids=(), workers=1):
    if (root/"manifest.json").exists():
        raise ValueError("output manifest already exists")
    if phase not in PHASES or fidelity not in MINIMUM_BUILDS or workers < 1 or seed < 0:
        raise ValueError("unknown simulation phase or fidelity, or invalid seed/worker count")
    count = MINIMUM_BUILDS[fidelity] if count is None else count
    if count < MINIMUM_BUILDS[fidelity]:
        raise ValueError("insufficient builds for requested fidelity")
    split = {"construction": "train", "training": "train", "development": "development", "final": "test"}[phase]
    if phase == "construction":
        raise ValueError("use construction simulations at task generation to keep certification and outcomes aligned")
    if phase == "final" and fidelity != "definitive":
        raise ValueError("final confirmation requires at least 100 fresh builds")
    manifest = load_envelope(source, "manifest")
    tasks = load_tasks(source)
    chosen = [t for t in tasks if t["split"] == split and (not task_ids or t["task_id"] in task_ids)]
    if not chosen or task_ids and set(task_ids) != {t["task_id"] for t in chosen}:
        raise ValueError("requested tasks must belong to the selected draw purpose")
    source_key = digest(provenance()["source_hashes"])
    jobs = [(task, source, root, phase, fidelity, count, seed, source_key) for task in chosen]
    if workers == 1:
        processed = list(map(grade_manufacturing, jobs))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            processed = list(pool.map(grade_manufacturing, jobs))
    revised = {t["task_id"]: (t, details) for t, details in processed}
    publish_tasks(root, source, manifest, tasks, revised, manufacturing=dict(
        parent_sha256=digest(manifest), phase=phase, fidelity=fidelity, builds=count, seed=seed,
        source_key=source_key, task_ids=sorted(revised)))
    return {t["task_id"]: {key: outcome["simulations"].get(phase) for key, outcome in t["outcomes"].items()}
            for t, _ in processed}


def publish_tasks(root, source, manifest, tasks, revised, **metadata):
    """Write a new manifest after every task and replay payload is durable."""
    index = []
    for task in tasks:
        name = task["task_id"]
        filename = name+".json"
        if name in revised:
            task, details = revised[name]
        else:
            details = load_replay_details(source, task)
        if "replay_details_sha256" in task and digest(details) != task["replay_details_sha256"]:
            raise ValueError(f"replay details mismatch for {name}")
        write_envelope(root/"tasks"/filename, task, "task")
        write_envelope(root/"replay-data"/filename, details, "details")
        index.append(dict(id=name, split=task["split"], group=task["group"], capability=task["capability"],
                          reason=task["capability_rejection"], families=task["construction"]["families"], sha256=digest(task)))
    write_envelope(root/"manifest.json", dict(manifest, **metadata) | {"task_index": index}, "manifest")

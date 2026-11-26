"""Assemble training situations from physically compatible frozen candidate outcomes."""

from collections import Counter, defaultdict
from copy import deepcopy
from itertools import combinations
import math
import random
import statistics

from lux_frontier.optical_grading import OpticalCache
from lux_frontier.v2.benchmark import characterize, load_envelope, load_replay_details, load_tasks, provenance, validate_splits
from lux_frontier.v2.curriculum import difficulty, optical_baselines
from lux_frontier.v2.evaluation import audit
from lux_frontier.v2.grading import digest, grade, submit
from lux_frontier.v2.manufacturing import cached_builds, namespace, physical_brief, publish_tasks, yield_record


def context(task):
    return digest([physical_brief(task["brief"]), task["profile"], task["protocol"], task["optics_version"]])


def signature(task):
    return digest([context(task), sorted(c["identity"] for c in task["candidates"])])


def report(tasks):
    result = audit(tasks)
    for split, row in result["splits"].items():
        selected = [t for t in tasks if t["split"] == split]
        identities = {c["identity"] for t in selected for c in t["candidates"]}
        lineages = {c["prescription"]["ancestry"] for t in selected for c in t["candidates"]}
        sites = {signature(t) for t in selected}
        phase = {"train": "training", "development": "development", "test": "final"}[split]
        rows = [dict(task_id=t["task_id"], situation=signature(t), difficulty=difficulty(t), **optical_baselines(t, phase))
                for t in selected]
        winners_by_position, winners_by_id = Counter(), Counter()
        opportunities_by_position, opportunities_by_id = Counter(), Counter()
        for task in selected:
            for pos, c in enumerate(task["candidates"], 1):
                opportunities_by_position[pos] += 1
                opportunities_by_id[c["id"]] += 1
                if grade(task, submit(c["id"]), phase=phase).final_pass:
                    winners_by_position[pos] += 1
                    winners_by_id[c["id"]] += 1
        row.update(distinct_situations=len(sites), presentation_duplicates=len(selected)-len(sites),
                   distinct_physics_contexts=len({context(t) for t in selected}),
                   recombined_candidate_sets=sum("assembly" in t for t in selected),
                   prescriptions=len(identities), ancestry=sorted(lineages),
                   difficulty=dict(Counter(r["difficulty"] for r in rows)),
                   mean_exact_random_pass=statistics.mean(r["random_pass"] for r in rows),
                   position_pass_rate={i: winners_by_position[i]/v for i, v in sorted(opportunities_by_position.items())},
                   candidate_id_pass_rate={i: winners_by_id[i]/v for i, v in sorted(opportunities_by_id.items())},
                   per_task=rows)
    return result


def subsets(ids, seed, limit):
    """Sample deterministic sets without enumerating large binomial spaces."""
    rng = random.Random(digest([seed, ids]))
    size = math.comb(len(ids), 8)
    if size <= 2*limit:
        options = list(combinations(ids, 8))
        rng.shuffle(options)
        yield from options
    else:
        seen = set()
        for _ in range(16*limit):
            option = tuple(sorted(rng.sample(ids, 8)))
            if option not in seen:
                seen.add(option)
                yield option


def expand(source, root, *, limit, seed=17292):
    if limit < 1 or seed < 0 or (root/"manifest.json").exists():
        raise ValueError("choose a positive expansion limit, nonnegative seed and new output manifest")
    manifest = load_envelope(source, "manifest")
    tasks = load_tasks(source)
    contexts = defaultdict(list)
    for task in tasks:
        if task["split"] == "train":
            contexts[context(task)].append(task)
    cache = OpticalCache(root/"cache")
    source_key = digest(provenance()["source_hashes"])
    revised = {}
    for key, parents in sorted(contexts.items()):
        lookup = {}
        for parent in parents:
            details = load_replay_details(source, parent)
            for candidate in parent["candidates"]:
                local = str(candidate["id"])
                lookup.setdefault(candidate["identity"], (parent, candidate, parent["outcomes"][local], details[local]))
        if len(lookup) <= 8:
            continue
        phase_group = digest([physical_brief(parents[0]["brief"]), sorted(lookup)])
        original_sets = {frozenset(c["identity"] for c in p["candidates"]) for p in parents}
        reused = {}
        made = 0
        for choice in subsets(sorted(lookup), seed, limit+len(original_sets)):
            if frozenset(choice) in original_sets:
                continue
            count = parents[0]["protocol"]["builds_per_phase"]
            ids = list(range(1, 9))
            name = "train-assembly-"+digest([key, choice])[:24]
            random.Random(digest([seed, name, "ids"])).shuffle(ids)
            candidates, outcomes, replay = [], {}, {}
            for identity, local in zip(choice, ids, strict=True):
                if identity not in reused:
                    parent, candidate, source_outcome, source_replay = lookup[identity]
                    outcome = deepcopy(source_outcome)
                    outcome["simulations"] = {}
                    outcome.pop("manufacturing_fidelities", None)
                    item = dict(trajectory=source_replay["trajectory"], builds={})
                    if outcome["nominal_pass"]:
                        for phase in ("construction", "training"):
                            prefix = namespace(seed, phase_group, phase, "screen")
                            builds = cached_builds(cache, parent, outcome, prefix, count, source_key)
                            item["builds"][phase] = builds
                            outcome["simulations"][phase] = yield_record(builds, parent["yield_target"], "screen")
                    reused[identity] = (candidate, outcome, item)
                candidate, outcome, item = reused[identity]
                candidates.append(dict(candidate, id=local))
                outcomes[str(local)] = outcome
                replay[str(local)] = item
            random.Random(digest([seed, name, "order"])).shuffle(candidates)
            parent = parents[0]
            task = dict(parent, task_id=name, group="assembly-"+phase_group[:24],
                        cohort="assembly-"+digest(sorted(lookup))[:24], candidates=candidates, outcomes=outcomes,
                        brief=dict(parent["brief"], brief_id=name), replay_details_sha256=digest(replay),
                        assembly=dict(context=key, parents=sorted({lookup[i][0]["task_id"] for i in choice}),
                                      statistical_unit="shared_prescription_ancestry"))
            task["construction"] = characterize(task)
            task["capability"] = "E" in task["construction"]["families"]
            if not task["capability"]:
                continue
            task["capability_rejection"] = None
            revised[name] = task, replay
            made += 1
            if made == limit:
                break
    # For the same eight designs, retain a counterfactual only when the preferred sets differ.
    siblings = defaultdict(list)
    for task, _ in revised.values():
        siblings[tuple(sorted(c["identity"] for c in task["candidates"]))].append(task)
    for group in siblings.values():
        best = {t["task_id"]: {c["identity"] for c in t["candidates"] if c["id"] in t["construction"]["best_ids"]}
                for t in group}
        for task in group:
            name = task["task_id"]
            differences = [t["task_id"] for t in group if best[name].isdisjoint(best[t["task_id"]])]
            if differences:
                task["construction"]["brief_switches"] = differences
                task["construction"]["families"].append("D")
                task["construction"]["families"].sort()
    assembled = [t for t, _ in revised.values()]
    if not assembled:
        raise ValueError("no compatible, informative candidate sets found; generate a second pool at the same settings")
    validate_splits([*tasks, *assembled])
    publish_tasks(root, source, manifest, [*tasks, *assembled], revised,
                  assembly=dict(parent_sha256=digest(manifest), source_key=source_key, seed=seed, max_per_context=limit))
    return dict(parent_tasks=len(tasks), assembled=len(assembled), compatible_contexts=sum(len(p)>1 for p in contexts.values()))

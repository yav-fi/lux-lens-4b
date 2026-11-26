"""Read the approved V2.2 exploration journal without provider calls."""

import argparse
from collections import Counter, defaultdict
from itertools import product
import json
from pathlib import Path
import statistics

from lux_frontier.v2.benchmark import FAMILIES, REPRESENTATIONS, load_tasks, write_envelope
from lux_frontier.v2.grading import digest, load_envelope

TEMPERATURES = (.6, 1.25)
STRATEGIES = ("batched", "separate")
ROTATIONS = (0, 4)
MEANINGFUL_SPREAD = .10


def lane(group):
    return group["representation"], group["sampling"], group["temperature"]


def group_stats(group):
    grades = [c["grade"] for c in group["completions"]]
    valid = [g for g in grades if g["error"] is None]
    rewards = [g["reward"] for g in valid]
    spread = max(rewards)-min(rewards) if rewards else 0.
    return dict(pass_rate=statistics.mean(g["final_pass"] for g in grades),
                invalid_fraction=1-len(valid)/len(grades),
                reward=statistics.mean(g["reward"] for g in grades),
                valid_reward_variance=statistics.pvariance(rewards) if rewards else 0.,
                valid_reward_range=spread, useful_group=spread>=MEANINGFUL_SPREAD,
                target_outcome_spread=len({g["final_pass"] for g in valid})>1,
                informative_group=group["diagnostics"]["optically_informative"],
                unique_valid=group["diagnostics"]["unique_valid_actions"],
                entropy_bits=group["diagnostics"]["valid_action_entropy_bits"])


def summarize(groups):
    summaries = [group_stats(g) for g in groups]
    if not summaries:
        return dict(groups=0, completions=0, metrics=None)
    return dict(groups=len(groups), completions=sum(len(g["completions"]) for g in groups),
                metrics={k: statistics.mean(s[k] for s in summaries) for k in summaries[0]},
                baselines={k: statistics.mean(g["baselines"][k] for g in groups) for k in
                           ("random_pass", "closest_pass", "physics_pass", "oracle_pass", "random_reward")},
                meaningful_groups=sum(s["useful_group"] for s in summaries),
                informative_groups=sum(s["informative_group"] for s in summaries),
                groups_by_unique_valid=dict(Counter(s["unique_valid"] for s in summaries)),
                failures=dict(Counter(f for g in groups for f in (
                    c["grade"]["error"] for c in g["completions"]) if f)))


def rotation_metrics(groups):
    pairs = defaultdict(dict)
    counts = Counter()
    ids = Counter()
    for group in groups:
        pairs[group["task_id"], lane(group)][group["position_offset"]] = group
        counts.update(group["diagnostics"]["position_counts"])
        ids.update(group["diagnostics"]["action_counts"])
    matched = stable_id = stable_position = 0
    for pair in pairs.values():
        if set(pair) != set(ROTATIONS):
            continue
        before, after = (pair[o] for o in ROTATIONS)
        for a, b in zip(before["completions"], after["completions"], strict=True):
            x, y = (row["grade"]["starter_id"] for row in (a,b))
            if x is None or y is None:
                continue
            matched += 1
            stable_id += x==y
            stable_position += before["candidate_order"].index(x)==after["candidate_order"].index(y)
    return dict(valid_pairs=matched, same_id=stable_id, same_position=stable_position,
                positions=dict(sorted(counts.items())), candidate_ids=dict(sorted(ids.items())),
                per_rotation={o: summarize([g for g in groups if g["position_offset"]==o]) for o in ROTATIONS})


def check_lane(groups, feasible_ids):
    panel = summarize(groups)
    rotation = rotation_metrics(groups)
    per_offset = rotation["per_rotation"]
    order_delta = abs(per_offset[0]["metrics"]["pass_rate"]-per_offset[4]["metrics"]["pass_rate"])
    useful = {g["task_id"] for g in groups if group_stats(g)["useful_group"]}
    checks = dict(both_feasible_tasks_have_meaningful_group=useful==feasible_ids,
                  invalid_fraction_at_most_ten_percent=panel["metrics"]["invalid_fraction"]<=.10,
                  pooled_order_delta_at_most_25_points=order_delta<=.25,
                  feasible_pass_at_least_exact_chance=panel["metrics"]["pass_rate"]>=panel["baselines"]["random_pass"])
    return dict(**panel, rotation=rotation, order_delta=order_delta, checks=checks, smoke_gate=all(checks.values()),
                mean_prompt_tokens=statistics.mean(g["prompt_tokens"] for g in groups))


def report(run, journal, tasks):
    grouped = {(g["task_id"], *lane(g), g["position_offset"]): g for g in journal}
    task_ids = {g["task_id"] for g in journal}
    expected = set(product(task_ids, REPRESENTATIONS, STRATEGIES, TEMPERATURES, ROTATIONS))
    chosen = {t["task_id"]: t for t in tasks if t["task_id"] in task_ids}
    if (run["status"]!="completed" or run["execution_backend"]!="tinker" or run["mode"]!="exploration"
            or len(task_ids)!=3 or set(chosen)!=task_ids or expected!=set(grouped) or len(journal)!=72
            or any(len(g["completions"])!=8 for g in journal) or run["sampled_completions"]!=576
            or any(g["task_sha256"]!=digest(chosen[g["task_id"]]) for g in journal)):
        raise ValueError("expected the complete authorized 72-group Tinker matrix on the frozen development tasks")
    feasible_ids = {g["task_id"] for g in journal if g["baselines"]["oracle_pass"]}
    if len(feasible_ids)!=2:
        raise ValueError("the authorized matrix requires two feasible briefs and a no-solution control")
    feasible = [g for g in journal if g["task_id"] in feasible_ids]
    lanes = {}
    for variant in product(REPRESENTATIONS, STRATEGIES, TEMPERATURES):
        key = f"{variant[0]}/{variant[1]}/{variant[2]:g}"
        lanes[key] = check_lane([g for g in feasible if lane(g)==variant], feasible_ids)
    passing = [key for key, row in lanes.items() if row["smoke_gate"]]
    # Select on group productivity before reward. The lane comparison is exploratory.
    selection = max(passing, key=lambda k: (lanes[k]["meaningful_groups"],
                     -lanes[k]["metrics"]["invalid_fraction"], lanes[k]["metrics"]["pass_rate"],
                     -lanes[k]["order_delta"], -lanes[k]["mean_prompt_tokens"]), default=None)
    return dict(model=run["model"], renderer=run["renderer"], run_sha256=digest(run),
                estimated_token_spend_usd=run["estimated_token_spend_usd"], usage=run["usage"], status=run["status"],
                overall=summarize(journal), feasible=summarize(feasible),
                impossible=summarize([g for g in journal if g["task_id"] not in feasible_ids]),
                by_task={t: summarize([g for g in journal if g["task_id"]==t]) for t in sorted(task_ids)},
                by_family={f: summarize([g for g in journal if f in chosen[g["task_id"]]["construction"]["families"]])
                           for f in FAMILIES},
                rotation=rotation_metrics(feasible), lanes=lanes, selected_lane=selection,
                meaningful_optical_reward_range=MEANINGFUL_SPREAD)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run = load_envelope(args.run, "run")
    path = args.run.with_name(run["group_journal"])
    journal = [json.loads(line) for line in path.read_text().splitlines()]
    result = report(run, journal, load_tasks(args.manifest, split="development"))
    write_envelope(args.output, result, "exploration")
    print(json.dumps({k: result[k] for k in ("overall", "feasible", "impossible", "by_family", "selected_lane")}, indent=2))


if __name__ == "__main__":
    main()

"""Mixed training allocation from rolling, all-attempt optical measurements."""

from collections import Counter
import math
import random
import statistics

from pydantic import Field, model_validator

from lux_frontier.contracts import Contract
from lux_frontier.v2.benchmark import FAMILIES
from lux_frontier.v2.grading import digest, grade, submit


class Curriculum(Contract):
    weights: dict[str, float] = Field(default_factory=lambda: dict(A=.15, B=.15, C=.20, D=.10, E=.40))
    target: float = Field(default=.30, ge=.20, le=.80)
    window: int = Field(default=64, ge=12)
    minimum_observations: int = Field(default=8, ge=4)
    adaptive: bool = True

    @model_validator(mode="after")
    def coherent(self):
        if set(self.weights) != set(FAMILIES) or any(not math.isfinite(v) or v <= 0 for v in self.weights.values()):
            raise ValueError("the mixture needs positive weights for all five families")
        if self.minimum_observations > self.window:
            raise ValueError("the observation threshold must fit inside the window")
        return self


def construction_chance(task):
    count = len(task["candidates"])
    successes = task["construction"].get("yield_target_passes", 1)
    return successes / count


def difficulty(task):
    """Use construction success prevalence as a cold-start difficulty proxy."""
    chance = construction_chance(task)
    return "hard" if chance <= .125 else "moderate" if chance < .5 else "easy"


def optical_baselines(task, phase):
    scores = {c["id"]: grade(task, submit(c["id"]), phase=phase) for c in task["candidates"]}
    counts = sum(g.final_pass for g in scores.values())
    construction = task["construction"]
    return dict(random_pass=counts/len(scores), random_reward=statistics.mean(g.reward for g in scores.values()),
                oracle_pass=bool(counts), closest_pass=(
                    scores[construction["heuristic_id"]].final_pass if "heuristic_id" in construction else None),
                physics_pass=(scores[construction["physics_id"]].final_pass if "physics_id" in construction else None))


def summarize(rows):
    if not rows:
        return dict(groups=0)
    fields = ("pass_rate", "mean_reward", "mean_training_reward", "reward_variance", "unique_valid_actions",
              "valid_action_entropy_bits", "optically_informative", "informative", "invalid_fraction")
    values = {k: statistics.mean(float(row[k]) for row in rows) for k in fields}
    if "accepted" in rows[0]:
        values["accepted_fraction"] = statistics.mean(r["accepted"] for r in rows)
    return dict(groups=len(rows), **values)


def observation(group):
    stats = group["diagnostics"]
    grades = [c["grade"] for c in group["completions"]]
    return dict(family=group.get("curriculum_family"), difficulty=group["difficulty"],
                pass_rate=statistics.mean(g["final_pass"] for g in grades),
                mean_reward=statistics.mean(g["reward"] for g in grades),
                mean_training_reward=statistics.mean(stats["training_rewards"]),
                reward_variance=stats["reward_variance"], unique_valid_actions=stats["unique_valid_actions"],
                valid_action_entropy_bits=stats["valid_action_entropy_bits"], informative=stats["informative"],
                optically_informative=stats["optically_informative"], invalid_fraction=stats["invalid"]/len(grades),
                failure_counts=stats["failure_counts"])


def allocation(tasks, config, history):
    rows = history[-config.window:]
    available = {f: [t for t in tasks if t["capability"] and f in t["construction"]["families"]]
                 for f in FAMILIES}
    family = {f: summarize([r for r in rows if r["family"] == f]) for f in FAMILIES}
    tiers = ("hard", "moderate", "easy")
    by_tier = {f: {d: summarize([r for r in rows if (r["family"], r["difficulty"]) == (f, d)])
                   for d in tiers} for f in FAMILIES}
    weights = {f: config.weights[f] for f in FAMILIES if available[f]}
    tier_weights = {f: dict.fromkeys(tiers, 1.) for f in weights}
    for family_name, tiers_for_family in tier_weights.items():
        for tier in tiers:
            chances = [construction_chance(task) for task in available[family_name] if difficulty(task) == tier]
            if chances:
                tiers_for_family[tier] = max(.2, 1-2*abs(statistics.mean(chances)-config.target))
    enough = lambda summary: summary["groups"] >= config.minimum_observations
    if config.adaptive:
        struggling = enough(family["E"]) and family["E"]["pass_rate"] < .20
        for f in weights:
            data = family[f]
            if enough(data) and data["pass_rate"] > .80:
                weights[f] *= .65
            if struggling:
                weights[f] *= .85 if f == "E" else 1.15
            for tier in tiers:
                observed = by_tier[f][tier]
                if enough(observed):
                    tier_weights[f][tier] = max(.55, 1.20-abs(observed["pass_rate"]-config.target)
                                                + .20*observed["optically_informative"])
                if enough(data) and data["pass_rate"] < .20:
                    tier_weights[f][tier] *= {"hard": .85, "moderate": 1., "easy": 1.25}[tier]
    if not weights:
        raise ValueError("no eligible tasks in the mixed curriculum")
    total = sum(weights.values())
    return available, dict(weights={f: w/total for f, w in weights.items()}, tier_weights=tier_weights,
                           rolling=summarize(rows), by_family=family, by_family_and_tier=by_tier)


def schedule(tasks, config, history, *, seed, step, attempts):
    available, snapshot = allocation(tasks, config, history)
    weights = snapshot["weights"]
    rng = random.Random(digest([seed, step, "mixed-task-schedule"]))
    selected, seen = [], set()
    for _ in range(attempts):
        pool = {f: [t for t in available[f] if t["task_id"] not in seen] for f in weights}
        families = [f for f in weights if pool[f]]
        if not families:
            seen.clear()
            pool = available
            families = list(weights)
        family = rng.choices(families, [weights[f] for f in families])[0]
        by_tier = {d: [t for t in pool[family] if difficulty(t) == d] for d in snapshot["tier_weights"][family]}
        tiers = [d for d, items in by_tier.items() if items]
        tier = rng.choices(tiers, [snapshot["tier_weights"][family][d] for d in tiers])[0]
        task = rng.choice(by_tier[tier])
        seen.add(task["task_id"])
        selected.append((task, family))
    return selected, snapshot


def training_summary(rows):
    families = sorted({r["family"] for r in rows if r["family"]})
    failure_counts = Counter()
    for row in rows:
        failure_counts.update(row.get("failure_counts", {}))
    return dict(overall=summarize(rows), by_family={f: summarize([r for r in rows if r["family"] == f]) for f in families},
                by_difficulty={d: summarize([r for r in rows if r["difficulty"] == d])
                               for d in sorted({r["difficulty"] for r in rows})},
                exposure=dict(Counter(r["family"] for r in rows)), failure_counts=dict(failure_counts))

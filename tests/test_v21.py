from dataclasses import replace

import pytest

from lux_frontier.grpo import Completion
from lux_frontier.v2.browser_replay import accepted_indices
from lux_frontier.v2.curriculum import Curriculum, allocation, schedule
from lux_frontier.v2.experiment import clusters, signed_randomization
from lux_frontier.v2.grading import PHASES, VERSION, submit
from lux_frontier.v2.manufacturing import namespace, paired_yield, yield_record
from lux_frontier.v2.rl import Settings, assess_group, random_rewards


def small_task(tag="A", *, ancestry=("glass-a",)):
    return dict(task_id=tag, group="group-"+tag, cohort="cohort-"+tag, capability=True, version=VERSION,
                yield_target=.625, construction=dict(families=[tag], yield_target_passes=2),
                candidates=[dict(id=i, prescription=dict(ancestry=ancestry[(i-1) % len(ancestry)])) for i in range(1, 9)],
                outcomes={str(i): dict(nominal_pass=True, near_miss=.02,
                                      simulations={p: dict(passed=8 if i == 1 else 1, tested=8) for p in PHASES})
                          for i in range(1, 9)})


def completion(candidate):
    text = submit(candidate)
    return Completion(text, text, [1, 2], [-.5, -.5], "stop", True)


def test_random_reward_independent_of_response_and_true_outcome():
    task = small_task()
    first = [completion(1)]*8
    other = [completion(2)]*8
    _, adv_first, stats_first = assess_group(task, first, "training", "random", reward_seed=42)
    _, adv_second, stats_second = assess_group(task, other, "training", "random", reward_seed=42)
    assert stats_first["training_rewards"] == stats_second["training_rewards"] == random_rewards(42, 8)
    assert adv_first == adv_second and any(adv_first)
    assert not stats_first["optically_informative"] and stats_first["informative"]
    _, _, mixed_stats = assess_group(task, [replace(first[0], clean=False), *first[1:]], "training", "random", reward_seed=42)
    assert mixed_stats["training_rewards"] == stats_first["training_rewards"]
    _, _, nominal = assess_group(task, [first[0], other[0]], "training", "nominal")
    assert not nominal["informative"] and nominal["optically_informative"]


def test_mixture_keeps_complete_task_and_uses_all_attempts():
    settings = Curriculum(minimum_observations=4)
    tasks = [small_task(f) for f in "ABCDE"]
    initial, before = schedule(tasks, settings, [], seed=17, step=0, attempts=15)
    assert {family for _, family in initial} == set("ABCDE")
    rate = {key: 0. for key in ("mean_reward", "mean_training_reward", "reward_variance", "unique_valid_actions",
                               "valid_action_entropy_bits", "optically_informative", "informative", "invalid_fraction")}
    rows = [dict(rate, family="E", difficulty="moderate", pass_rate=0., accepted=False) for _ in range(4)]
    rows += [dict(rate, family="A", difficulty="moderate", pass_rate=1., accepted=False) for _ in range(4)]
    _, after = allocation(tasks, settings, rows)
    assert after["weights"]["E"] < before["weights"]["E"]
    assert 0 < after["weights"]["A"] < before["weights"]["A"]
    assert after["rolling"]["groups"] == 8
    assert after["rolling"]["accepted_fraction"] == 0
    with pytest.raises(ValueError, match="fixed mixture"):
        Settings(matched_attempts=True)


def test_optimizer_shows_only_actual_incumbents():
    def trial(axis, score, passing=False, measured=True):
        return dict(axis=axis, score=score, measured={} if measured else None, failures=[] if passing else ["failure"])

    trials = [trial("initial", 1), trial(0, 0.9), trial(0, 0.8), trial(2, 0.7, measured=False),
              trial(2, 1.5, passing=True), trial(3, 0.5), trial(3, 1.5, passing=True)]
    assert accepted_indices(trials) == [0, 2, 4]


def test_manufacturing_uncertainty_and_independent_purposes():
    prefixes = {tuple(namespace(19261, "paired", phase, fidelity))
                for phase in PHASES for fidelity in ("screen", "investigation", "definitive")}
    assert len(prefixes) == 12
    prefix = namespace(19261, "paired", "final", "definitive")
    first = [dict(stream=[*prefix, i], passed=i < 62) for i in range(100)]
    second = [dict(stream=[*prefix, i], passed=i < 82) for i in range(100)]
    assert yield_record(first, .625, "definitive")["threshold_status"] == "unresolved"
    assert yield_record(second, .625, "definitive")["threshold_status"] == "above"
    result = paired_yield(first, second)
    assert result["difference_second_minus_first"] == .20 and result["exact_mcnemar_p"] < .001
    with pytest.raises(ValueError, match="identical draw"):
        paired_yield(first, second[::-1])


def test_related_lineages_are_one_paired_unit():
    tasks = [small_task("A", ancestry=("one",)), small_task("B", ancestry=("one", "two")),
             small_task("C", ancestry=("two",)), small_task("D", ancestry=("three",))]
    labels = clusters(tasks)
    assert labels[0] == labels[1] == labels[2] != labels[3]
    assert signed_randomization([1, 1, 1], labels[:3]) is None
    assert signed_randomization([1, 1, 1, 1], labels) == .5

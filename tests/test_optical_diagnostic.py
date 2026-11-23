import asyncio
import sys

import pytest

from lux_frontier.grpo import Completion
from lux_frontier.optical_diagnostic import freeze_choice, heuristic_ranking, selection_inputs, summarize, verify_common_draws
from lux_frontier.optical_grading import OpticalSettings, ScreenedCandidate, select_shipped
from lux_frontier.synthetic_training import make_environment
from lux_frontier.training import Settings, run_live, validate_task
from lux_frontier.training_tasks import FixedTaskSource, TrainingTask


def diagnostic_rows():
    screens = [15, 10, 12, 14, 18, 11, 13, 9]
    fresh = [40, 99, 60, 80, 20, 95, 51, 90]
    return [dict(candidate_id=i, nominal_pass=True, screening=dict(builds_passed=s),
                 refined_prescription_sha256=f"{i:064x}",
                 evaluation=dict(status="completed", builds_passed=f, builds_tested=100, yield_fraction=f / 100,
                                 builds=[dict(passed=index < f, draw=dict(index=index)) for index in range(100)]),
                 final_pass=f >= 90) for i, s, f in zip(range(1, 9), screens, fresh, strict=True)]


def test_all_28_pairs_exact_random_baseline_and_distinct_hindsight():
    report = summarize(diagnostic_rows(), [2, 8], 0.9)
    assert len(report["pairs"]) == 28
    random = report["baselines"]["uniform_random_pair"]
    # Ascending screen rank gives winner frequencies 0,1,...,7 among 28 pairs.
    assert random["exact_expected_shipped_yield"] == "1453/2800"
    assert random["exact_probability_of_passing"] == "3/28"
    assert report["baselines"]["try_all_eight"]["shipped_candidate_id"] == 5
    assert report["baselines"]["try_all_eight"]["shipped_yield"] == 0.2
    assert report["hindsight"] == dict(deployable_selector=False, highest_fresh_yield=0.99, candidate_ids=[2])
    assert report["starters_meeting_target"] == 3
    assert len(report["distinct_shipped_rewards"]) == 7


def test_no_fresh_outcome_can_change_frozen_shipping_choices():
    rows = diagnostic_rows()
    original = summarize(rows, [2, 8], 0.9)
    for row in rows:
        row["evaluation"]["yield_fraction"] = 1 - row["evaluation"]["yield_fraction"]
        row["evaluation"]["builds_passed"] = 100 - row["evaluation"]["builds_passed"]
    changed = summarize(rows, [2, 8], 0.9)
    assert [p["shipped_candidate_id"] for p in original["pairs"]] == [p["shipped_candidate_id"] for p in changed["pairs"]]
    assert original["baselines"]["try_all_eight"]["shipped_candidate_id"] == changed["baselines"]["try_all_eight"]["shipped_candidate_id"]
    assert not hasattr(selection_inputs(rows)[0], "evaluation")


def test_paired_counts_and_screening_order_reversal():
    report = summarize(diagnostic_rows(), [2, 8], 0.9)
    assert report["paired_3_7"] == dict(both_passed=51, only_3_passed=9, only_7_passed=0, neither_passed=40)
    assert report["ordering_3_7"] == dict(screening_pass_difference_7_minus_3=1, fresh_pass_difference_7_minus_3=-9,
                                         screening_lead_predicted_strict_fresh_order=False)


def test_pair_with_two_nominal_failures_is_a_failed_zero_reward_submission():
    rows = diagnostic_rows()
    for row in rows[:2]:
        row.update(nominal_pass=False, screening=None, evaluation=None, final_pass=False)
    report = summarize(rows, [1, 2], 0.9)
    failed = report["baselines"]["closest_to_spec"]
    assert failed["failed_submission"] and failed["reward"] == 0 and not failed["final_pass"]
    assert sum(p["failed_submission"] for p in report["pairs"]) == 1


def test_shipping_tie_break_and_summary_are_invariant_to_id_remapping():
    rows = diagnostic_rows()
    a = summarize(rows, [2, 8], 0.9)
    for row in rows:
        row["candidate_id"] += 100
    b = summarize(list(reversed(rows)), [102, 108], 0.9)
    assert a["baselines"]["uniform_random_pair"] == b["baselines"]["uniform_random_pair"]
    assert b["baselines"]["try_all_eight"]["refined_prescription_sha256"] == a["baselines"]["try_all_eight"]["refined_prescription_sha256"]
    assert select_shipped([ScreenedCandidate(999, True, 10, "b"), ScreenedCandidate(7, True, 10, "a")]).candidate_id == 7
    with pytest.raises(ValueError, match="requires completed screening"):
        select_shipped([ScreenedCandidate(1, True, None, "a")])
    assert freeze_choice([ScreenedCandidate(1, False, None, "a")])["shipped_candidate_id"] is None


def test_heuristic_uses_only_prompt_precision_and_ignores_ids_order():
    env = make_environment()
    original = heuristic_ranking(env.case, env.tolerance_profile)
    payload = env.case.model_dump()
    payload["candidates"] = list(reversed(payload["candidates"]))
    for c in payload["candidates"]:
        c["candidate_id"] += 100
        c["nominal_metrics"]["effective_focal_length_mm"] += 1e-10  # invisible under :g
    changed = heuristic_ranking(type(env.case).model_validate(payload), env.tolerance_profile)
    assert [(r["candidate_id"] + 100, r["score"], r["tie_break_sha256"]) for r in original] == [
        (r["candidate_id"], r["score"], r["tie_break_sha256"]) for r in changed]


def test_offline_training_dispatch_keeps_fractional_below_target_reward(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "tinker", None)
    env = make_environment()
    calls = []

    def grader(environment, submission):
        calls.append(submission.starter_ids)
        return 0.51

    task = TrainingTask(env, grader)
    record = validate_task(task, '{"starter_ids":[3,7]}', tmp_path)
    assert record.data["groups"][0]["completions"][0]["reward"] == 0.51
    assert record.data["backend"]["tinker_calls"] == 0 and calls == [(3, 7)]
    assert task.grade('{"starter_ids":[3,3]}').reward == 0
    assert calls == [(3, 7)]


def test_live_loop_uses_the_same_task_render_parse_and_grader_dispatch(tmp_path):
    env = make_environment()
    task = TrainingTask(env, lambda *_: 0.51)
    seen = []

    class OfflineBackend:
        async def initialize(self, training):
            return {"offline_mock": True}

        def encode(self, text):
            seen.append(text)
            return [10, 20]

        async def sample(self, tokens, count, seed):
            raw = '{"starter_ids":[3,7]}'
            return [Completion(raw, raw, [30], [-0.1], "stop", True) for _ in range(count)]

        async def close(self, status):
            pass

    record = asyncio.run(run_live(Settings(samples_per_brief=2), "sample", tmp_path,
                                   backend=OfflineBackend(), task_source=FixedTaskSource(task)))
    assert seen == [task.render()]
    assert [c["reward"] for c in record.data["groups"][0]["completions"]] == [0.51, 0.51]
    assert [c["parsed_choices"] for c in record.data["groups"][0]["completions"]] == [(3, 7), (3, 7)]


def test_common_draw_verification_rejects_candidate_specific_or_overlapping_streams():
    settings = OpticalSettings(screening_builds=2, evaluation_builds=3)
    rows = []
    for candidate_id in (3, 7):
        row = dict(candidate_id=candidate_id, nominal_pass=True)
        for phase, count, seed, code in [("screening", 2, settings.screening_seed, 100),
                                         ("evaluation", 3, settings.evaluation_seed, 200)]:
            row[phase] = dict(builds=[dict(draw=dict(phase=phase, seed=seed, index=i, stream_key=[seed, code, i])) for i in range(count)])
        rows.append(row)
    assert verify_common_draws(rows, settings)["disjoint_phases"]
    rows[1]["evaluation"]["builds"][0]["draw"]["unexpected_candidate_specific_value"] = True
    with pytest.raises(ValueError, match="differ across candidates"):
        verify_common_draws(rows, settings)

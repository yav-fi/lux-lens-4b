import copy
import json
import sys

import pytest

from lux_frontier import baseline
from lux_frontier.contracts import BriefCase
from lux_frontier.optical_prescriptions import DevelopmentProfile


def test_packaged_responses_replay_without_outputs_sdk_or_optical_work(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("offline replay attempted new optical work or model access")

    import lux_frontier.optical_batch as batch
    from lux_frontier.optical_grading import OpticalGrader
    monkeypatch.setattr(batch, "measure", forbidden)
    monkeypatch.setattr(OpticalGrader, "refine", forbidden)
    monkeypatch.setattr(OpticalGrader, "simulate", forbidden)
    monkeypatch.setitem(sys.modules, "tinker", None)
    monkeypatch.setitem(sys.modules, "tinker_cookbook", None)
    monkeypatch.chdir(tmp_path)
    report = baseline.replay()
    assert report["tinker_calls"] == report["new_optical_simulations"] == 0
    assert report["pairs_verified"] == 224
    assert report["summary"] == dict(valid_answers=8, passes=2, denominator=8,
        mean_reward=.31, heuristic_mean_reward=.31,
        exact_random_mean_reward="419/1600", exact_random_expected_passes="13/7")
    assert [r["reward"] for r in report["briefs"]] == [1., .13, .95, 0., 0., 0., .22, .18]
    assert list(tmp_path.iterdir()) == []


def test_cached_ingress_keeps_strict_json_and_truncation_policy():
    _, cache = baseline.load_result()
    item = cache["briefs"][0]
    case = BriefCase.model_validate(item["case"])
    profile = DevelopmentProfile.model_validate(cache["profile"])
    for response, clean in [('{}', True), ('{"starter_ids":[2,2]}', True),
                            ('{"starter_ids":[2,999]}', True),
                            ('```json\n{"starter_ids":[2,7]}\n```', True),
                            ('{"starter_ids":[2,7]}', False)]:
        grade = baseline.score(case, profile, item["pairs"], response, clean=clean)
        assert not grade["valid"] and grade["reward"] == 0 and not grade["final_pass"]
    assert baseline.score(case, profile, item["pairs"], '{"starter_ids":[7,2]}')["reward"] == 1.


def test_replay_rejects_modified_cache_before_scoring(tmp_path):
    result, cache = baseline.load_result()
    cache["briefs"][0]["pairs"][0]["reward"] = .123
    (tmp_path / result["replay_cache"]["file"]).write_text(json.dumps(cache))
    path = tmp_path / baseline.RESULT_NAME
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="cache hash mismatch"):
        baseline.replay(path)


def test_shipping_rule_and_prompt_integrity_are_verified_independently():
    _, cache = baseline.load_result()
    profile = DevelopmentProfile.model_validate(cache["profile"])
    item = copy.deepcopy(cache["briefs"][1])
    item["pairs"][0]["reward"] = .123
    with pytest.raises(ValueError, match="frozen screening/scoring rule"):
        baseline.validated_pairs(item, profile)
    item = copy.deepcopy(cache["briefs"][1])
    item["case"]["candidates"][0]["geometry"]["minimum_edge_thickness_mm"] += .01
    with pytest.raises(ValueError, match="prompt hash mismatch"):
        baseline.validated_pairs(item, profile)


def test_conveyor_diagnosis_is_a_single_substitution_under_actual_screening_rule():
    comparison = baseline.conveyor_comparison()
    assert comparison["diagnostic_only"]
    assert comparison["qwen_pair"] == [3, 5]
    assert comparison["comparison_pair"] == [3, 7]
    assert comparison["substitution"] == dict(removed=[5], added=[7])
    assert comparison["qwen_score"]["shipped_candidate_id"] == 3
    assert comparison["qwen_score"]["reward"] == .13
    assert comparison["comparison_score"]["shipped_candidate_id"] == 7
    assert comparison["comparison_score"]["reward"] == .57
    assert [comparison["post_grading"][str(i)]["screening"]["builds_passed"] for i in (3, 7)] == [4, 9]
    failures = comparison["post_grading"]["5"]["nominal_failures"]
    assert [(f["constraint"], f["measured"], f["minimum"]) for f in failures] == [
        ("mtf[0]", .5997368501792373, .65), ("mtf[1]", .3428894628597908, .35)]
    assert "Minimum edge thickness: 0.818028 mm." in comparison["prompt_visible_blocks"]["7"]

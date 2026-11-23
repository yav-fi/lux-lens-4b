import json
from dataclasses import replace
from itertools import combinations

import pytest

from lux_frontier.contracts import BriefCase
from lux_frontier.grpo import Completion, group_advantages, rl_arrays
from lux_frontier.prompting import ResolvedToleranceProfile, render_prompt, resolve_profile
from lux_frontier.synthetic_fixture import synthetic_fixture
from lux_frontier.synthetic_training import candidate_score, grade_response, make_environment


def test_prompt_contains_requirements_assumptions_and_eight_distinct_blocks():
    env = make_environment()
    text = render_prompt(env.case, env.tolerance_profile)
    b = env.case.brief
    for value in [f"{b.target_focal_length_mm:g} +/- {b.focal_length_tolerance_mm:g} mm",
                  f"{b.max_first_surface_to_sensor_mm:g} mm", f"{b.target_f_number:g} +/- {b.f_number_tolerance:g}",
                  f"{b.minimum_simulated_yield_fraction:g} (fraction)", '"starter_ids":[3,7]',
                  "truncated_normal", "uniform", "standard deviation", "hard bounds", "dimensionless",
                  "deg", "lp/mm", "um", "sagittal", "tangential", "sensor shift bound +/- 0.2 mm",
                  "Error correlations:", "SYNTHETIC"]:
        assert value in text
    assert text.count("Candidate ID ") == 8
    assert len({c.layout for c in env.case.candidates}) == 8
    for c in env.case.candidates:
        block = text.split(f"Candidate ID {c.candidate_id}:\n")[1].split("\n\n")[0]
        for value in [c.layout, c.glass_types[0], f"{c.nominal_metrics.effective_focal_length_mm:g} mm",
                      f"{c.geometry.minimum_edge_thickness_mm:g} mm", "1/mm", "Nominal MTF"]:
            assert value in block
    assert "|" not in text  # no Markdown tables


def test_hidden_grading_and_artifact_references_are_not_model_input():
    env = make_environment()
    text = render_prompt(env.case, env.tolerance_profile)
    for forbidden in ["reward", "candidate_score", "final_pass", "expected_answer", "toy_yield", "builds_passed",
                      env.case.brief.brief_id, env.tolerance_profile.reference.uri,
                      *(c.prescription_ref.uri for c in env.case.candidates)]:
        assert forbidden not in text
    # Grading cannot mutate the immutable prompt inputs.
    grade_response(env, '{"starter_ids":[3,7]}')
    assert render_prompt(env.case, env.tolerance_profile) == text
    assert all(c.prescription_ref.revision == "v1" for c in env.case.candidates)


def test_resolution_requires_exact_revision_and_renderer_requires_contents():
    env = make_environment()
    profile = env.tolerance_profile
    key = (profile.reference.uri, profile.reference.revision)
    assert resolve_profile(profile.reference, {key: profile}) == profile
    with pytest.raises(ValueError, match="cannot be resolved"):
        resolve_profile(profile.reference, {})
    with pytest.raises(ValueError, match="unresolved"):
        render_prompt(env.case, None)
    wrong = profile.model_copy(update={"reference": profile.reference.model_copy(update={"revision": "v2"})})
    with pytest.raises(ValueError, match="does not match"):
        render_prompt(env.case, wrong)
    with pytest.raises(ValueError, match="cannot be resolved"):
        resolve_profile(profile.reference, {key: wrong})


@pytest.mark.parametrize("change", [
    {"lower_bound": 1, "upper_bound": -1}, {"standard_deviation": None}, {"units": "mm"},
])
def test_invalid_resolved_tolerances(change):
    payload = make_environment().tolerance_profile.model_dump()
    payload["errors"][0].update(change)
    with pytest.raises(ValueError):
        ResolvedToleranceProfile.model_validate(payload)


def test_original_contract_fixture_still_works():
    case, result = synthetic_fixture()
    assert case.parse_submission('{"starter_ids":[3,7]}').starter_ids == (3, 7)
    assert case.parse_result(result.model_dump_json()) == result
    assert all(c.geometry is None for c in case.candidates)


def test_reproducible_variation_and_disjoint_splits():
    assert make_environment(23, "train", 2) == make_environment(23, "train", 2)
    cases = [make_environment(23, split, i).case for split in ("train", "eval") for i in range(6)]
    assert len({c.brief.brief_id for c in cases}) == 12
    assert len({c.brief.target_focal_length_mm for c in cases}) > 1
    assert len({c.candidates[0].nominal_metrics for c in cases}) > 1
    assert len({tuple(c.candidate_id for c in case.candidates) for case in cases}) > 1
    for case in cases:
        assert set(c.candidate_id for c in case.candidates) == set(range(1, 9))
        assert len({c.prescription_ref.uri for c in case.candidates}) == 8
        assert BriefCase.model_validate_json(case.model_dump_json()) == case


def test_remapping_and_order_do_not_change_feature_based_scores():
    env = make_environment()
    mapping = {c.candidate_id: 100 + 7 * i for i, c in enumerate(env.case.candidates)}
    payload = env.case.model_dump()
    payload["candidates"] = list(reversed(payload["candidates"]))
    for c in payload["candidates"]:
        c["candidate_id"] = mapping[c["candidate_id"]]
    changed = replace(env, case=BriefCase.model_validate(payload))
    for a, b in combinations(mapping, 2):
        first = grade_response(env, json.dumps({"starter_ids": [a, b]}))
        remapped = grade_response(changed, json.dumps({"starter_ids": [mapping[a], mapping[b]]}))
        assert first.reward == remapped.reward
    assert len({grade_response(env, json.dumps({"starter_ids": pair})).reward
                for pair in combinations(mapping, 2)}) == 28


@pytest.mark.parametrize("raw", ["", "{", '```json\n{"starter_ids":[3,7]}\n```',
    '{"starter_ids":[3,3]}', '{"starter_ids":[3,999]}', '{"starter_ids":[3]}',
    '{"starter_ids":[3,7,2]}', '{"starter_ids":["3",7]}', '{"starter_ids":[true,7]}',
    '{"starter_ids":[3,7],"reason":"x"}', '{"starter_ids":[3,7]} trailing',
])
def test_parser_failures_receive_zero_without_repair(raw):
    def forbidden_grader(*_):
        pytest.fail("grader must not run for invalid submissions")
    result = grade_response(make_environment(), raw, grader=forbidden_grader)
    assert result.reward == 0 and result.parsed_choices is None and result.error
    assert result.raw_response == raw


def test_even_complete_json_is_zero_if_generation_was_truncated():
    raw = '{"starter_ids":[3,7]}'
    result = grade_response(make_environment(), raw, truncated=True,
                            grader=lambda *_: pytest.fail("truncated response graded"))
    assert result.raw_response == raw and result.parsed_choices == (3, 7)
    assert result.reward == 0 and "truncated" in result.error


def test_manual_synthetic_rule_and_replaceable_grader():
    env = make_environment()
    payload = env.case.model_dump()
    b = env.case.brief
    for index, c in enumerate(payload["candidates"][:2]):
        c["geometry"].update(element_count=3, minimum_edge_thickness_mm=1,
                             minimum_air_gap_mm=1, maximum_absolute_curvature_per_mm=0.1)
        c["nominal_metrics"].update(effective_focal_length_mm=b.target_focal_length_mm + index * b.focal_length_tolerance_mm,
                                     f_number=b.target_f_number, first_surface_to_sensor_mm=b.max_first_surface_to_sensor_mm)
        for m, t in zip(c["nominal_metrics"]["mtf"], b.mtf_thresholds):
            m["mtf_fraction"] = t.minimum_mtf_fraction
    env = replace(env, case=BriefCase.model_validate(payload))
    first, second = env.case.candidates[:2]
    # Stress = (.06 + .03 + .06 + .03 + .012) * .8 = .1536.
    # Toy yield ~ .867 exceeds this brief's .85; only focal penalty differs.
    assert candidate_score(env, first) == 1
    assert candidate_score(env, second) == 0.5
    raw = json.dumps({"starter_ids": [first.candidate_id, second.candidate_id]})
    assert grade_response(env, raw).reward == 0.75
    assert grade_response(env, raw).reward == grade_response(env, raw).reward
    assert grade_response(env, raw, grader=lambda e, s: 0.123).reward == 0.123
    with pytest.raises(ValueError, match="non-finite"):
        grade_response(env, raw, grader=lambda *_: float("nan"))


def test_constant_rewards_and_next_token_alignment():
    assert group_advantages([0.1] * 8) == [0] * 8
    assert group_advantages([0] * 8) == [0] * 8
    assert group_advantages([0, 0.5, 1]) == [-0.5, 0, 0.5]
    completion = Completion("raw", "text", [31, 32], [-0.3, -0.4], "stop", True)
    assert rl_arrays([11, 12, 13], completion, -0.5) == dict(
        input_tokens=[11, 12, 13, 31], target_tokens=[12, 13, 31, 32],
        logprobs=[0, 0, -0.3, -0.4], advantages=[0, 0, -0.5, -0.5])
    for bad in [replace(completion, logprobs=None), replace(completion, logprobs=[-1]),
                replace(completion, logprobs=[float("nan"), -1])]:
        with pytest.raises(ValueError):
            rl_arrays([11], bad, 1)

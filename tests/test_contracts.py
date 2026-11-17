import json

import pytest
from pydantic import ValidationError

from lux_frontier.contracts import BriefCase, GraderResult, SimulationResult, Submission
from lux_frontier.synthetic_fixture import synthetic_fixture


def test_synthetic_json_round_trip():
    case, result = synthetic_fixture()
    assert BriefCase.model_validate_json(case.model_dump_json()) == case
    assert case.parse_result(result.model_dump_json()) == result
    submission = case.parse_submission('{"starter_ids":[3,7]}')
    assert Submission.model_validate_json(submission.model_dump_json()) == submission
    assert json.loads(submission.model_dump_json()) == {"starter_ids": [3, 7]}
    # Measurements retain their field, frequency, wavelength, and orientation.
    assert case.candidates[0].nominal_metrics.mtf[0].field_angle_deg == (0, 0)
    assert case.candidates[0].nominal_metrics.mtf[0].spatial_frequency_lp_per_mm == 30


@pytest.mark.parametrize("ids", [[3, 3], [3, 999], [3], [3, 7, 9], ["3", 7], [True, 7]])
def test_invalid_submissions(ids):
    case, _ = synthetic_fixture()
    with pytest.raises(ValueError):
        case.parse_submission(json.dumps({"starter_ids": ids}))


def test_brief_namespace_and_result_binding():
    case, result = synthetic_fixture()
    payload = json.loads(case.model_dump_json())
    payload["candidates"][1]["candidate_id"] = 3
    with pytest.raises(ValidationError, match="unique"):
        BriefCase.model_validate(payload)
    payload = json.loads(result.model_dump_json())
    payload["brief_id"] = "different-brief"
    with pytest.raises(ValueError, match="different brief"):
        case.parse_result(json.dumps(payload))
    payload["brief_id"] = case.brief.brief_id
    payload["starter_ids"] = [3, 999]
    with pytest.raises(ValueError, match="unknown"):
        case.parse_result(json.dumps(payload))


def test_unmeasured_is_distinct_from_zero_yield():
    unrun = SimulationResult()
    failed = SimulationResult(status="failed", error="trace failed")
    zero = SimulationResult(status="completed", builds_passed=0,
                            builds_tested=100, yield_fraction=0)
    for simulation in (unrun, failed, zero):
        assert SimulationResult.model_validate_json(simulation.model_dump_json()) == simulation
    assert unrun.yield_fraction is failed.yield_fraction is None
    assert zero.yield_fraction == 0


@pytest.mark.parametrize("payload", [
    {"status": "completed"},
    {"status": "completed", "builds_passed": 0, "builds_tested": 0, "yield_fraction": 0},
    {"status": "completed", "builds_passed": 101, "builds_tested": 100, "yield_fraction": 1},
    {"status": "completed", "builds_passed": 93, "builds_tested": 100, "yield_fraction": 0.9},
    {"status": "not_run", "yield_fraction": 0},
    {"status": "failed"},
    {"status": "failed", "error": "broken", "builds_tested": 100},
])
def test_inconsistent_simulation_rejected(payload):
    with pytest.raises(ValidationError):
        SimulationResult.model_validate(payload)


@pytest.mark.parametrize("field,value", [
    ("minimum_simulated_yield_fraction", 90),
    ("target_focal_length_mm", float("nan")),
    ("f_number_tolerance", -0.1),
    ("unexpected", "typo"),
])
def test_units_and_invalid_numbers(field, value):
    case, _ = synthetic_fixture()
    payload = case.brief.model_dump()
    payload[field] = value
    with pytest.raises(ValidationError):
        type(case.brief).model_validate(payload)


def test_pending_result_and_incomplete_pass():
    case, _ = synthetic_fixture()
    pending = GraderResult(brief_id=case.brief.brief_id, starter_ids=(3, 7))
    assert pending.final_pass is pending.reward is None
    assert case.parse_result(pending.model_dump_json()) == pending
    payload = pending.model_dump()
    payload["final_pass"] = True
    with pytest.raises(ValidationError, match="pass requires"):
        GraderResult.model_validate(payload)

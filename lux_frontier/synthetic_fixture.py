"""SYNTHETIC contract example, not an Optiland fixture or manufacturing evidence.

Artifact URIs are versioned placeholders, not resolvable optical prescriptions.
All measurements, build counts, decisions, and reward below are invented.
Run ``python -m lux_frontier.synthetic_fixture`` for JSON and rejection examples.
"""

import json

from lux_frontier.contracts import BriefCase, GraderResult


def synthetic_fixture() -> tuple[BriefCase, GraderResult]:
    point = dict(field_angle_deg=(0, 0), spatial_frequency_lp_per_mm=30,
                 wavelength_um=0.55, orientation="sagittal")
    off_axis = dict(field_angle_deg=(0, 20), spatial_frequency_lp_per_mm=60,
                    wavelength_um=0.55, orientation="tangential")
    metrics = dict(effective_focal_length_mm=50, first_surface_to_sensor_mm=57,
                   f_number=5, mtf=[dict(point, mtf_fraction=0.7),
                                    dict(off_axis, mtf_fraction=0.25)])
    case = BriefCase.model_validate({
        "brief": {
            "brief_id": "synthetic-001",
            "target_focal_length_mm": 50,
            "focal_length_tolerance_mm": 1,
            "max_first_surface_to_sensor_mm": 58,
            "target_f_number": 5,
            "f_number_tolerance": 0.1,
            "mtf_thresholds": [dict(point, minimum_mtf_fraction=0.65),
                               dict(off_axis, minimum_mtf_fraction=0.2)],
            "minimum_simulated_yield_fraction": 0.9,
            "manufacturing_tolerance_profile_ref": {
                "uri": "synthetic://tolerance-profiles/small-spherical", "revision": "v1",
            },
        },
        "candidates": [{
            "candidate_id": candidate_id,
            "prescription_ref": {"uri": f"synthetic://prescriptions/{candidate_id}", "revision": "v1"},
            "layout": "Three separated spherical elements: positive / negative / positive; stop after element 2.",
            "glass_types": ["hikari:SK16", "schott:F2", "hikari:SK16"],
            "nominal_metrics": metrics,
        } for candidate_id in (3, 7)],
    })
    submission = case.parse_submission('{"starter_ids":[3,7]}')
    result = GraderResult(
        brief_id=case.brief.brief_id, starter_ids=submission.starter_ids,
        shipped_prescription_ref=case.candidates[0].prescription_ref,
        nominal_metrics=case.candidates[0].nominal_metrics,
        simulation={"status": "completed", "builds_passed": 93,
                    "builds_tested": 100, "yield_fraction": 0.93},
        final_pass=True, reward=0.5,
    )
    return case, result


def main() -> None:
    case, result = synthetic_fixture()
    submission = case.parse_submission('{"starter_ids":[3,7]}')
    assert BriefCase.model_validate_json(case.model_dump_json()) == case
    assert case.parse_result(result.model_dump_json()) == result
    print(json.dumps({
        "synthetic": True,
        "brief": case.brief.model_dump(mode="json"),
        "candidate": case.candidates[0].model_dump(mode="json"),
        "submission": submission.model_dump(mode="json"),
        "grader_result": result.model_dump(mode="json"),
    }, indent=2))
    print("JSON round-trip: OK")
    for payload in ('{"starter_ids":[3,3]}', '{"starter_ids":[3,999]}'):
        try:
            case.parse_submission(payload)
        except ValueError:
            print(f"Rejected: {payload}")
        else:
            raise AssertionError(f"unexpectedly accepted {payload}")


if __name__ == "__main__":
    main()

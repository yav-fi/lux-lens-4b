import pytest

from lux_frontier.optical_fine_diagnostic import PAIRED_DEVELOPMENT_SEED, freeze_selectors, paired_selector_summary, unique_confirmation_designs


def screening_rows():
    counts = [15, 10, 12, 18, 17, 11, 19, 9]
    return [dict(candidate_id=i, nominal_pass=True, screening=dict(builds_passed=n),
                 refined_prescription_sha256=f"{i:064x}") for i, n in zip(range(1, 9), counts, strict=True)]


def paired_builds(passed_indices):
    return [dict(passed=i in passed_indices, draw=dict(seed=PAIRED_DEVELOPMENT_SEED, index=i)) for i in range(100)]


def test_finer_screening_controls_shipping_with_evaluation_outcomes_excluded():
    rows = screening_rows()
    plan = freeze_selectors(rows, [5, 4])
    assert len(plan["pairs"]) == 28
    assert plan["heuristic"]["shipped_candidate_id"] == 4
    assert plan["try_all_eight"]["shipped_candidate_id"] == 7
    rows[3]["screening"]["builds_passed"] = 11
    assert freeze_selectors(rows, [5, 4])["heuristic"]["shipped_candidate_id"] == 5
    rows[0]["evaluation"] = dict(yield_fraction=1)
    with pytest.raises(ValueError, match="must not contain evaluation"):
        freeze_selectors(rows, [5, 4])


def test_same_physical_design_is_evaluated_once_and_has_zero_paired_difference():
    rows = screening_rows()
    rows[4]["screening"]["builds_passed"] = 20
    plan = freeze_selectors(rows, [5, 4])
    assert plan["heuristic"]["shipped_candidate_id"] == plan["try_all_eight"]["shipped_candidate_id"] == 5
    assert len(unique_confirmation_designs(plan)) == 1
    values = paired_builds(set(range(76)))
    report = paired_selector_summary(values, values)
    assert report["counts"] == dict(both_passed=76, heuristic_only_passed=0, try_all_only_passed=0, neither_passed=24)
    assert report["try_all_minus_heuristic_percentage_points"] == 0


def test_deduplication_uses_prescription_identity_even_with_different_local_ids():
    plan = freeze_selectors(screening_rows(), [5, 4])
    plan["try_all_eight"]["refined_prescription_sha256"] = plan["heuristic"]["refined_prescription_sha256"]
    assert len(unique_confirmation_designs(plan)) == 1


def test_paired_wins_losses_gap_and_fixed_90_threshold():
    a, b = paired_builds(set(range(70))), paired_builds(set(range(10, 90)))
    result = paired_selector_summary(a, b)
    assert result["counts"] == dict(both_passed=60, heuristic_only_passed=10, try_all_only_passed=20, neither_passed=10)
    assert result["try_all_minus_heuristic_percentage_points"] == 10
    assert not result["heuristic_reaches_90"] and not result["try_all_eight_reaches_90"]
    exact = paired_selector_summary(paired_builds(set(range(90))), paired_builds(set(range(89))))
    assert exact["heuristic_reaches_90"] and not exact["try_all_eight_reaches_90"]
    assert exact["try_all_minus_heuristic_percentage_points"] == -1


def test_new_draws_are_paired_and_reserved_seed_differs_from_all_previous_streams():
    assert PAIRED_DEVELOPMENT_SEED not in (20261123, 20261124, 2026120101)
    a, b = paired_builds(set()), paired_builds(set())
    b[0]["draw"]["seed"] = 9
    with pytest.raises(ValueError, match="different paired"):
        paired_selector_summary(a, b)
    with pytest.raises(ValueError, match="exactly 100"):
        paired_selector_summary(a[:99], a[:99])


def test_freezing_is_invariant_to_display_order_and_local_id_remapping():
    rows = screening_rows()
    original = freeze_selectors(rows, [5, 4])
    for row in rows:
        row["candidate_id"] += 100
    changed = freeze_selectors(list(reversed(rows)), [105, 104])
    for selector in ("heuristic", "try_all_eight"):
        assert original[selector]["refined_prescription_sha256"] == changed[selector]["refined_prescription_sha256"]

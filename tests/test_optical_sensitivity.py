import copy

import numpy as np
import pytest

from lux_frontier.contracts import NominalMetrics
from lux_frontier.optical_engine import constraint_failures, cooke_prescription
from lux_frontier.optical_grading import OpticalCache, OpticalGrader, OpticalSettings, build_draw
from lux_frontier.optical_prescriptions import ArtifactStore, development_profile
from lux_frontier.optical_sensitivity import CONFIRMATION_SEED, FAMILIES, MarginGrader, compare_builds, evaluate_focus, intervene, nested_focus_grid
from lux_frontier.synthetic_fixture import synthetic_fixture


def test_33_grid_retains_every_exact_original_position_and_same_travel():
    original, fine = nested_focus_grid(0.2, 9), nested_focus_grid(0.2, 33)
    assert len(fine) == 33 and set(original) <= set(fine)
    assert min(fine) == -0.2 and max(fine) == 0.2 and fine[0] == 0
    assert np.diff(sorted(fine)) == pytest.approx([0.0125] * 32)
    with pytest.raises(ValueError):
        nested_focus_grid(0.2, 17)


@pytest.mark.parametrize("family", FAMILIES)
def test_ablation_changes_only_one_realized_family_without_mutating_source(family):
    profile = development_profile()
    original = build_draw(profile, OpticalSettings(), "evaluation", 2)
    preserved = copy.deepcopy(original)
    changed = intervene(original, family)
    assert original == preserved
    assert np.all(np.asarray(changed["errors"][family]) == 0)
    for other in FAMILIES:
        if other != family:
            assert changed["errors"][other] == original["errors"][other]
    for key in ["seed", "phase", "index", "stream_key", "underlying_uniform_draws"]:
        assert changed[key] == original[key]


def test_zero_error_intervention_and_confirmation_development_streams():
    profile, settings = development_profile(), OpticalSettings()
    draw = build_draw(profile, settings, "evaluation", 0)
    zero = intervene(draw, "all")
    assert all(np.all(np.asarray(zero["errors"][name]) == 0) for name in FAMILIES)
    new = build_draw(profile, settings.model_copy(update={"evaluation_seed": CONFIRMATION_SEED}), "evaluation", 0)
    assert new["stream_key"] != draw["stream_key"]
    assert new["seed"] not in [settings.screening_seed, settings.evaluation_seed]
    assert new["errors"] != draw["errors"]


def test_nested_replay_reuses_nine_scores_and_one_common_sensor_per_trial(tmp_path, monkeypatch):
    import lux_frontier.optical_sensitivity as module

    case, _ = synthetic_fixture()
    p, profile, settings = cooke_prescription(), development_profile(), OpticalSettings()
    draw = build_draw(profile, settings, "evaluation", 0)
    calls = []
    target_shift = min(v for v in nested_focus_grid(0.2, 33) if v > 0)

    def measurement(shift, ratio):
        data = case.candidates[0].nominal_metrics.model_dump()
        for value, requirement in zip(data["mtf"], case.brief.mtf_thresholds, strict=True):
            value["mtf_fraction"] = requirement.minimum_mtf_fraction * ratio
        metrics = NominalMetrics.model_validate(data)
        return metrics, dict(sensor_shift_mm=shift, sensor_z_mm=57 + shift, pupil_samples=64)

    def fake_measure(prescription, brief, *, pupil_samples, errors, sensor_shift_mm):
        calls.append(sensor_shift_mm)
        return measurement(sensor_shift_mm, 1.2 if sensor_shift_mm == target_shift else 0.6)

    old_metrics, old_details = measurement(0, 0.8)
    original = dict(metrics=old_metrics.model_dump(mode="json"), details=old_details,
                    failures=constraint_failures(old_metrics, case.brief), draw=draw,
                    prescription_ref=p.reference.model_dump(mode="json"),
                    refocus_trials=[dict(sensor_shift_mm=v, minimum_mtf_ratio=0.8 if v == 0 else 0.5)
                                    for v in nested_focus_grid(0.2, 9)])
    monkeypatch.setattr(module, "measure", fake_measure)
    cache = OpticalCache(tmp_path)
    result = evaluate_focus(p, case.brief, profile, settings, draw, nested_focus_grid(0.2, 33), cache, saved_build=original)
    assert len(calls) == 24 and not set(calls) & set(nested_focus_grid(0.2, 9))
    assert result["work"]["original_trials_reused"] == 9
    assert result["work"]["new_metric_evaluations"] == 24
    assert result["passed"]
    assert result["details"]["sensor_shift_mm"] == target_shift
    # All MTF measurements were returned by a single common-sensor call.
    assert len(result["metrics"]["mtf"]) == len(case.brief.mtf_thresholds)
    changed = intervene(draw, "tilt_deg")
    with pytest.raises(ValueError, match="realized errors"):
        evaluate_focus(p, case.brief, profile, settings, changed, nested_focus_grid(0.2, 33), cache, saved_build=original)
    wrong = dict(original, prescription_ref={"uri": "wrong", "revision": "wrong"})
    with pytest.raises(ValueError, match="different prescription"):
        evaluate_focus(p, case.brief, profile, settings, draw, nested_focus_grid(0.2, 33), cache, saved_build=wrong)


def test_constraint_transitions_record_repairs_and_new_failures():
    def build(passed, failures):
        return dict(passed=passed, failures=[dict(constraint=f) for f in failures])

    before = [build(True, []), build(False, ["mtf[0]"]), build(False, ["mtf[1]"])]
    after = [build(True, []), build(True, []), build(False, ["mtf[2]"])]
    result = compare_builds(before, after)
    assert result["transitions"] == dict(both_passed=1, fail_to_pass=1, pass_to_fail=0, both_failed=1)
    assert result["constraint_repairs"] == {"mtf[0]": 1, "mtf[1]": 1}
    assert result["constraint_introductions"] == {"mtf[2]": 1}


def test_margin_merit_is_cache_separated_and_reuses_original_budget(tmp_path):
    case, _ = synthetic_fixture()
    store, cache = ArtifactStore(tmp_path / "artifacts"), OpticalCache(tmp_path / "cache")
    settings, profile, p = OpticalSettings(), development_profile(), cooke_prescription()
    original, alternative = OpticalGrader(store, cache, settings, profile), MarginGrader(store, cache, settings, profile)
    assert original.context(p, case.brief) != alternative.context(p, case.brief)
    assert alternative.settings.refinement_evaluations == original.settings.refinement_evaluations == 37
    assert alternative.nominal_objective(case.candidates[0].nominal_metrics, case.brief) > original.nominal_objective(case.candidates[0].nominal_metrics, case.brief)

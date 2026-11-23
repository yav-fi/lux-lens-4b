import json
from types import SimpleNamespace

import numpy as np
import pytest
from optiland.mtf import FFTMTF

from lux_frontier.contracts import BriefCase, GraderResult, NominalMetrics, Submission
from lux_frontier.optical_engine import BuildErrors, constraint_failures, cooke_prescription, geometry_summary, make_optic, measure, mtf_value, normalized_field, rotation_matrix
from lux_frontier.optical_grading import OpticalCache, OpticalGrader, OpticalSettings, build_draw, development_brief, make_development_case, replay_evaluation
from lux_frontier.optical_prescriptions import ArtifactStore, DevelopmentProfile, development_profile, verify_profile
from lux_frontier.prompting import render_prompt


@pytest.fixture(scope="module")
def optical(tmp_path_factory):
    directory = tmp_path_factory.mktemp("optical")
    store, cache = ArtifactStore(directory / "artifacts"), OpticalCache(directory / "cache")
    settings = OpticalSettings(pupil_samples=32, screening_builds=2, evaluation_builds=3, refocus_positions=3)
    profile = development_profile()
    case = make_development_case(store, cache, settings, profile)
    return SimpleNamespace(directory=directory, store=store, cache=cache, settings=settings, profile=profile, case=case)


def test_zero_perturbation_reproduces_actual_nominal_metrics():
    p = cooke_prescription()
    brief = development_brief(development_profile())
    nominal, _ = measure(p, brief, pupil_samples=32)
    perturbed, details = measure(p, brief, pupil_samples=32, errors=BuildErrors(), sensor_shift_mm=0)
    assert nominal == perturbed
    assert nominal.effective_focal_length_mm == pytest.approx(49.9997830714, abs=1e-5)
    assert details["sensor_z_mm"] == p.sensor_z_mm
    assert len(details["field_mapping"]) == len(brief.mtf_thresholds)


def test_whole_element_rigid_transforms_move_both_vertices():
    p = cooke_prescription()
    errors = BuildErrors(decenter_mm=((0, 0), (0.03, -0.02), (0, 0)),
                         tilt_deg=((0, 0), (0.03, -0.02), (0, 0)))
    lens = make_optic(p, errors)
    front, rear = lens.surfaces[3].geometry.cs, lens.surfaces[4].geometry.cs
    def vector(cs):
        return np.array([cs.x.item(), cs.y.item(), cs.z.item()])
    assert vector(front)[:2] == pytest.approx([0.03, -0.02])
    expected = rotation_matrix((0.03, -0.02)) @ np.array([0, 0, p.spacings_mm[2]])
    assert vector(rear) - vector(front) == pytest.approx(expected, abs=1e-13)
    assert np.linalg.norm(vector(rear) - vector(front)) == pytest.approx(p.spacings_mm[2])
    assert front.rx == rear.rx == pytest.approx(np.deg2rad(0.03))
    assert front.ry == rear.ry == pytest.approx(np.deg2rad(-0.02))
    assert lens.surfaces[5].geometry.cs.x == 0  # no unintended downstream rigid motion
    assert lens.image_surface.geometry.cs.z == p.sensor_z_mm


def test_center_and_air_gap_errors_have_separate_scopes_and_fixed_sensor():
    p = cooke_prescription()
    lens = make_optic(p, BuildErrors(center_thickness_mm=(0.01, 0, 0), air_gap_mm=(0.02, 0)))
    assert lens.surfaces[2].geometry.cs.z == pytest.approx(p.spacings_mm[0] + 0.01)
    assert lens.surfaces[3].geometry.cs.z == pytest.approx(sum(p.spacings_mm[:2]) + 0.03)
    assert lens.surfaces[5].geometry.cs.z == pytest.approx(sum(p.spacings_mm[:4]) + 0.03)
    assert lens.image_surface.geometry.cs.z == p.sensor_z_mm
    assert make_optic(p, sensor_shift_mm=0.125).image_surface.geometry.cs.z == pytest.approx(p.sensor_z_mm + 0.125)


def test_field_wavelength_frequency_and_orientation_against_pinned_fft_api():
    p = cooke_prescription()
    assert normalized_field(p, (0, 14)) == (0, 0.7)
    assert normalized_field(p, (0, 20)) == (0, 1)
    calculator = FFTMTF(make_optic(p), fields=[(0, 1)], wavelength=0.55, num_rays=32, grid_size=128,
                        strategy="chief_ray", remove_tilt=False)
    sagittal, detail = mtf_value(calculator, 0, "sagittal", 30)
    tangential, _ = mtf_value(calculator, 0, "tangential", 30)
    assert detail["orientation_index"] == 1
    assert sagittal == pytest.approx(np.interp(30, calculator.freq_sag[0], calculator.mtf[0][1]))
    assert tangential == pytest.approx(np.interp(30, calculator.freq_tang[0], calculator.mtf[0][0]))
    assert sagittal != pytest.approx(tangential, abs=0.001)
    brief = development_brief(development_profile())
    metrics, _ = measure(p, brief, pupil_samples=32)
    assert metrics.mtf[2].mtf_fraction == pytest.approx(sagittal)
    # A different configured wavelength is really traced, not merely relabeled.
    payload = brief.model_dump()
    payload["mtf_thresholds"] = [dict(payload["mtf_thresholds"][2], wavelength_um=0.65)]
    red, red_details = measure(p, type(brief).model_validate(payload), pupil_samples=32)
    assert red.mtf[0].wavelength_um == red_details["field_mapping"][0]["wavelength_um"] == 0.65
    assert red.mtf[0].mtf_fraction != pytest.approx(sagittal, abs=0.001)
    with pytest.raises(ValueError, match="unconfigured"):
        normalized_field(p, (1, 14))
    with pytest.raises(ValueError, match="exceeds"):
        mtf_value(calculator, 0, "sagittal", 1e8)


def test_eight_actual_traceable_candidates_and_optical_prompt(optical):
    case = optical.case
    assert len(case.candidates) == len({c.prescription_ref.uri for c in case.candidates}) == 8
    for candidate in case.candidates:
        p = optical.store.resolve(candidate.prescription_ref)
        assert candidate.geometry == geometry_summary(p)
        actual, _ = measure(p, case.brief, pupil_samples=32)
        assert candidate.nominal_metrics == actual
    prompt = render_prompt(case, optical.profile, provenance="development_optical")
    assert "DEVELOPMENT optical" in prompt and "PROVISIONAL" in prompt
    assert "SYNTHETIC training data" not in prompt
    assert "air_gap_error" in prompt and prompt.count("Candidate ID ") == 8
    assert "reward" not in prompt and "local://" not in prompt


def test_content_addressed_prescription_and_profile_integrity(optical):
    candidate = optical.case.candidates[0]
    p = optical.store.resolve(candidate.prescription_ref)
    assert optical.store.put(p) == candidate.prescription_ref
    verify_profile(optical.profile)
    payload = optical.profile.model_dump()
    payload["errors"][0]["upper_bound"] *= 1.1
    modified = DevelopmentProfile.model_validate(payload)
    with pytest.raises(ValueError, match="immutable"):
        verify_profile(modified)
    fake_store = ArtifactStore(optical.directory / "tamper")
    ref = fake_store.put(p)
    path = next(fake_store.directory.glob("*.json"))
    data = json.loads(path.read_text())
    data["sensor_z_mm"] += 0.01
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="immutable"):
        fake_store.resolve(ref)


@pytest.mark.parametrize("change", [dict(correlations="all radii correlated"), dict(tilt_pivot="back_vertex"),
                                      dict(sensor_mount="follows_last_element")])
def test_unsupported_profile_settings_fail_explicitly(change):
    payload = development_profile().model_dump()
    payload.update(change)
    with pytest.raises(ValueError):
        DevelopmentProfile.model_validate(payload)


def test_seeded_draw_repeatability_bounds_and_disjoint_phases(optical):
    screening = build_draw(optical.profile, optical.settings, "screening", 0)
    assert screening == build_draw(optical.profile, optical.settings, "screening", 0)
    evaluation = build_draw(optical.profile, optical.settings, "evaluation", 0)
    assert screening["stream_key"] != evaluation["stream_key"]
    assert screening["underlying_uniform_draws"] != evaluation["underlying_uniform_draws"]
    # The generator has no candidate argument: both starters share these draws.
    assert screening["errors"] == build_draw(optical.profile, optical.settings, "screening", 0)["errors"]
    names = {"radius_error": "radius_fraction", "thickness_error": "center_thickness_mm",
             "air_gap_error": "air_gap_mm", "decenter": "decenter_mm", "tilt": "tilt_deg"}
    for error in optical.profile.errors:
        array = np.asarray(screening["errors"][names[error.parameter]])
        assert np.all(array >= error.lower_bound) and np.all(array <= error.upper_bound)
    with pytest.raises(ValueError, match="disjoint"):
        OpticalSettings(screening_seed=1, evaluation_seed=1)


def test_measured_repeatability_cache_and_id_order_invariance(optical):
    grader = OpticalGrader(optical.store, optical.cache, optical.settings, optical.profile)
    first = grader.grade(optical.case, Submission(starter_ids=(3, 7)))
    assert all(not s["refinement"]["after"]["constraint_failures"] for s in first["starters"])
    assert all(s["refinement"]["work"]["objective_evaluations"] == 37 for s in first["starters"])
    assert first["evaluation"]["builds_tested"] == 3
    assert first["starters"][0]["screening"]["builds"][0]["draw"] == first["starters"][1]["screening"]["builds"][0]["draw"]
    second = grader.grade(optical.case, Submission(starter_ids=(3, 7)))
    assert first["result"] == second["result"]
    assert all(s["refinement"]["cache_hit"] for s in second["starters"])
    assert all(b["cache_hit"] and b["work_this_run"] == 0 for b in second["evaluation"]["builds"])
    payload = optical.case.model_dump()
    payload["candidates"] = list(reversed(payload["candidates"]))
    for c in payload["candidates"]:
        c["candidate_id"] += 100
    changed = BriefCase.model_validate(payload)
    remapped = grader.grade(changed, Submission(starter_ids=(107, 103)))
    assert remapped["result"]["shipped_prescription_ref"] == first["result"]["shipped_prescription_ref"]
    assert remapped["result"]["simulation"] == first["result"]["simulation"]
    assert remapped["selection"]["frozen_before_evaluation"]
    assert optical.case.parse_result(json.dumps(first["result"])) == GraderResult.model_validate(first["result"])
    env = SimpleNamespace(case=changed, tolerance_profile=optical.profile)
    assert grader(env, Submission(starter_ids=(107, 103))) == remapped["result"]["reward"]


def test_invalid_builds_count_in_denominator_software_errors_do_not(optical, monkeypatch):
    import lux_frontier.optical_grading as module

    p = optical.store.resolve(optical.case.candidates[0].prescription_ref)
    grader = OpticalGrader(optical.store, optical.cache, optical.settings, optical.profile)
    draw = build_draw(optical.profile, optical.settings, "evaluation", 300)
    draw["errors"]["center_thickness_mm"] = [-100, 0, 0]
    monkeypatch.setattr(module, "build_draw", lambda *args: draw)
    outcome = grader.simulate(p, optical.case.brief, "evaluation", 2)
    assert outcome["status"] == "completed"
    assert outcome["builds_passed"] == 0 and outcome["builds_tested"] == 2 and outcome["yield_fraction"] == 0
    assert all(b["status"] == "physically_invalid" for b in outcome["builds"])
    monkeypatch.setattr(module, "build_draw", lambda *args: dict(draw, index=999))
    monkeypatch.setattr(module, "measure", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("software defect")))
    failed = grader.simulate(p, optical.case.brief, "evaluation", 1)
    assert failed["status"] == "software_error" and "yield_fraction" not in failed
    assert failed["software_error"]["type"] == "RuntimeError"


def test_no_nominal_pass_never_runs_manufacturing(optical, monkeypatch):
    payload = optical.case.model_dump()
    for point in payload["brief"]["mtf_thresholds"]:
        point["minimum_mtf_fraction"] = 1
    case = BriefCase.model_validate(payload)
    settings = optical.settings.model_copy(update={"refinement_evaluations": 1})
    grader = OpticalGrader(optical.store, optical.cache, settings, optical.profile)
    monkeypatch.setattr(grader, "simulate", lambda *args: pytest.fail("nonpassing nominal design simulated"))
    run = grader.grade(case, Submission(starter_ids=(3, 7)))
    assert run["result"]["final_pass"] is False and run["result"]["reward"] == 0
    assert run["result"]["simulation"]["status"] == "not_run"
    assert run["result"]["shipped_prescription_ref"] is None


def test_constraint_validation_rejects_relabeled_mtf_points(optical):
    payload = optical.case.candidates[0].nominal_metrics.model_dump()
    payload["mtf"][0]["orientation"] = "tangential"
    with pytest.raises(ValueError, match="orientation mismatch"):
        constraint_failures(NominalMetrics.model_validate(payload), optical.case.brief)


def test_screening_tie_break_is_prescription_based_not_id_or_order(optical):
    class TiedGrader(OpticalGrader):
        def simulate(self, p, brief, phase, count):
            return dict(status="completed", builds_passed=count, builds_tested=count, yield_fraction=1.0, builds=[])

    grader = TiedGrader(optical.store, optical.cache, optical.settings, optical.profile)
    first = grader.grade(optical.case, Submission(starter_ids=(7, 3)))
    refs = [s["refinement"]["refined_prescription_ref"]["uri"] for s in first["starters"]]
    assert first["result"]["shipped_prescription_ref"]["uri"] == min(refs)
    second = grader.grade(optical.case, Submission(starter_ids=(3, 7)))
    assert first["result"]["shipped_prescription_ref"] == second["result"]["shipped_prescription_ref"]


def test_single_build_replay_from_recorded_prescription_and_draws(optical):
    grader = OpticalGrader(optical.store, optical.cache, optical.settings, optical.profile)
    run = grader.grade(optical.case, Submission(starter_ids=(3, 7)))
    run.update(settings=optical.settings.model_dump(mode="json"), profile=optical.profile.model_dump(mode="json"))
    source = optical.directory / "source-run.json"
    source.write_text(json.dumps(run))
    path, replay = replay_evaluation(source, 0, optical.directory / "replayed")
    assert path.is_file() and replay["exact_reproduction"]
    assert replay["build"]["draw"] == run["evaluation"]["builds"][0]["draw"]


def test_refinement_software_error_is_not_a_nominal_optical_failure(optical, monkeypatch):
    grader = OpticalGrader(optical.store, optical.cache, optical.settings, optical.profile)

    def broken(*args):
        raise RuntimeError("optimizer implementation defect")

    monkeypatch.setattr(grader, "refine", broken)
    run = grader.grade(optical.case, Submission(starter_ids=(3, 7)))
    assert run["result"]["simulation"]["status"] == "failed"
    assert run["result"]["reward"] is None and run["result"]["final_pass"] is None
    assert run["software_error"]["stage"] == "refinement"

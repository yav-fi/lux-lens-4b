import json
from pathlib import Path

import pytest

import lux_frontier.optical_batch as batch
from lux_frontier.contracts import Brief, NominalMetrics
from lux_frontier.optical_engine import cooke_prescription, geometry_summary
from lux_frontier.optical_prescriptions import development_profile, digest


@pytest.fixture(scope="module")
def recipe():
    return batch.make_manifest(cooke_prescription(), development_profile())


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    reference = tmp_path / "reference.json"
    reference.write_text('{"separate":"Cooke reference"}')
    monkeypatch.setattr(batch, "REFERENCE", reference)
    monkeypatch.setattr(batch, "CACHE", tmp_path / "cache")
    monkeypatch.setattr(batch, "FOCUS_CACHE", tmp_path / "focus-cache")
    monkeypatch.setattr(batch, "ARTIFACTS", tmp_path / "artifacts")
    batch.load_manifest.cache_clear()
    path = batch.freeze_batch(tmp_path / "batch")
    yield path
    batch.load_manifest.cache_clear()


def metrics(p, brief, **kwargs):
    return NominalMetrics(effective_focal_length_mm=50, f_number=brief.target_f_number,
        first_surface_to_sensor_mm=p.sensor_z_mm,
        mtf=[dict(t.model_dump(exclude={"minimum_mtf_fraction"}), mtf_fraction=.8) for t in brief.mtf_thresholds]), {}


def test_recipe_has_eight_meaningfully_distinct_supported_camera_tasks(recipe):
    assert len(recipe["briefs"]) == 8
    assert len({digest(e["brief"]) for e in recipe["briefs"]}) == 8
    assert len({e["brief"]["target_f_number"] for e in recipe["briefs"]}) >= 4
    assert len({max(t["field_angle_deg"][1] for t in e["brief"]["mtf_thresholds"]) for e in recipe["briefs"]}) >= 6
    assert {t["wavelength_um"] for e in recipe["briefs"] for t in e["brief"]["mtf_thresholds"]} == {.55, .65}
    assert "All 64 starters share" in recipe["diversity_limits"]
    assert all(e["use_case"] and len(e["requirement_rationale"]) == 8 for e in recipe["briefs"])


def test_all_exact_prescriptions_are_distinct_traceable_and_randomly_remapped(recipe):
    refs, mappings = set(), set()
    for entry in recipe["briefs"]:
        candidates = entry["candidates"]
        assert len(candidates) == 8
        assert sorted(c["candidate_id"] for c in candidates) == list(range(1, 9))
        assert [c["candidate_id"] for c in candidates] != list(range(1, 9))
        assert {c["lineage"] for c in candidates} == {v[0] for v in batch.VARIANTS}
        mapping = tuple(sorted((c["lineage"], c["candidate_id"]) for c in candidates))
        mappings.add(mapping)
        for c in candidates:
            assert c["parent_prescription_ref"] == recipe["parent_prescription_ref"]
            assert c["prescription_ref"]["uri"].endswith(digest(c["prescription"]))
            refs.add(c["prescription_ref"]["uri"])
    assert len(refs) == 64 and len(mappings) == 8


def test_shared_profile_workload_fixed_target_and_disjoint_matched_draws(recipe):
    keys = set()
    for entry in recipe["briefs"]:
        s = entry["settings"]
        assert [s[k] for k in ("refinement_evaluations", "screening_builds", "evaluation_builds", "refocus_positions", "pupil_samples")] == [37, 20, 100, 33, 64]
        assert entry["brief"]["minimum_simulated_yield_fraction"] == .9
        assert entry["brief"]["manufacturing_tolerance_profile_ref"] == recipe["profile"]["reference"]
        for phase, count in (("screening", 20), ("evaluation", 100)):
            assert len(entry["draw_streams"][phase]) == count
            for d in entry["draw_streams"][phase]:
                assert tuple(d["stream_key"]) not in keys
                keys.add(tuple(d["stream_key"]))
    assert len(keys) == 960
    assert set(batch.nested_focus_grid(.2, 9)) <= set(recipe["focus_positions_mm"])


def test_all_starters_fit_the_existing_common_optimizer_bounds(recipe):
    for entry in recipe["briefs"]:
        for c in entry["candidates"]:
            p = batch.Prescription.model_validate(c["prescription"])
            assert geometry_summary(p).minimum_edge_thickness_mm > 0
            values = (p.sensor_z_mm, p.spacings_mm[1], p.spacings_mm[3])
            assert all(lo <= x <= hi for x, (lo, hi) in zip(values, recipe["refinement"]["bounds_mm"], strict=True))
            assert p.stop_diameter_mm < 2 * p.clear_semidiameters_mm[1]


def test_freezing_is_non_grading_and_is_immutable(frozen, monkeypatch):
    monkeypatch.setattr(batch, "measure", lambda *a, **k: pytest.fail("freeze must not measure"))
    monkeypatch.setattr(batch.OpticalGrader, "refine", lambda *a, **k: pytest.fail("freeze must not refine"))
    original = frozen.read_bytes()
    assert batch.freeze_batch(frozen.parent) == frozen
    assert frozen.read_bytes() == original
    envelope = json.loads(original)
    assert envelope["manifest_sha256"] == digest(envelope["manifest"])
    assert not list(frozen.parent.glob("briefs/*/initial/*.json"))


def test_manifest_tampering_is_rejected_before_work(frozen):
    envelope = json.loads(frozen.read_text())
    envelope["manifest"]["briefs"][0]["brief"]["minimum_simulated_yield_fraction"] = .8
    frozen.write_text(json.dumps(envelope))
    batch.load_manifest.cache_clear()
    with pytest.raises(ValueError, match="modified frozen"):
        batch.load_manifest(frozen)


def test_checkpoint_survives_interruption_reuses_exact_work_and_checks_integrity(tmp_path):
    path, calls = tmp_path / "record.json", []
    def broken():
        raise RuntimeError("interrupted")
    with pytest.raises(RuntimeError):
        batch.checkpoint(path, "key", broken)
    assert not path.exists()
    def complete():
        calls.append(1)
        return {"passed": False, "draw": {"seed": 42}}
    first, reused = batch.checkpoint(path, "key", complete)
    assert not reused
    second, reused = batch.checkpoint(path, "key", complete)
    assert first == second and reused and calls == [1]
    with pytest.raises(ValueError, match="integrity"):
        batch.checkpoint(path, "different-settings", complete)
    data = json.loads(path.read_text())
    data["value"]["passed"] = True
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="integrity"):
        batch.checkpoint(path, "key", complete)


def test_prepare_can_resume_without_remeasuring_and_precommits_prompt_only_rule(frozen, monkeypatch):
    calls = []
    def measured(*args, **kwargs):
        assert frozen.is_file()
        calls.append(1)
        return metrics(*args, **kwargs)
    monkeypatch.setattr(batch, "measure", measured)
    result = batch.run_batch(frozen, workers=1, stop_after="prepare")
    assert result["stopped_after"] == "prepare" and len(calls) == 64
    batch.run_batch(frozen, workers=1, stop_after="prepare")
    assert len(calls) == 64
    m, _ = batch.load_manifest(frozen)
    for e in m["briefs"]:
        root = batch.case_root(frozen, e)
        precommit = json.loads((root / "heuristic-precommit.json").read_text())
        assert precommit["rule"] == batch.HEURISTIC_RULE
        assert len(precommit["selected_ids"]) == 2
        prompt = json.loads((root / "prompt.json").read_text())["prompt"]
        assert prompt.count("Candidate ID ") == 8
        assert "reward" not in prompt and "local://" not in prompt
        assert not (root / "selection-plan.json").exists()


def test_no_nominally_qualified_pair_is_recorded_as_a_failed_contract(recipe):
    entry = recipe["briefs"][0]
    b = Brief.model_validate(entry["brief"])
    candidates = []
    rows = []
    for c in entry["candidates"]:
        p = batch.Prescription.model_validate(c["prescription"])
        candidates.append(batch.Candidate(candidate_id=c["candidate_id"], prescription_ref=p.reference,
            layout="test Cooke", glass_types=("SK16", "F2", "SK16"), nominal_metrics=metrics(p, b)[0]))
        rows.append(dict(candidate_id=c["candidate_id"], nominal_pass=False,
                         refined_prescription_sha256=digest(p), screening=None))
    case = batch.BriefCase(brief=b, candidates=candidates)
    plan = batch.freeze_selectors(rows, [1, 2])
    for row in rows:
        row.update(evaluation=None, final_pass=False)
    result = batch.summarize_brief(rows, [1, 2], plan, case)
    assert result["pairs_meeting_target"] == 0
    assert result["hindsight_best_achievable_pair"]["shipped_yield"] == 0
    assert len(result["pairs"]) == 28
    assert all(p["failed_submission"] and p["grader_result"]["reward"] == 0 for p in result["pairs"])


def test_resume_record_is_bound_to_manifest_phase_and_build_index(recipe):
    entry = recipe["briefs"][0]
    key = batch.job_key("frozen", entry, 1, "screening", 0)
    assert key != batch.job_key("changed-manifest", entry, 1, "screening", 0)
    assert key != batch.job_key("frozen", entry, 1, "evaluation", 0)
    assert key != batch.job_key("frozen", entry, 1, "screening", 1)
    assert Path(batch.job_path(Path("outputs/batch/manifest.json"), entry, 1, "screening", 0)).suffix == ".json"


def test_hindsight_achievable_pair_respects_shipping_and_positive_result_contracts(recipe):
    entry = recipe["briefs"][0]
    b, rows, candidates = Brief.model_validate(entry["brief"]), [], []
    for c in entry["candidates"]:
        i = c["candidate_id"]
        p = batch.Prescription.model_validate(c["prescription"])
        nominal = metrics(p, b)[0]
        candidates.append(batch.Candidate(candidate_id=i, prescription_ref=p.reference, layout="test Cooke",
                                          glass_types=("SK16", "F2", "SK16"), nominal_metrics=nominal))
        rows.append(dict(candidate_id=i, nominal_pass=True, refined_prescription_sha256=digest(p),
                         screening=dict(builds_passed=i), refinement=dict(
                             refined_prescription_ref=p.reference.model_dump(mode="json"), after=dict(metrics=nominal.model_dump(mode="json")))))
    case = batch.BriefCase(brief=b, candidates=candidates)
    plan = batch.freeze_selectors(rows, [1, 2])
    for row in rows:
        n = 100 if row["candidate_id"] == 1 else 90 if row["candidate_id"] == 2 else 50
        row.update(final_pass=n >= 90, evaluation=dict(status="completed", builds_passed=n,
            builds_tested=100, yield_fraction=n / 100,
            builds=[dict(draw=dict(index=k), passed=k < n) for k in range(100)]))
    result = batch.summarize_brief(rows, [1, 2], plan, case)
    # Starter 1's perfect outcome is unreachable: every other starter beats its
    # screening count, so no valid two-starter submission can ship it.
    assert result["hindsight"]["highest_fresh_yield"] == 1
    assert result["hindsight_best_achievable_pair"]["shipped_yield"] == .9
    assert result["hindsight_best_achievable_pair"]["candidate_pairs"] == [[1, 2]]
    assert result["pairs_meeting_target"] == 1
    pair = next(p for p in result["pairs"] if p["candidate_ids"] == [1, 2])
    assert pair["grader_result"]["reward"] == .9
    assert pair["grader_result"]["final_pass"]


def test_exploratory_report_uses_exact_saved_prompt_blocks_only_after_outcomes():
    from lux_frontier.optical_batch_report import exploratory_example
    prompt = "Candidate ID 1:\nEFL: 50 mm\nMTF: 0.4\n\nCandidate ID 2:\nEFL: 49.9 mm\nMTF: 0.6\n\nCandidate ID 3:\nEFL: 50.1 mm\nMTF: 0.7\n\nOutput JSON"
    result = dict(brief_id="example", baselines=dict(closest_to_spec=dict(candidate_ids=[1, 2],
        shipped_candidate_id=1, reward=.8, final_pass=False)), pairs=[dict(candidate_ids=[2, 3],
        shipped_candidate_id=3, reward=.95, final_pass=True)])
    report = exploratory_example([dict(result=result, prompt=prompt)])
    assert report["exists"] and report["exploratory_not_causal"]
    assert report["heuristic_pair"] == [1, 2] and report["hindsight_pair"] == [2, 3]
    assert report["exact_model_visible_blocks"]["3"] == "Candidate ID 3:\nEFL: 50.1 mm\nMTF: 0.7"
    assert "Output JSON" not in report["exact_model_visible_blocks"]["3"]


def test_no_exploratory_example_is_invented_when_the_heuristic_already_passes():
    from lux_frontier.optical_batch_report import exploratory_example
    result = dict(baselines=dict(closest_to_spec=dict(final_pass=True)), pairs=[dict(final_pass=True)])
    assert not exploratory_example([dict(result=result, prompt="")])["exists"]

"""Post-evaluation audit and exploratory prompt-feature extraction; no grading."""

import argparse
import json
from fractions import Fraction
from pathlib import Path

from lux_frontier.contracts import BriefCase, NominalMetrics
from lux_frontier.optical_batch import DEFAULT_DIRECTORY, load_manifest, save_once
from lux_frontier.optical_diagnostic import sha_file
from lux_frontier.optical_engine import constraint_failures


def visible_block(prompt: str, candidate_id: int) -> str:
    heading = f"Candidate ID {candidate_id}:\n"
    if prompt.count(heading) != 1:
        raise ValueError("candidate must appear exactly once in the saved prompt")
    return heading + prompt.split(heading, 1)[1].split("\n\n", 1)[0]


def exploratory_example(records: list[dict]) -> dict:
    """Select the first qualifying brief in manifest order, only after grading."""
    for item in records:
        r = item["result"]
        heuristic = r["baselines"]["closest_to_spec"]
        winners = [p for p in r["pairs"] if p["final_pass"]]
        if heuristic["final_pass"] or not winners:
            continue
        # This is explicitly a hindsight example, never a deployable selector.
        chosen = min(winners, key=lambda p: (-p["reward"], p["candidate_ids"]))
        ids = sorted(set(heuristic["candidate_ids"] + chosen["candidate_ids"]))
        return dict(exists=True, exploratory_not_causal=True, brief_id=r["brief_id"],
                    heuristic_pair=heuristic["candidate_ids"], heuristic_shipped_id=heuristic["shipped_candidate_id"],
                    heuristic_yield=heuristic["reward"], hindsight_pair=chosen["candidate_ids"],
                    hindsight_shipped_id=chosen["shipped_candidate_id"], hindsight_yield=chosen["reward"],
                    exact_model_visible_blocks={str(i): visible_block(item["prompt"], i) for i in ids},
                    interpretation="These initial prompt-visible measurements and geometry summaries are hypotheses about useful features; this post-outcome comparison does not establish a causal predictor or an independent validation result.")
    return dict(exists=False, exploratory_not_causal=True,
                reason="No brief has a failing heuristic and a passing alternative pair; no contrasting feature example was manufactured.")


def audit_batch(manifest_path: Path) -> dict:
    load_manifest.cache_clear()
    m, key = load_manifest(manifest_path)
    root = manifest_path.parent
    result = json.loads((root / "run.json").read_text())
    if result["manifest_sha256"] != key:
        raise ValueError("batch result does not belong to this frozen manifest")
    records, initial, refinements, screens, evaluations, precommits, plans = [], [], [], [], [], [], []
    total_builds, physically_invalid, zero_nominal_ties, qualified_best = 0, 0, 0, 0
    stream_keys = set()
    for entry, summary in zip(m["briefs"], result["briefs"], strict=True):
        directory = root / "briefs" / entry["brief"]["brief_id"]
        case = BriefCase.model_validate_json((directory / "case.json").read_text())
        r = json.loads((directory / "run.json").read_text())
        assert r["brief_id"] == case.brief.brief_id == summary["brief_id"]
        assert r["manifest_sha256"] == key and len(r["starters"]) == 8 and len(r["pairs"]) == 28
        initial.extend(directory.glob("initial/*.json"))
        refinements.extend(directory.glob("refinement/*.json"))
        screens.extend(directory.glob("screening/*/*.json"))
        evaluations.extend(directory.glob("evaluation/*/*.json"))
        precommits.append(directory / "heuristic-precommit.json")
        plans.append(directory / "selection-plan.json")
        for phase in ("screening", "evaluation"):
            for d in entry["draw_streams"][phase]:
                assert tuple(d["stream_key"]) not in stream_keys
                stream_keys.add(tuple(d["stream_key"]))
        for row in r["starters"]:
            assert row["refinement"]["work"]["objective_evaluations"] == 37
            for phase, count in (("screening", 20), ("evaluation", 100)):
                if not row["nominal_pass"]:
                    assert row[phase] is None
                    continue
                builds = row[phase]["builds"]
                assert len(builds) == count
                assert [b["draw"] for b in builds] == entry["draw_streams"][phase]
                assert row[phase]["builds_passed"] == sum(b["passed"] for b in builds)
                for build in builds:
                    total_builds += 1
                    assert len(build["refocus_trials"]) == 33
                    assert sorted(t["sensor_shift_mm"] for t in build["refocus_trials"]) == sorted(m["focus_positions_mm"])
                    if build["status"] == "physically_invalid":
                        physically_invalid += 1
                        assert not build["passed"]
                    else:
                        measured = NominalMetrics.model_validate(build["metrics"])
                        assert constraint_failures(measured, case.brief) == build["failures"]
                        assert build["details"]["sensor_z_mm"] == measured.first_surface_to_sensor_mm
                    assert build["passed"] == (not build["failures"])
        for pair in r["pairs"]:
            parsed = case.parse_result(json.dumps(pair["grader_result"]))
            assert parsed.reward == pair["reward"] and parsed.final_pass == pair["final_pass"]
        expected = sum((Fraction(p["builds_passed"], 100) if not p["failed_submission"] else Fraction(0)
                        for p in r["pairs"]), Fraction()) / 28
        assert str(expected) == summary["exact_expected_random_yield"]
        best = summary["heuristic_yield"] == summary["highest_observed_achievable_pair_yield"]
        nominal_count = sum(row["nominal_pass"] for row in r["starters"])
        zero_nominal_ties += best and nominal_count == 0
        qualified_best += best and nominal_count > 0
        records.append(dict(result=r, prompt=json.loads((directory / "prompt.json").read_text())["prompt"]))
    assert len(initial) == len(refinements) == 64
    assert all(p.stat().st_mtime_ns > manifest_path.stat().st_mtime_ns for p in initial)
    assert max(p.stat().st_mtime_ns for p in precommits) < min(p.stat().st_mtime_ns for p in refinements)
    if screens:
        assert max(p.stat().st_mtime_ns for p in screens) < min(p.stat().st_mtime_ns for p in plans)
    if evaluations:
        assert max(p.stat().st_mtime_ns for p in plans) < min(p.stat().st_mtime_ns for p in evaluations)
    assert sha_file(Path(m["cooke_reference"]["path"])) == m["cooke_reference"]["sha256"]
    assert not list((root / "software-errors").glob("*.json"))
    sessions = sorted((json.loads(p.read_text()) for p in (root / "sessions").glob("*.json")), key=lambda s: s["started_utc"])
    report = dict(manifest_sha256=key, manifest_file_sha256=sha_file(manifest_path),
        counts=result["counts"], heuristic_best_including_failed_nominal_ties=result["counts"]["heuristic_achieves_highest_observed_yield"],
        all_nominal_failure_ties=zero_nominal_ties, heuristic_best_with_nominally_qualified_starters=qualified_best,
        briefs=result["briefs"], exploratory=exploratory_example(records),
        audit=dict(manifest_before_measurement=True, heuristics_before_refinement=True,
                   all_shipping_choices_before_any_evaluation=True, matched_disjoint_seed_streams=len(stream_keys),
                   initial_measurements=64, refinements=64, objective_evaluations=64 * 37,
                   screening_builds=len(screens), evaluation_builds=len(evaluations), physically_invalid_builds=physically_invalid,
                   manufacturing_builds_checked=total_builds, pairs_checked=224,
                   fixed_profile_and_33_position_protocol=True, cooke_reference_unchanged=True, software_errors=0),
        execution_sessions=sessions, final_study_test_set_accessed=False, tinker_calls=0)
    # Session metadata is kept in its own append-only records; an analysis can be
    # regenerated after resume without changing the immutable grading results.
    stable = dict(report, execution_sessions=[])
    save_once(root / "analysis.json", stable)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_DIRECTORY / "manifest.json")
    args = parser.parse_args()
    report = audit_batch(args.manifest)
    print(json.dumps(dict(record=str(args.manifest.parent / "analysis.json"), counts=report["counts"],
                          audit=report["audit"], exploratory=report["exploratory"]), indent=2))


if __name__ == "__main__":
    main()

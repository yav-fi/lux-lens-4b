"""Offline all-eight diagnostic on a preserved optical run; fixed acceptance target."""

import argparse
import hashlib
import json
import multiprocessing
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from fractions import Fraction
from itertools import combinations
from pathlib import Path
from uuid import uuid4

from lux_frontier.contracts import BriefCase
from lux_frontier.optical_engine import constraint_failures
from lux_frontier.optical_grading import OpticalCache, OpticalGrader, OpticalSettings, ScreenedCandidate, load_recorded_case, select_shipped
from lux_frontier.optical_prescriptions import ArtifactStore, DevelopmentProfile, Prescription, digest
from lux_frontier.prompting import render_prompt

DIAGNOSTIC_PROTOCOL = "all-eight-diagnostic-v1"
DEFAULT_SOURCE = Path("outputs/optical/20261123T170131Z-development-daea8868/run.json")
HEURISTIC_RULE = (
    "Use initial prompt-displayed numbers (six significant digits): "
    "(abs(EFL-target)/focal_tolerance)^2 + (abs(f_number-target)/f_number_tolerance)^2 "
    "+ max(0,length/max_length-1)^2 + mean((max(0,minimum_MTF-MTF)/minimum_MTF)^2). "
    "Choose the two lowest scores; ties use SHA-256 of the model-visible candidate block excluding its local ID. "
    "No refined metrics, screening, prescription references or fresh outcomes enter this rule."
)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_manifest(directory: Path) -> dict[str, str]:
    return {str(p): sha_file(p) for p in directory.rglob("*") if p.is_file()}


def load_source(source_path: Path):
    source, case, profile, settings = load_recorded_case(source_path)
    if (len(case.candidates), settings.refinement_evaluations, settings.screening_builds,
            settings.evaluation_builds, case.brief.minimum_simulated_yield_fraction) != (8, 37, 20, 100, 0.9):
        raise ValueError("this diagnostic requires eight starters, 37 refinements, 20/100 draws and the fixed 90% target")
    return source, case, profile, settings


def heuristic_ranking(case: BriefCase, profile: DevelopmentProfile) -> list[dict]:
    """Predeclared prompt-only score; uses exactly the renderer's numeric precision."""
    def visible(value):
        return float(f"{value:g}")
    b = case.brief
    prompt = render_prompt(case, profile, provenance="development_optical")
    rows = []
    for candidate in case.candidates:
        m = candidate.nominal_metrics
        focal = abs(visible(m.effective_focal_length_mm) - visible(b.target_focal_length_mm)) / visible(b.focal_length_tolerance_mm)
        fno = abs(visible(m.f_number) - visible(b.target_f_number)) / visible(b.f_number_tolerance)
        track = max(0, visible(m.first_surface_to_sensor_mm) / visible(b.max_first_surface_to_sensor_mm) - 1)
        mtf = sum((max(0, visible(t.minimum_mtf_fraction) - visible(p.mtf_fraction)) /
                   visible(t.minimum_mtf_fraction)) ** 2 for p, t in zip(m.mtf, b.mtf_thresholds, strict=True)) / len(m.mtf)
        block = prompt.split(f"Candidate ID {candidate.candidate_id}:\n", 1)[1].split("\n\n", 1)[0]
        rows.append(dict(candidate_id=candidate.candidate_id, score=focal ** 2 + fno ** 2 + track ** 2 + mtf,
                         tie_break_sha256=hashlib.sha256(block.encode()).hexdigest(),
                         terms=dict(focal_squared=focal ** 2, f_number_squared=fno ** 2,
                                    relative_length_excess_squared=track ** 2, mean_relative_mtf_shortfall_squared=mtf)))
    return sorted(rows, key=lambda r: (r["score"], r["tie_break_sha256"]))


def evaluate_starter(payload: dict) -> dict:
    """Worker processes isolate Optiland's mutable backend state across candidates."""
    started = time.perf_counter()
    case = BriefCase.model_validate(payload["case"])
    profile = DevelopmentProfile.model_validate(payload["profile"])
    settings = OpticalSettings.model_validate(payload["settings"])
    candidate = next(c for c in case.candidates if c.candidate_id == payload["candidate_id"])
    store, cache = ArtifactStore(Path(payload["artifact_directory"])), OpticalCache(Path(payload["cache_directory"]))
    grader = OpticalGrader(store, cache, settings, profile)
    p = store.resolve(candidate.prescription_ref)
    path = Path(payload["record_path"])
    row = dict(candidate_id=candidate.candidate_id, initial_prescription_ref=candidate.prescription_ref.model_dump(mode="json"),
               status="refining", initial_metrics=candidate.nominal_metrics.model_dump(mode="json"),
               initial_failures=constraint_failures(candidate.nominal_metrics, case.brief))
    write_json(path, row)
    refinement = grader.refine(p, case.brief)
    row["refinement"] = refinement
    row["nominal_failures"] = refinement["after"].get("constraint_failures", [{"constraint": "physically_invalid"}])
    row["nominal_pass"] = "metrics" in refinement["after"] and not row["nominal_failures"]
    row["refined_prescription_sha256"] = digest(refinement["refined_prescription"])
    if row["nominal_pass"]:
        design = Prescription.model_validate(refinement["refined_prescription"])
        row["status"] = "screening"
        write_json(path, row)
        row["screening"] = grader.simulate(design, case.brief, "screening", settings.screening_builds)
        if row["screening"]["status"] != "completed":
            raise RuntimeError(f"software error screening candidate {candidate.candidate_id}: {row['screening']}")
        row["status"] = "evaluation"
        write_json(path, row)
        row["evaluation"] = grader.simulate(design, case.brief, "evaluation", settings.evaluation_builds)
        if row["evaluation"]["status"] != "completed":
            raise RuntimeError(f"software error evaluating candidate {candidate.candidate_id}: {row['evaluation']}")
        row["final_pass"] = row["evaluation"]["yield_fraction"] >= case.brief.minimum_simulated_yield_fraction
    else:
        row["screening"] = row["evaluation"] = None
        row["final_pass"] = False
    row.update(status="completed", elapsed_seconds=time.perf_counter() - started,
               cache_hits=dict(cache.hits), cache_misses=dict(cache.misses))
    write_json(path, row)
    return row


def selection_inputs(rows: list[dict]) -> list[ScreenedCandidate]:
    # This projection deliberately removes every fresh evaluation value.
    return [ScreenedCandidate(r["candidate_id"], r["nominal_pass"],
                              r["screening"]["builds_passed"] if r["nominal_pass"] else None,
                              r["refined_prescription_sha256"]) for r in rows]


def freeze_choice(inputs: list[ScreenedCandidate]) -> dict:
    winner = select_shipped(inputs)
    return dict(candidate_ids=sorted(c.candidate_id for c in inputs),
                shipped_candidate_id=winner.candidate_id if winner else None,
                refined_prescription_sha256=winner.refined_prescription_sha256 if winner else None,
                frozen_using_only_nominal_and_screening=True)


def score_frozen(choice: dict, evaluations: dict[int, dict], target: float) -> dict:
    shipped = choice["shipped_candidate_id"]
    if shipped is None:
        return dict(choice, failed_submission=True, shipped_yield=0.0, reward=0.0, final_pass=False,
                    failure="neither selected starter qualifies nominally")
    evaluation = evaluations[shipped]
    if evaluation["status"] != "completed":
        raise ValueError("fresh evaluation is incomplete")
    return dict(choice, failed_submission=False, shipped_yield=evaluation["yield_fraction"],
                reward=evaluation["yield_fraction"], final_pass=evaluation["yield_fraction"] >= target,
                builds_passed=evaluation["builds_passed"], builds_tested=evaluation["builds_tested"])


def verify_common_draws(rows: list[dict], settings: OpticalSettings) -> dict:
    passing = [r for r in rows if r["nominal_pass"]]
    streams = {}
    for phase, count, seed in [("screening", settings.screening_builds, settings.screening_seed),
                               ("evaluation", settings.evaluation_builds, settings.evaluation_seed)]:
        reference = None
        for row in passing:
            draws = [b["draw"] for b in row[phase]["builds"]]
            if len(draws) != count or any(d["phase"] != phase or d["seed"] != seed or d["index"] != i
                                          for i, d in enumerate(draws)):
                raise ValueError("candidate used incorrect draw count, phase, seed or order")
            if reference is not None and draws != reference:
                raise ValueError("manufacturing draws differ across candidates")
            reference = draws
        keys = {tuple(d["stream_key"]) for d in reference or []}
        if reference is not None and len(keys) != count:
            raise ValueError("manufacturing streams repeat within a phase")
        streams[phase] = keys
    if streams["screening"] & streams["evaluation"]:
        raise ValueError("screening and fresh evaluation draws overlap")
    return dict(common_across_candidates=True, disjoint_phases=True,
                screening_streams=len(streams["screening"]), evaluation_streams=len(streams["evaluation"]),
                candidate_ids=sorted(r["candidate_id"] for r in passing))


def summarize(rows: list[dict], heuristic_ids: list[int], target: float) -> dict:
    inputs = selection_inputs(rows)
    # Freeze ALL pair and baseline choices before joining ANY fresh outcomes.
    pairs = [freeze_choice(list(pair)) for pair in combinations(sorted(inputs, key=lambda c: c.candidate_id), 2)]
    heuristic = freeze_choice([c for c in inputs if c.candidate_id in heuristic_ids])
    try_all = freeze_choice(inputs)
    evaluations = {r["candidate_id"]: r["evaluation"] for r in rows if r["nominal_pass"]}
    scored_pairs = [score_frozen(p, evaluations, target) for p in pairs]
    yields = [Fraction(p["builds_passed"], p["builds_tested"]) if not p["failed_submission"] else Fraction(0) for p in scored_pairs]
    expected = sum(yields, Fraction(0)) / len(pairs)
    probability = Fraction(sum(p["final_pass"] for p in scored_pairs), len(pairs))
    hindsight = max(evaluations.values(), key=lambda e: e["yield_fraction"])["yield_fraction"] if evaluations else None
    paired = Counter()
    ordering = None
    if 3 in evaluations and 7 in evaluations:
        for a, b in zip(evaluations[3]["builds"], evaluations[7]["builds"], strict=True):
            if a["draw"] != b["draw"]:
                raise ValueError("paired candidate draws are not identical")
            label = "both_passed" if a["passed"] and b["passed"] else "only_3_passed" if a["passed"] else "only_7_passed" if b["passed"] else "neither_passed"
            paired[label] += 1
        by_id = {r["candidate_id"]: r for r in rows}
        screen_difference = by_id[7]["screening"]["builds_passed"] - by_id[3]["screening"]["builds_passed"]
        fresh_difference = evaluations[7]["builds_passed"] - evaluations[3]["builds_passed"]
        ordering = dict(screening_pass_difference_7_minus_3=screen_difference,
                        fresh_pass_difference_7_minus_3=fresh_difference,
                        screening_lead_predicted_strict_fresh_order=screen_difference * fresh_difference > 0)
    return dict(pairs=scored_pairs,
                baselines=dict(uniform_random_pair=dict(expected_shipped_yield=float(expected),
                    exact_expected_shipped_yield=str(expected), probability_of_passing=float(probability),
                    exact_probability_of_passing=str(probability), pair_count=len(pairs)),
                    closest_to_spec=score_frozen(heuristic, evaluations, target),
                    try_all_eight=score_frozen(try_all, evaluations, target)),
                hindsight=dict(deployable_selector=False, highest_fresh_yield=hindsight,
                               candidate_ids=sorted(k for k, v in evaluations.items() if v["yield_fraction"] == hindsight)),
                paired_3_7={key: paired[key] for key in ("both_passed", "only_3_passed", "only_7_passed", "neither_passed")},
                ordering_3_7=ordering,
                starters_meeting_target=sum(r["final_pass"] for r in rows),
                distinct_shipped_rewards=sorted(set(p["reward"] for p in scored_pairs)),
                shipped_reward_counts=dict(Counter(str(p["reward"]) for p in scored_pairs)))


def run_diagnostic(source_path: Path, output_root: Path, workers: int = 4,
                   precommit_path: Path | None = None) -> tuple[Path, dict]:
    started = time.perf_counter()
    source, case, profile, settings = load_source(source_path)
    manifest = source_manifest(source_path.parent)
    directory = output_root / f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    directory.mkdir(parents=True)
    ranking = heuristic_ranking(case, profile)
    heuristic = dict(protocol=DIAGNOSTIC_PROTOCOL, rule=HEURISTIC_RULE, ranking=ranking,
                     selected_ids=[r["candidate_id"] for r in ranking[:2]],
                     fixed_before_new_evaluations=True, source_record_sha256=sha_file(source_path))
    if precommit_path:
        original_precommit = json.loads(precommit_path.read_text())
        if original_precommit["ranking"] != ranking or original_precommit["selected_ids"] != heuristic["selected_ids"]:
            raise ValueError("heuristic does not match the original pre-evaluation precommit")
        heuristic["original_precommit_verified"] = dict(path=str(precommit_path), sha256=sha_file(precommit_path))
    write_json(directory / "heuristic-precommit.json", heuristic)
    print(json.dumps(dict(heuristic_precommit=str(directory / "heuristic-precommit.json"), selected_ids=heuristic["selected_ids"])), flush=True)
    store, cache = ArtifactStore(Path(source["artifact_directory"])), OpticalCache(Path(source["cache_directory"]))
    grader = OpticalGrader(store, cache, settings, profile)
    reproduced = grader.grade(case, case.parse_submission('{"starter_ids":[3,7]}'))
    if reproduced["result"] != source["result"] or reproduced["result"]["simulation"]["builds_passed"] != 51:
        raise RuntimeError("starter 7's preserved 51/100 result did not reproduce")
    write_json(directory / "reproduction.json", reproduced)
    print(json.dumps(dict(reproduced_starter_7="51/100", reward=reproduced["result"]["reward"])), flush=True)
    payloads = [dict(case=case.model_dump(mode="json"), profile=profile.model_dump(mode="json"), settings=settings.model_dump(mode="json"),
                     candidate_id=c.candidate_id, artifact_directory=source["artifact_directory"], cache_directory=source["cache_directory"],
                     record_path=str(directory / "starters" / f"{c.candidate_id}.json")) for c in sorted(case.candidates, key=lambda c: c.candidate_id)]
    rows = []
    if workers == 1:
        for payload in payloads:
            rows.append(evaluate_starter(payload))
            print(json.dumps(dict(completed_candidate=rows[-1]["candidate_id"])), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as executor:
            for future in as_completed([executor.submit(evaluate_starter, payload) for payload in payloads]):
                rows.append(future.result())
                print(json.dumps(dict(completed_candidate=rows[-1]["candidate_id"])), flush=True)
    rows.sort(key=lambda r: r["candidate_id"])
    draw_verification = verify_common_draws(rows, settings)
    result = summarize(rows, heuristic["selected_ids"], 0.9)
    result["draw_verification"] = draw_verification
    from lux_frontier.training import validate_task
    from lux_frontier.training_tasks import load_optical_task

    task = load_optical_task(source_path)
    validation = validate_task(task, '{"starter_ids":[3,7]}', directory / "training-interface")
    received = validation.data["groups"][0]["completions"][0]["reward"]
    if received != 0.51 or validation.data["optical_grader_result"]["final_pass"] is not False:
        raise RuntimeError("offline training interface failed to preserve the optical yield reward")
    result["training_interface"] = dict(record=str(validation.path), received_scalar_reward=received,
                                        final_pass=False, target=0.9, tinker_calls=0, reward_policy_changed=False)
    # Every originally present record/cache file must remain byte-for-byte intact.
    changed = [p for p, expected in manifest.items() if not Path(p).exists() or sha_file(Path(p)) != expected]
    if changed:
        raise RuntimeError(f"preserved source records changed: {changed}")
    result.update(protocol=DIAGNOSTIC_PROTOCOL, source_record=str(source_path), source_record_sha256=sha_file(source_path),
                  original_files_verified_unchanged=len(manifest), target_yield=0.9, settings=settings.model_dump(mode="json"),
                  profile=profile.model_dump(mode="json"), brief=case.model_dump(mode="json"), heuristic_precommit=heuristic,
                  reproduction=dict(matches_original_result=True, shipped_candidate_id=7, builds_passed=51, builds_tested=100, reward=0.51),
                  starters=rows, workers=workers, elapsed_seconds=time.perf_counter() - started,
                  tinker_calls=0, notes="All selection uses nominal/screening data only. Fresh yields are diagnostic scores, not selection inputs.")
    path = directory / "run.json"
    write_json(path, result)
    print(json.dumps(dict(record=str(path), baselines=result["baselines"], hindsight=result["hindsight"],
                          paired_3_7=result["paired_3_7"], elapsed_seconds=result["elapsed_seconds"]), indent=2), flush=True)
    return path, result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-record", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/optical_diagnostics"))
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=4)
    parser.add_argument("--precommit", type=Path, help="verify the ranking against a prior pre-evaluation heuristic record")
    args = parser.parse_args()
    if not args.output_dir.resolve().is_relative_to((Path.cwd() / "outputs").resolve()):
        parser.error("diagnostic records must be inside ignored outputs/")
    run_diagnostic(args.source_record, args.output_dir, args.workers, args.precommit)


if __name__ == "__main__":
    main()

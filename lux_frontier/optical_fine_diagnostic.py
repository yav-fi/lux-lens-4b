"""All-eight diagnostic under the frozen nested 33-position focus protocol."""

import json
import multiprocessing
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from uuid import uuid4

from lux_frontier.optical_diagnostic import freeze_choice, heuristic_ranking, selection_inputs, sha_file, source_manifest, summarize, verify_common_draws, write_json
from lux_frontier.optical_grading import OpticalCache, build_draw, load_recorded_case
from lux_frontier.optical_prescriptions import Prescription, digest
from lux_frontier.optical_sensitivity import build_summary, evaluate_focus, nested_focus_grid

PROTOCOL = "all-eight-frozen-refocus33-v1"
PAIRED_DEVELOPMENT_SEED = 2026120102  # distinct from previous confirmation 2026120101
FOCUS_CACHE = Path("outputs/optical_sensitivity/cache")
FROZEN_FOCUS_PLAN = Path("outputs/optical_sensitivity/20261123T203833Z-confirmation-0b3373b2/confirmation-plan.json")


@lru_cache(maxsize=4)
def load_coarse(path: Path):
    coarse = json.loads(path.read_text())
    source, case, profile, settings = load_recorded_case(Path(coarse["source_record"]))
    if (len(case.candidates), settings.refocus_positions, settings.refinement_evaluations,
            settings.screening_builds, settings.evaluation_builds,
            case.brief.minimum_simulated_yield_fraction) != (8, 9, 37, 20, 100, 0.9):
        raise ValueError("requires the completed original eight-starter development diagnostic")
    if coarse["brief"] != case.model_dump(mode="json") or coarse["profile"] != profile.model_dump(mode="json"):
        raise ValueError("completed diagnostic does not match its immutable case/profile")
    rows = {r["candidate_id"]: r for r in coarse["starters"]}
    if set(rows) != {c.candidate_id for c in case.candidates}:
        raise ValueError("completed diagnostic is missing a starter")
    for row in rows.values():
        if row["refinement"]["work"]["objective_evaluations"] != 37:
            raise ValueError("refinement budget differs from the frozen protocol")
        if row["refined_prescription_sha256"] != digest(row["refinement"]["refined_prescription"]):
            raise ValueError("refined prescription fingerprint mismatch")
    ranking = heuristic_ranking(case, profile)
    if ranking != coarse["heuristic_precommit"]["ranking"]:
        raise ValueError("heuristic differs from the original precommit")
    frozen = json.loads(FROZEN_FOCUS_PLAN.read_text())
    if (frozen["alternative_focus_positions"] != nested_focus_grid(0.2, 33)
            or frozen["original_focus_positions"] != nested_focus_grid(0.2, 9)
            or frozen["requirements_sha256"] != digest(case.brief)
            or frozen["profile_sha256"] != digest(profile)):
        raise ValueError("focus grid, requirements or profile differ from the frozen 33-position protocol")
    return coarse, source, case, profile, settings, rows


def freeze_selectors(rows: list[dict], heuristic_ids: list[int]) -> dict:
    """Accept screening-only rows; intentionally require absent evaluation keys."""
    if any("evaluation" in r or "final_pass" in r for r in rows):
        raise ValueError("selection rows must not contain evaluation outcomes")
    inputs = selection_inputs(rows)
    return dict(selection_inputs=[asdict(c) for c in inputs],
                pairs=[freeze_choice(list(pair)) for pair in combinations(sorted(inputs, key=lambda c: c.candidate_id), 2)],
                heuristic=freeze_choice([c for c in inputs if c.candidate_id in heuristic_ids]),
                try_all_eight=freeze_choice(inputs))


def unique_confirmation_designs(plan: dict) -> dict[str, int]:
    """Deduplicate physical designs, including selectors that ship the same lens."""
    result = {}
    for selector in ("heuristic", "try_all_eight"):
        choice = plan[selector]
        if choice["shipped_candidate_id"] is not None:
            result[choice["refined_prescription_sha256"]] = choice["shipped_candidate_id"]
    return result


def paired_selector_summary(heuristic: list[dict], try_all: list[dict]) -> dict:
    counts = Counter()
    if len(heuristic) != 100 or len(try_all) != 100:
        raise ValueError("paired comparison requires exactly 100 fresh development draws")
    for a, b in zip(heuristic, try_all, strict=True):
        if a["draw"] != b["draw"]:
            raise ValueError("selectors used different paired development errors")
        label = "both_passed" if a["passed"] and b["passed"] else "heuristic_only_passed" if a["passed"] else "try_all_only_passed" if b["passed"] else "neither_passed"
        counts[label] += 1
    a_pass, b_pass = sum(b["passed"] for b in heuristic), sum(b["passed"] for b in try_all)
    return dict(counts={k: counts[k] for k in ("both_passed", "heuristic_only_passed", "try_all_only_passed", "neither_passed")},
                heuristic_passes=a_pass, try_all_eight_passes=b_pass,
                heuristic_yield=a_pass / 100, try_all_eight_yield=b_pass / 100,
                try_all_minus_heuristic_percentage_points=b_pass - a_pass,
                heuristic_reaches_90=a_pass >= 90, try_all_eight_reaches_90=b_pass >= 90)


def fine_worker(payload: dict) -> dict:
    _, _, case, profile, settings, rows = load_coarse(Path(payload["coarse_diagnostic"]))
    row = rows[payload["candidate_id"]]
    p = Prescription.model_validate(row["refinement"]["refined_prescription"])
    phase, index = payload["phase"], payload["index"]
    if phase == "new_development":
        settings = settings.model_copy(update={"evaluation_seed": PAIRED_DEVELOPMENT_SEED})
        draw = build_draw(profile, settings, "evaluation", index)
        saved = None
    else:
        saved = row[phase]["builds"][index]
        draw = saved["draw"]
    value = evaluate_focus(p, case.brief, profile, settings, draw, nested_focus_grid(0.2, 33),
                           OpticalCache(Path(payload["focus_cache"])), saved_build=saved)
    write_json(Path(payload["record_path"]), value)
    return dict(candidate_id=payload["candidate_id"], phase=phase, index=index,
                passed=value["passed"], new_metric_evaluations=value["work"]["new_metric_evaluations"])


def run_jobs(payloads: list[dict], workers: int):
    if workers == 1:
        for payload in payloads:
            yield fine_worker(payload)
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as executor:
            for future in as_completed([executor.submit(fine_worker, p) for p in payloads]):
                yield future.result()


def aggregate(builds: list[dict]) -> dict:
    summary = build_summary(builds)
    return dict(summary, status="completed", yield_fraction=summary["builds_passed"] / summary["builds_tested"], builds=builds)


def run_fine_diagnostic(coarse_path: Path, output_root: Path, workers: int) -> tuple[Path, dict]:
    started = time.perf_counter()
    coarse, source, case, profile, settings, old_rows = load_coarse(coarse_path)
    manifest = {}
    for root in (Path(source["cache_directory"]).parent, coarse_path.parent, FOCUS_CACHE.parent):
        manifest.update(source_manifest(root))
    directory = output_root / f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-refocus33-{uuid4().hex[:8]}"
    directory.mkdir(parents=True)
    write_json(directory / "preserved-manifest.json", manifest)
    original_precommit = coarse["heuristic_precommit"]
    heuristic_ids = original_precommit["selected_ids"]
    write_json(directory / "heuristic-precommit.json", original_precommit)
    write_json(directory / "protocol-plan.json", dict(protocol=PROTOCOL, coarse_diagnostic=str(coarse_path),
               source_sha256=sha_file(coarse_path), requirements=case.brief.model_dump(mode="json"),
               profile=profile.model_dump(mode="json"),
               settings=settings.model_copy(update={"refocus_positions": 33}).model_dump(mode="json"),
               source_cache_settings=settings.model_dump(mode="json"),
               original_focus_positions=nested_focus_grid(0.2, 9), frozen_focus_positions=nested_focus_grid(0.2, 33),
               frozen_focus_source=str(FROZEN_FOCUS_PLAN), frozen_focus_source_sha256=sha_file(FROZEN_FOCUS_PLAN),
               refinement_policy="reuse identical original 37-evaluation refinements; no new optimization",
               new_development_seed=PAIRED_DEVELOPMENT_SEED,
               prior_development_seeds=[settings.screening_seed, settings.evaluation_seed, 2026120101],
               final_study_test_set_accessed=False))

    def payload(candidate_id, phase, index, path):
        return dict(coarse_diagnostic=str(coarse_path), focus_cache=str(FOCUS_CACHE), candidate_id=candidate_id,
                    phase=phase, index=index, record_path=str(path))

    screening_payloads = [payload(i, "screening", k, directory / "starters" / str(i) / "screening" / f"{k:03d}.json")
                          for i, row in sorted(old_rows.items()) if row["nominal_pass"] for k in range(20)]
    progress, work = Counter(), Counter()
    for item in run_jobs(screening_payloads, workers):
        progress[item["candidate_id"]] += 1
        work["screening_new_metric_evaluations"] += item["new_metric_evaluations"]
        if progress[item["candidate_id"]] == 20:
            print(json.dumps(dict(screening_completed_candidate=item["candidate_id"])), flush=True)
    rows = []
    for i, old in sorted(old_rows.items()):
        screening = aggregate([json.loads((directory / "starters" / str(i) / "screening" / f"{k:03d}.json").read_text())
                               for k in range(20)]) if old["nominal_pass"] else None
        rows.append(dict(candidate_id=i, nominal_pass=old["nominal_pass"], screening=screening,
                         refined_prescription_sha256=old["refined_prescription_sha256"]))
    # No 33-position evaluation work is scheduled until ALL screening finishes
    # and every deployable/pair choice has been written to this frozen plan.
    plan = freeze_selectors(rows, heuristic_ids)
    plan.update(protocol=PROTOCOL, frozen_before_evaluation=True, frozen_before_new_development_draws=True,
                target_yield=0.9, new_development_seed=PAIRED_DEVELOPMENT_SEED,
                source_heuristic_precommit_sha256=digest(original_precommit),
                frozen_focus_positions=nested_focus_grid(0.2, 33))
    write_json(directory / "selection-plan.json", plan)
    print(json.dumps(dict(selection_plan=str(directory / "selection-plan.json"), heuristic=plan["heuristic"],
                          try_all_eight=plan["try_all_eight"]), indent=2), flush=True)
    unique = unique_confirmation_designs(plan)
    if not unique:
        raise ValueError("neither baseline qualifies nominally")
    confirm_settings = settings.model_copy(update={"evaluation_seed": PAIRED_DEVELOPMENT_SEED})
    prior_seeds = {settings.screening_seed, settings.evaluation_seed, 2026120101}
    if PAIRED_DEVELOPMENT_SEED in prior_seeds:
        raise RuntimeError("new paired development seed was previously used")
    draws = [build_draw(profile, confirm_settings, "evaluation", k) for k in range(100)]
    write_json(directory / "paired-development-draws.json", dict(role="new_development_selector_comparison", draws=draws))
    # Candidate 7 replays are cached and run first, giving an early 76/100 check.
    order = sorted(old_rows, key=lambda i: (i != 7, i))
    evaluations = [payload(i, "evaluation", k, directory / "starters" / str(i) / "evaluation" / f"{k:03d}.json")
                   for i in order if old_rows[i]["nominal_pass"] for k in range(100)]
    new_draw_jobs = [payload(i, "new_development", k, directory / "new-development" / sha / f"{k:03d}.json")
                     for sha, i in unique.items() for k in range(100)]
    progress.clear()
    for item in run_jobs(evaluations + new_draw_jobs, workers):
        key = (item["candidate_id"], item["phase"])
        progress[key] += 1
        work[item["phase"] + "_new_metric_evaluations"] += item["new_metric_evaluations"]
        if progress[key] % 20 == 0:
            print(json.dumps(dict(candidate_id=key[0], phase=key[1], completed=progress[key])), flush=True)
    for row in rows:
        i = row["candidate_id"]
        row["evaluation"] = aggregate([json.loads((directory / "starters" / str(i) / "evaluation" / f"{k:03d}.json").read_text())
                                        for k in range(100)]) if row["nominal_pass"] else None
        row["final_pass"] = row["nominal_pass"] and row["evaluation"]["builds_passed"] >= 90
        row.update(refinement=old_rows[i]["refinement"], initial_metrics=old_rows[i]["initial_metrics"],
                   nominal_failures=old_rows[i]["nominal_failures"],
                   nine_position_yield=old_rows[i]["evaluation"]["yield_fraction"] if row["nominal_pass"] else None)
    by_id = {r["candidate_id"]: r for r in rows}
    if by_id[7]["evaluation"]["builds_passed"] != 76:
        raise RuntimeError("starter 7 did not reproduce the frozen-grid 76/100 result")
    result = summarize(rows, heuristic_ids, 0.9)
    if [p["shipped_candidate_id"] for p in plan["pairs"]] != [p["shipped_candidate_id"] for p in result["pairs"]]:
        raise RuntimeError("evaluation altered a frozen pair shipping choice")
    if (plan["heuristic"]["shipped_candidate_id"] != result["baselines"]["closest_to_spec"]["shipped_candidate_id"]
            or plan["try_all_eight"]["shipped_candidate_id"] != result["baselines"]["try_all_eight"]["shipped_candidate_id"]):
        raise RuntimeError("evaluation altered a frozen baseline shipping choice")
    result["draw_verification"] = verify_common_draws(rows, settings)
    evaluated = {sha: [json.loads((directory / "new-development" / sha / f"{k:03d}.json").read_text()) for k in range(100)]
                 for sha in unique}
    for builds in evaluated.values():
        if [b["draw"] for b in builds] != json.loads(json.dumps(draws)):
            raise RuntimeError("new paired development errors differ from the frozen stream")
    a_sha, b_sha = plan["heuristic"]["refined_prescription_sha256"], plan["try_all_eight"]["refined_prescription_sha256"]
    paired = paired_selector_summary(evaluated[a_sha], evaluated[b_sha])
    paired.update(seed=PAIRED_DEVELOPMENT_SEED, role="new_development_selector_comparison",
                  heuristic_shipped_id=plan["heuristic"]["shipped_candidate_id"],
                  try_all_eight_shipped_id=plan["try_all_eight"]["shipped_candidate_id"],
                  unique_physical_designs_evaluated=len(unique), build_evaluations=100 * len(unique),
                  same_design=a_sha == b_sha, final_study_test_set_accessed=False)
    if a_sha == b_sha and paired["try_all_minus_heuristic_percentage_points"] != 0:
        raise RuntimeError("identical designs received different paired outcomes")
    best = max(p["reward"] for p in result["pairs"])
    heuristic_reward = result["baselines"]["closest_to_spec"]["reward"]
    opportunity = dict(hindsight_only=True, best_shipped_pair_yield=best,
                       best_pair_ids=[p["candidate_ids"] for p in result["pairs"] if p["reward"] == best],
                       heuristic_shipped_yield=heuristic_reward,
                       a_pair_can_beat_heuristic=best > heuristic_reward,
                       maximum_observed_gain_percentage_points=round(100 * (best - heuristic_reward), 10))
    old_pair_choices = {tuple(p["candidate_ids"]): p["shipped_candidate_id"] for p in coarse["pairs"]}
    changes = dict(heuristic=dict(original9=coarse["baselines"]["closest_to_spec"]["shipped_candidate_id"],
                                  frozen33=plan["heuristic"]["shipped_candidate_id"]),
                   try_all_eight=dict(original9=coarse["baselines"]["try_all_eight"]["shipped_candidate_id"],
                                      frozen33=plan["try_all_eight"]["shipped_candidate_id"]),
                   pairs_with_changed_ship=sum(old_pair_choices[tuple(p["candidate_ids"])] != p["shipped_candidate_id"] for p in result["pairs"]))
    changed_files = [path for path, sha in manifest.items() if not Path(path).is_file() or sha_file(Path(path)) != sha]
    if changed_files:
        raise RuntimeError(f"preserved records changed: {changed_files}")
    result.update(protocol=PROTOCOL, coarse_diagnostic=str(coarse_path), source_record=coarse["source_record"],
                  source_sha256=sha_file(coarse_path), original_files_verified_unchanged=len(manifest),
                  settings=settings.model_copy(update={"refocus_positions": 33}).model_dump(mode="json"),
                  source_cache_settings=settings.model_dump(mode="json"), frozen_focus_positions=nested_focus_grid(0.2, 33),
                  requirements=case.brief.model_dump(mode="json"), profile=profile.model_dump(mode="json"),
                  target_yield=0.9, refinement_policy="identical original 37-evaluation refinements reused",
                  original_precommitted_heuristic=original_precommit, selection_plan=plan,
                  starters=rows, starter7_reproduced_76=True, new_development_comparison=paired,
                  shipping_changes=changes, opportunity=opportunity, optical_work=dict(work),
                  final_study_test_set_accessed=False, tinker_calls=0, workers=workers,
                  elapsed_seconds=time.perf_counter() - started)
    path = directory / "run.json"
    write_json(path, result)
    print(json.dumps(dict(record=str(path), baselines=result["baselines"], hindsight=result["hindsight"],
                          shipping_changes=changes, opportunity=opportunity, new_development_comparison=paired,
                          elapsed_seconds=result["elapsed_seconds"]), indent=2), flush=True)
    return path, result

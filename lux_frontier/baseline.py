"""Offline export and replay of the first frozen eight-brief Qwen baseline.

The packaged versioned result and compact score cache suffice on a fresh checkout.
Export can rebuild those artifacts from the original ignored run and optical batch;
it neither samples a model nor runs an optical simulation.
"""

import argparse
from fractions import Fraction
import hashlib
from itertools import combinations
import json
from pathlib import Path

from lux_frontier.contracts import BriefCase, NominalMetrics
from lux_frontier.optical_batch import initial_case, load_manifest, pair_contract
from lux_frontier.optical_diagnostic import (freeze_choice, heuristic_ranking, score_frozen,
                                           selection_inputs, sha_file)
from lux_frontier.optical_engine import constraint_failures
from lux_frontier.optical_prescriptions import DevelopmentProfile
from lux_frontier.prompting import render_prompt
from lux_frontier.synthetic_training import LensEnvironment, grade_response

RESEARCH = Path(__file__).with_name("research")
RESULT_NAME = "development-eight-v1-qwen35-4b-v1.json"
CACHE_NAME = "development-eight-v1-replay-v1.json"
DEFAULT_RESULT = RESEARCH / RESULT_NAME
ORIGINAL_RUN = Path("outputs/baselines/development-eight-v1-qwen35-4b-first")
DEFAULT_MANIFEST = Path("outputs/optical_batches/development-eight-v1/manifest.json")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def save_once(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(read(path) == value, f"versioned artifact differs: {path}")
        return
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def score(case, profile, pairs, response, *, clean=True):
    """Original strict ingress and cached reward lookup, including truncation policy."""
    by_pair = {tuple(p["candidate_ids"]): p for p in pairs}

    def cached_grader(environment, submission):
        return by_pair[tuple(sorted(submission.starter_ids))]["reward"]

    grade = grade_response(LensEnvironment(case, profile), response,
                           truncated=not clean, grader=cached_grader)
    if grade.error:
        return dict(valid=False, selected_ids=grade.parsed_choices, shipped_candidate_id=None,
                    reward=0., final_pass=False, error=grade.error)
    pair = by_pair[tuple(sorted(grade.parsed_choices))]
    return dict(valid=True, selected_ids=list(grade.parsed_choices),
                shipped_candidate_id=pair["shipped_candidate_id"], reward=grade.reward,
                final_pass=pair["final_pass"], error=None)


def export_result(source_run=ORIGINAL_RUN, manifest_path=DEFAULT_MANIFEST, output_dir=RESEARCH):
    """Project original caches into a small, portable, versioned research artifact."""
    manifest, key = load_manifest(manifest_path)
    frozen, live = read(source_run / "frozen-inputs.json"), read(source_run / "run.json")
    require(live["status"] == "complete" and live["completed_briefs"] == 8, "source baseline is incomplete")
    require(key == frozen["manifest_sha256"] == live["manifest_sha256"], "batch identity mismatch")
    require(sha_file(manifest_path) == frozen["manifest_file_sha256"], "manifest bytes changed")
    profile = DevelopmentProfile.model_validate(manifest["profile"])
    cache = dict(schema_version=1, kind="frozen_optical_replay_cache", batch_id="development-eight-v1",
                 manifest_sha256=key, manifest_file_sha256=sha_file(manifest_path),
                 profile=manifest["profile"], briefs=[])
    result = dict(schema_version=1, result_id="development-eight-v1-qwen35-4b-first-v1",
                  batch_id=cache["batch_id"], manifest_sha256=key,
                  manifest_file_sha256=cache["manifest_file_sha256"],
                  source_run=str(source_run), source_run_sha256=sha_file(source_run / "run.json"),
                  frozen_inputs_sha256=sha_file(source_run / "frozen-inputs.json"),
                  model_id=frozen["model_id"], renderer=frozen["renderer"],
                  thinking_enabled=frozen["thinking_enabled"], direct_base_model=frozen["direct_base_model"],
                  decoding=frozen["decoding"], dependency_versions=frozen["versions"],
                  original_source_sha256=frozen["source_sha256"],
                  token_usage=live["token_usage"], estimated_cost_usd=live["estimated_cost_usd"],
                  expected_summary=live["summary"], briefs=[])
    for entry, prompt_info, observed in zip(manifest["briefs"], frozen["prompts"], live["briefs"], strict=True):
        brief_id = entry["brief"]["brief_id"]
        root = manifest_path.parent / "briefs" / brief_id
        case = BriefCase.model_validate(read(root / "case.json"))
        require(case == initial_case(manifest_path, manifest, key, entry), "candidate/prescription mapping changed")
        prompt = render_prompt(case, profile, provenance="development_optical")
        require(prompt == prompt_info["prompt"] == read(root / "prompt.json")["prompt"], "prompt changed")
        require(hashlib.sha256(prompt.encode()).hexdigest() == prompt_info["prompt_sha256"], "prompt hash mismatch")
        source = read(root / "run.json")
        starters = []
        for row in source["starters"]:
            compact = {k: row[k] for k in ("candidate_id", "lineage", "nominal_pass", "nominal_failures",
                                            "refined_prescription_sha256")}
            compact["refinement"] = dict(refined_prescription_ref=row["refinement"]["refined_prescription_ref"],
                after={k: row["refinement"]["after"][k] for k in ("metrics", "constraint_failures")
                       if k in row["refinement"]["after"]})
            for phase in ("screening", "evaluation"):
                compact[phase] = ({k: row[phase][k] for k in
                    ("status", "builds_passed", "builds_tested", "yield_fraction")} if row[phase] else None)
            starters.append(compact)
        cache["briefs"].append(dict(brief_id=brief_id, case=case.model_dump(mode="json"),
            prompt_sha256=prompt_info["prompt_sha256"], starters=starters,
            heuristic_precommit=read(root / "heuristic-precommit.json"),
            pairs=[{k:v for k,v in p.items() if k != "grader_result"} for p in source["pairs"]],
            source_sha256={name:sha_file(root / name) for name in
                           ("case.json", "prompt.json", "run.json", "heuristic-precommit.json", "selection-plan.json")}))
        completion = read(source_run / "completions" / (brief_id + ".json"))
        raw = read(source_run / "raw" / (brief_id + ".json"))
        require(len(raw["sequences"]) == 1, "source has multiple completions")
        require(all(completion[k] == v for k,v in raw["sequences"][0].items()), "first response changed")
        require(completion["raw_response"] == observed["raw_answer"], "raw answer mismatch")
        shipped = observed["qwen"]["shipped_starter"]
        design = next((s for s in source["starters"] if s["candidate_id"] == shipped), None)
        result["briefs"].append(dict(brief_id=brief_id, name=observed["name"],
            prompt_sha256=prompt_info["prompt_sha256"], prompt_tokens=prompt_info["prompt_tokens"],
            prompt_token_ids_sha256=prompt_info["token_ids_json_sha256"],
            completion=completion, selected_ids=observed["qwen"]["selected_ids"],
            shipped_candidate_id=shipped,
            shipped_design=None if design is None else dict(lineage=design["lineage"],
                prescription_ref=design["refinement"]["refined_prescription_ref"],
                prescription_sha256=design["refined_prescription_sha256"]),
            reward=observed["qwen"]["reward"], final_pass=observed["qwen"]["final_pass"],
            heuristic_reward=observed["heuristic"]["reward"]))
    save_once(output_dir / CACHE_NAME, cache)
    result["replay_cache"] = dict(file=CACHE_NAME, sha256=sha_file(output_dir / CACHE_NAME))
    save_once(output_dir / RESULT_NAME, result)
    return replay(output_dir / RESULT_NAME)


def load_result(path=DEFAULT_RESULT):
    result = read(path)
    require(result["schema_version"] == 1, "unsupported research result version")
    cache_path = path.parent / result["replay_cache"]["file"]
    require(sha_file(cache_path) == result["replay_cache"]["sha256"], "replay cache hash mismatch")
    cache = read(cache_path)
    require(cache["schema_version"] == 1 and cache["batch_id"] == result["batch_id"] and
            cache["manifest_sha256"] == result["manifest_sha256"], "replay cache identity mismatch")
    require(len(cache["briefs"]) == len(result["briefs"]) == 8, "expected eight frozen briefs")
    return result, cache


def validated_pairs(item, profile):
    """Replay the actual nominal/screening shipping rule before reading evaluation counts."""
    case = BriefCase.model_validate(item["case"])
    prompt = render_prompt(case, profile, provenance="development_optical")
    require(hashlib.sha256(prompt.encode()).hexdigest() == item["prompt_sha256"], "rendered prompt hash mismatch")
    precommit = item["heuristic_precommit"]
    ranking = heuristic_ranking(case, profile)
    require(ranking == precommit["ranking"] and
            [r["candidate_id"] for r in ranking[:2]] == precommit["selected_ids"], "heuristic precommit changed")
    by_id = {s["candidate_id"]:s for s in item["starters"]}
    require(set(by_id) == {c.candidate_id for c in case.candidates}, "cached candidate IDs differ")
    for row in by_id.values():
        after = row["refinement"]["after"]
        failures = constraint_failures(NominalMetrics.model_validate(after["metrics"]), case.brief)
        require(failures == row["nominal_failures"] and row["nominal_pass"] == (not failures), "nominal qualification changed")
        for phase, count in (("screening", 20), ("evaluation", 100)):
            measured = row[phase]
            if row["nominal_pass"]:
                require(measured["status"] == "completed" and measured["builds_tested"] == count and
                        0 <= measured["builds_passed"] <= count and
                        measured["yield_fraction"] == measured["builds_passed"] / count, "invalid cached build counts")
            else:
                require(measured is None, "unqualified starter has cached builds")
    inputs = selection_inputs(item["starters"])
    evaluations = {s["candidate_id"]:s["evaluation"] for s in item["starters"] if s["nominal_pass"]}
    choices = [freeze_choice(list(pair)) for pair in combinations(sorted(inputs, key=lambda c:c.candidate_id), 2)]
    require(len(item["pairs"]) == len(choices) == 28, "incomplete pair cache")
    for saved, choice in zip(item["pairs"], choices, strict=True):
        actual = score_frozen(choice, evaluations, case.brief.minimum_simulated_yield_fraction)
        require(actual == saved, "cached pair differs from frozen screening/scoring rule")
        contract = pair_contract(case, actual, by_id)
        require(contract["reward"] == saved["reward"] and contract["final_pass"] == saved["final_pass"], "graded contract differs")
    return case, prompt


def replay(path=DEFAULT_RESULT):
    result, cache = load_result(path)
    profile = DevelopmentProfile.model_validate(cache["profile"])
    rows, heuristic_rewards, random_rewards, random_passes = [], [], [], []
    for observed, item in zip(result["briefs"], cache["briefs"], strict=True):
        case, _ = validated_pairs(item, profile)
        require(observed["brief_id"] == item["brief_id"] and observed["prompt_sha256"] == item["prompt_sha256"], "brief identity mismatch")
        completion = observed["completion"]
        require(len(completion["tokens"]) <= result["decoding"]["max_tokens"], "completion exceeds frozen limit")
        require(completion["raw_response"] == completion["response_text"] + "<|im_end|>", "saved chat framing differs")
        actual = score(case, profile, item["pairs"], completion["response_text"],
                       clean=completion["clean"] and completion["stop_reason"] == "stop")
        for key in ("selected_ids", "shipped_candidate_id", "reward", "final_pass"):
            require(actual[key] == observed[key], f"saved result mismatch: {case.brief.brief_id} {key}")
        if actual["shipped_candidate_id"] is not None:
            design = next(s for s in item["starters"] if s["candidate_id"] == actual["shipped_candidate_id"])
            require(observed["shipped_design"]["prescription_ref"] == design["refinement"]["refined_prescription_ref"] and
                    observed["shipped_design"]["prescription_sha256"] == design["refined_prescription_sha256"], "shipped design differs")
        heuristic = score(case, profile, item["pairs"], json.dumps(dict(starter_ids=item["heuristic_precommit"]["selected_ids"])))
        require(heuristic["reward"] == observed["heuristic_reward"], "heuristic reward differs")
        rows.append(dict(brief_id=item["brief_id"], name=observed["name"], **actual))
        heuristic_rewards.append(heuristic["reward"])
        random_rewards.append(sum((Fraction(p["builds_passed"], p["builds_tested"]) if not p["failed_submission"] else Fraction()
                                   for p in item["pairs"]), Fraction()) / 28)
        random_passes.append(Fraction(sum(p["final_pass"] for p in item["pairs"]), 28))
    summary = dict(valid_answers=sum(r["valid"] for r in rows), passes=sum(r["final_pass"] for r in rows),
                   denominator=8, mean_reward=sum(r["reward"] for r in rows) / 8,
                   heuristic_mean_reward=sum(heuristic_rewards) / 8,
                   exact_random_mean_reward=str(sum(random_rewards, Fraction()) / 8),
                   exact_random_expected_passes=str(sum(random_passes, Fraction())))
    expected = result["expected_summary"]["qwen"]
    require((summary["valid_answers"], summary["passes"], summary["mean_reward"]) ==
            (expected["valid_answers"], expected["briefs_reaching_90_of_100"], expected["mean_reward_all_eight"]), "baseline summary mismatch")
    return dict(result_id=result["result_id"], batch_id=result["batch_id"], manifest_sha256=result["manifest_sha256"],
                offline_only=True, tinker_calls=0, new_optical_simulations=0, pairs_verified=224, summary=summary, briefs=rows)


def conveyor_comparison(path=DEFAULT_RESULT):
    replay(path)
    result, cache = load_result(path)
    item = next(b for b in cache["briefs"] if b["brief_id"].endswith("conveyor-vision"))
    observed = next(b for b in result["briefs"] if b["brief_id"] == item["brief_id"])
    qwen_ids = set(observed["selected_ids"])
    best_reward = max(p["reward"] for p in item["pairs"])
    better = min((p for p in item["pairs"] if p["reward"] == best_reward),
                 key=lambda p:(-len(qwen_ids & set(p["candidate_ids"])), p["candidate_ids"]))
    relevant = sorted(qwen_ids | set(better["candidate_ids"]))
    prompt = render_prompt(BriefCase.model_validate(item["case"]), DevelopmentProfile.model_validate(cache["profile"]), provenance="development_optical")
    by_id = {s["candidate_id"]:s for s in item["starters"]}
    return dict(diagnostic_only=True, qwen_pair=observed["selected_ids"], comparison_pair=better["candidate_ids"],
        substitution=dict(removed=sorted(qwen_ids - set(better["candidate_ids"])),
                          added=sorted(set(better["candidate_ids"]) - qwen_ids)),
        requirements=item["case"]["brief"],
        prompt_visible_blocks={str(i):f"Candidate ID {i}:\n" + prompt.split(f"Candidate ID {i}:\n", 1)[1].split("\n\n", 1)[0] for i in relevant},
        post_grading={str(i):by_id[i] for i in relevant},
        qwen_score=next(p for p in item["pairs"] if set(p["candidate_ids"]) == qwen_ids), comparison_score=better)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("replay", "export", "conveyor"))
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--source-run", type=Path, default=ORIGINAL_RUN)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=RESEARCH)
    args = parser.parse_args()
    if args.command == "export":
        report = export_result(args.source_run, args.manifest, args.output_dir)
    elif args.command == "conveyor":
        report = conveyor_comparison(args.result)
    else:
        report = replay(args.result)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

"""Command line for the V2 optical benchmark and experiments."""

import argparse
import asyncio
from dataclasses import asdict
import json
from pathlib import Path
import statistics

from lux_frontier.v2.benchmark import (
    Protocol, REPRESENTATIONS, generate, load_envelope, load_tasks, render, write_envelope, write_json,
)
from lux_frontier.v2.grading import MODES, PHASES, grade, submit


def selectors(parser, *, many=False):
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--split", choices=("train", "development", "test"), default="development")
    parser.add_argument("--distribution", choices=("representative", "capability"), default="representative")
    parser.add_argument("--task", action="append" if many else "store")


def prediction_files(values):
    files = {}
    for option in values or []:
        label, filename = option.split("=", 1)
        path = Path(filename)
        raw = json.loads(path.read_text())
        if "run" in raw:
            raw = load_envelope(path, "run")
        if label in files:
            raise ValueError(f"duplicate prediction label {label}")
        files[label] = raw
    return files


def model_arguments(parser):
    parser.add_argument("--settings", type=Path, help="JSON settings overrides")
    parser.add_argument("--execute", action="store_true", help="execute paid API calls; default prints an offline cost plan")
    parser.add_argument("--output", type=Path, default=Path("outputs/v2-runs/run.json"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser("generate", help="generate and exhaustively grade briefs")
    make.add_argument("output", type=Path)
    make.add_argument("--workers", type=int, default=1)
    make.add_argument("--scenarios", type=int, default=7)
    make.add_argument("--pools", type=int, default=1)
    make.add_argument("--profiles", nargs="+")
    make.add_argument("--contrast-parent", type=Path, help="append controlled train-only contrasts to a frozen pilot")
    for key, field in Protocol.model_fields.items():
        make.add_argument("--"+key.replace("_", "-"), type=int, default=field.default)
    validate = commands.add_parser("validate", help="check reference optics or cached submissions")
    validate.add_argument("--manifest", type=Path)
    validate.add_argument("--task")
    validate.add_argument("--response")
    validate.add_argument("--phase", choices=PHASES, default="development")
    validate.add_argument("--mode", choices=MODES, default="yield")
    val = commands.add_parser("study", help="measure budget and focus convergence on a task")
    selectors(val)
    val.add_argument("--builds", type=int, default=16)
    val.add_argument("--output", type=Path, required=True)
    prompt = commands.add_parser("prompt", help="print one frozen model prompt")
    selectors(prompt)
    prompt.add_argument("--representation", choices=REPRESENTATIONS, default="physics")
    count = commands.add_parser("tokenize", help="count exact Qwen prompt tokens without Tinker calls")
    selectors(count, many=True)
    count.add_argument("--output", type=Path)
    ev = commands.add_parser("evaluate", help="matched offline evaluation of cached physical outcomes")
    selectors(ev, many=True)
    ev.add_argument("--predictions", action="append", metavar="LABEL=FILE")
    ev.add_argument("--paired-reference", default="closest_to_spec")
    ev.add_argument("--output", type=Path, required=True)
    audit = commands.add_parser("audit", help="summarize task construction, difficulty and measured reversals")
    audit.add_argument("manifest", type=Path)
    audit.add_argument("--output", type=Path)
    for name in ("explore", "sample"):
        opt = commands.add_parser(name, help="sample Qwen or a checkpoint with a bounded workload")
        selectors(opt, many=True)
        model_arguments(opt)
        opt.add_argument("--checkpoint")
        opt.add_argument("--temperatures", nargs="+", type=float, default=(.6, 1., 1.35))
    train = commands.add_parser("train", help="bounded Tinker GRPO, one-step by default")
    train.add_argument("manifest", type=Path)
    model_arguments(train)
    train.add_argument("--resume", type=Path)
    train.add_argument("--reward", choices=MODES, help="override the training reward mode in settings")
    replay = commands.add_parser("replay", help="export measured ray paths and optical history")
    selectors(replay)
    replay.add_argument("--predictions", action="append", metavar="LABEL=FILE")
    replay.add_argument("--output", type=Path, required=True)
    plot = commands.add_parser("plot", help="render a local replay as a figure")
    plot.add_argument("replay", type=Path)
    plot.add_argument("--output", type=Path, required=True)
    plot.add_argument("--candidate", type=int)
    harbor = commands.add_parser("harbor", help="export Harbor task folders")
    selectors(harbor, many=True)
    harbor.add_argument("--output", type=Path, required=True)
    harbor.add_argument("--representation", choices=REPRESENTATIONS, default="physics")
    args = parser.parse_args()
    command = args.command
    phase = {"train": "training", "development": "development", "test": "final"}.get(getattr(args, "split", ""))

    def choose():
        tasks = load_tasks(args.manifest, split=args.split, distribution=args.distribution)
        if args.task:
            desired = args.task if isinstance(args.task, list) else [args.task]
            tasks = [t for t in tasks if t["task_id"] in desired]
            if set(desired) != {t["task_id"] for t in tasks}:
                raise ValueError("a requested task is missing from the selected split or distribution")
        if not tasks:
            raise ValueError("empty task selection")
        return tasks

    if command == "generate":
        protocol = Protocol(**{name: getattr(args, name) for name in Protocol.model_fields})
        result = generate(args.output, protocol, workers=args.workers, scenarios=args.scenarios, pools=args.pools,
                          profiles=args.profiles, contrast_parent=args.contrast_parent)
        result = dict(manifest=str(args.output/"manifest.json"), tasks=len(result["task_index"]),
                      physical_rejections=len(result["rejection_details"]), runtime_seconds=result["runtime_seconds"])
    elif command == "validate":
        from lux_frontier.v2.validation import validate_references
        if args.manifest is None:
            result = validate_references()
        else:
            tasks = load_tasks(args.manifest)
            if args.task:
                tasks = [t for t in tasks if t["task_id"] == args.task]
            if not tasks:
                raise ValueError("requested task not in manifest")
            result = {t["task_id"]: [asdict(grade(t, response, phase=args.phase, mode=args.mode))
                for response in ([args.response] if args.response is not None else [submit(c["id"]) for c in t["candidates"]])]
                for t in tasks}
    elif command == "study":
        from lux_frontier.v2.validation import study
        if not 2 <= args.builds <= 500:
            raise ValueError("study builds must be in 2–500")
        result = study(choose()[0], builds=args.builds)
        write_json(args.output, result)
        result = dict(output=str(args.output), runtime_seconds=result["runtime_seconds"])
    elif command == "prompt":
        print(render(choose()[0], args.representation))
        return
    elif command == "tokenize":
        from lux_frontier.tinker_backend import MODEL, RENDERER, load_renderer
        from lux_frontier.v2.rl import RATES
        _, renderer = load_renderer()
        tasks = choose()
        sizes = {rep: {t["task_id"]: len(renderer.build_generation_prompt([
            {"role": "user", "content": render(t, rep)}]).to_ints()) for t in tasks} for rep in REPRESENTATIONS}
        result = dict(model=MODEL, renderer=RENDERER, tasks=len(tasks), representations={rep:
                      dict(minimum=min(counts.values()), mean=statistics.mean(counts.values()), maximum=max(counts.values()),
                           uncached_prefill_usd_per_1000=round(statistics.mean(counts.values())*RATES["prefill"]/1000, 6),
                           tokens_by_task=counts) for rep, counts in sizes.items()})
        if args.output:
            write_json(args.output, result)
    elif command == "evaluate":
        from lux_frontier.v2.evaluation import evaluate
        result = evaluate(choose(), phase=phase, train_tasks=load_tasks(args.manifest, split="train"),
                          predictions=prediction_files(args.predictions), paired_reference=args.paired_reference)
        write_envelope(args.output, result, "evaluation")
        result = dict(output=str(args.output), overall=result["overall"])
    elif command == "audit":
        from lux_frontier.v2.evaluation import audit
        result = audit(load_tasks(args.manifest))
        if args.output:
            write_envelope(args.output, result, "audit")
    elif command in ("train", "explore", "sample"):
        from lux_frontier.v2.rl import Settings, estimate, inference, train, training_plan
        settings = Settings.model_validate_json(args.settings.read_text()) if args.settings else Settings()
        if command == "train":
            if args.reward:
                settings = settings.model_copy(update={"mode": args.reward})
            training, development = (load_tasks(args.manifest, split=s) for s in ("train", "development"))
            result = training_plan(settings, len(development))
            if args.execute:
                result = asyncio.run(train(settings, training, development, args.output, resume=args.resume))
        else:
            tasks = choose()
            explore = command == "explore"
            result = estimate(settings, groups=len(tasks)*len(REPRESENTATIONS)*2*len(args.temperatures) if explore else 0,
                              other_samples=0 if explore else len(tasks)*settings.eval_samples)
            if args.execute:
                result = asyncio.run(inference(settings, tasks, args.output, exploration=explore,
                                     checkpoint=args.checkpoint, temperatures=args.temperatures))
        if args.execute:
            print(json.dumps(dict(output=str(args.output), status=result["status"],
                                  estimated_token_spend_usd=result["estimated_token_spend_usd"]), indent=2))
            if result["status"] != "completed":
                raise SystemExit(1)
            return
        result["fits_configured_limits"] = (result["budget_estimate_usd"] <= settings.budget_usd
                                           and result["maximum_sampled_completions"] <= settings.max_completions)
    elif command == "replay":
        from lux_frontier.v2.artifacts import export_replay
        data = export_replay(choose()[0], args.manifest, args.output, phase=phase,
                             model_records=prediction_files(args.predictions))
        result = dict(task_id=data["task_id"], output=str(args.output), candidates=len(data["candidates"]))
    elif command == "plot":
        from lux_frontier.v2.artifacts import plot_replay
        plot_replay(args.replay, args.output, args.candidate)
        result = dict(output=str(args.output))
    elif command == "harbor":
        from lux_frontier.v2.artifacts import export_harbor
        exported = export_harbor(choose(), args.output, phase=phase, representation=args.representation)
        result = dict(output=str(args.output), exported=len(exported))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

"""`agentcheck` command line.

Exit codes for `compare`: 0 no regression detected, 1 regression, 2 usage or input error.
"""

import argparse
import sys
from collections.abc import Sequence

from inspect_ai.log import EvalLog, read_eval_log
from inspect_ai.model import get_model

from agentcheck import calibrate, judge, power, summary
from agentcheck.compare import Verdict, compare_logs, merged_records, render
from agentcheck.run import load_env
from agentcheck.stats.sequential import look_alphas


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="agentcheck", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    c = sub.add_parser("compare", help="baseline vs candidate: regression, improvement or not")
    c.add_argument("baseline", help="baseline .eval log")
    c.add_argument("candidate", help="candidate .eval log")
    c.add_argument("--alpha", type=float, default=0.05, help="significance level (default 0.05)")
    c.add_argument("--fdr", type=float, default=0.05, help="per-category FDR (default 0.05)")
    c.add_argument("--resamples", type=int, default=10_000, help="bootstrap resamples")
    c.add_argument("--seed", type=int, default=0, help="bootstrap seed")
    c.add_argument(
        "--baseline-also", action="append", default=[], help="more baseline logs (epochs pooled)"
    )
    c.add_argument("--candidate-also", action="append", default=[], help="more candidate logs")
    c.add_argument("--looks", type=int, default=1, help="planned sequential looks (default 1)")
    c.add_argument("--look", type=int, default=1, help="which look this is (1..--looks)")
    c.add_argument(
        "--min-effect",
        type=float,
        default=0.10,
        help="drop that 'more runs' advice is sized for (default 0.10)",
    )

    s = sub.add_parser("summary", help="per-category summary of one log")
    s.add_argument("log", nargs="?")
    s.add_argument("--log-dir", default="logs")

    k = sub.add_parser("calibrate", help="false-alarm and detection rates by resampling a pool")
    k.add_argument("logs", nargs="*", help="pool: .eval logs of one agent (>= 2 epochs/sample)")
    k.add_argument("--synthetic", action="store_true", help="use a synthetic pool instead")
    k.add_argument("--mean", type=float, default=0.8, help="synthetic pool mean pass rate")
    k.add_argument("--spread", type=float, default=4.0, help="synthetic Beta concentration")
    k.add_argument("--samples", type=int, default=60, help="synthetic pool samples")
    k.add_argument("--epochs", type=int, default=1, help="epochs per simulated run")
    k.add_argument("--drop", type=float, default=0.10, help="injected pass-rate drop")
    k.add_argument("--reps", type=int, default=1000, help="simulated comparisons")
    k.add_argument("--alpha", type=float, default=0.05)
    k.add_argument("--resamples", type=int, default=2000, help="bootstrap resamples per test")
    k.add_argument("--seed", type=int, default=0)
    k.add_argument("--looks", type=int, default=1, help="sequential looks (must divide --epochs)")

    w = sub.add_parser("power", help="detection rate over a grid of samples x epochs")
    w.add_argument("logs", nargs="*", help="pool: .eval logs of one agent (>= 2 epochs/sample)")
    w.add_argument("--synthetic", action="store_true", help="use a synthetic pool instead")
    w.add_argument("--mean", type=float, default=0.8)
    w.add_argument("--spread", type=float, default=4.0)
    w.add_argument("--samples", default="60,120,240", help="comma-separated sample counts")
    w.add_argument("--epochs", default="1,3,5,10", help="comma-separated epoch counts")
    w.add_argument("--drop", type=float, default=0.10)
    w.add_argument("--target", type=float, default=0.80)
    w.add_argument("--reps", type=int, default=300)
    w.add_argument("--alpha", type=float, default=0.05)
    w.add_argument("--resamples", type=int, default=2000)
    w.add_argument("--seed", type=int, default=0)

    j = sub.add_parser("judge", help="calibrate an LLM judge against the deterministic scorer")
    j.add_argument("log", help=".eval log whose replies the judge grades")
    j.add_argument("--other", help="a second log of the same samples, for the position test")
    j.add_argument("--model", required=True, help="judge model, e.g. groq/openai/gpt-oss-20b")
    j.add_argument("--max-calls", type=int, required=True, help="hard cap on judge calls")
    j.add_argument("--max-tokens", type=int, required=True, help="hard cap on total tokens")

    args = p.parse_args(argv)
    if args.command == "judge":
        return run_judge(args)
    if args.command == "power":
        return run_power(args)
    if args.command == "calibrate":
        return run_calibrate(args)
    if args.command == "summary":
        forwarded = [args.log] if args.log else []
        return summary.main([*forwarded, "--log-dir", args.log_dir])

    if not 1 <= args.look <= args.looks:
        print(f"error: --look must be in 1..{args.looks}", file=sys.stderr)
        return 2
    bases = [read_eval_log(p) for p in [args.baseline, *args.baseline_also]]
    cands = [read_eval_log(p) for p in [args.candidate, *args.candidate_also]]
    base, cand = bases[0], cands[0]
    look_alpha = look_alphas(args.looks, args.alpha)[args.look - 1]
    try:
        result = compare_logs(
            bases,
            cands,
            alpha=look_alpha,
            fdr=args.fdr,
            n_resamples=args.resamples,
            seed=args.seed,
        )
    except ValueError as ex:
        print(f"error: {ex}", file=sys.stderr)
        return 2
    if args.looks > 1:
        print(
            f"Sequential look {args.look} of {args.looks}: tested at alpha {look_alpha:.4g} "
            f"(O'Brien-Fleming spending of {args.alpha} overall)."
        )
    print(render(result, base, cand))
    if result.verdict is Verdict.INCONCLUSIVE:
        if args.look < args.looks:
            print(
                f"Continue: run the next batch of epochs for both agents, then rerun with "
                f"--look {args.look + 1} and every log so far (--baseline-also/--candidate-also)."
            )
        else:
            print(more_runs_advice(bases, args.min_effect, args.seed))
    return 1 if result.verdict is Verdict.REGRESSION else 0


def more_runs_advice(baselines: list[EvalLog], min_effect: float, seed: int) -> str:
    records = merged_records(baselines)
    epochs = min(len(r.passes) for r in records.values())
    if epochs < 2:
        return (
            "More runs: the baseline has 1 epoch per sample, so its per-sample noise is "
            "unknown. Rerun it with --epochs 2 or more to get an estimate."
        )
    pool = {k: r.passes for k, r in records.items()}
    try:
        advice = power.epochs_needed(pool, epochs, drop=min_effect, seed=seed)
    except ValueError as ex:
        return f"More runs: no estimate ({ex})."
    return power.render_advice(advice, min_effect)


def load_pool(args: argparse.Namespace, n_samples: int = 60) -> tuple[calibrate.Run, str]:
    if args.synthetic == bool(args.logs):
        raise ValueError("give pool logs or --synthetic, not both or neither")
    if args.synthetic:
        pool = calibrate.synthetic_pool(
            n_samples=n_samples, mean=args.mean, spread=args.spread, seed=args.seed
        )
        return pool, f"synthetic (mean {args.mean}, spread {args.spread}, seed {args.seed})"
    logs = [read_eval_log(path) for path in args.logs]
    return calibrate.pool_from_logs(logs), f"{logs[0].eval.model}, {len(logs)} log(s)"


def run_judge(args: argparse.Namespace) -> int:
    log = read_eval_log(args.log)
    other = read_eval_log(args.other) if args.other else None
    try:
        graded_items, pairs = judge.plan(log, other)
    except ValueError as ex:
        print(f"error: {ex}", file=sys.stderr)
        return 2
    calls = judge.calls_needed(len(graded_items), len(pairs))
    tokens = judge.max_tokens_bound(graded_items, pairs)
    print(
        f"Plan: grade {len(graded_items)} replies + {len(pairs)} pairs x 2 orders on "
        f"{args.model}\n  exactly {calls} judge calls; worst case {tokens} tokens"
    )
    if calls > args.max_calls or tokens > args.max_tokens:
        print(
            f"REFUSED: over the caps ({args.max_calls} calls, {args.max_tokens} tokens). "
            "Nothing was sent.",
            file=sys.stderr,
        )
        return 2
    load_env()
    report = judge.judge(get_model(args.model), graded_items, pairs)
    print(judge.render(report, args.model))
    return 0


def run_power(args: argparse.Namespace) -> int:
    try:
        pool, label = load_pool(args)
        cells = power.power_grid(
            pool,
            [int(x) for x in args.samples.split(",")],
            [int(x) for x in args.epochs.split(",")],
            drop=args.drop,
            reps=args.reps,
            alpha=args.alpha,
            n_resamples=args.resamples,
            seed=args.seed,
        )
    except ValueError as ex:
        print(f"error: {ex}", file=sys.stderr)
        return 2
    print(power.render(cells, args.drop, label, args.target))
    return 0


def run_calibrate(args: argparse.Namespace) -> int:
    try:
        pool, label = load_pool(args, args.samples)
        result = calibrate.calibrate(
            pool,
            run_epochs=args.epochs,
            drop=args.drop,
            reps=args.reps,
            alpha=args.alpha,
            n_resamples=args.resamples,
            seed=args.seed,
            looks=args.looks,
        )
    except ValueError as ex:
        print(f"error: {ex}", file=sys.stderr)
        return 2
    print(calibrate.render(result, label))
    return 0


if __name__ == "__main__":
    sys.exit(main())

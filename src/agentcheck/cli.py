"""`agentcheck` command line.

Exit codes for `compare`: 0 no regression detected, 1 regression, 2 usage or input error.
"""

import argparse
import sys
from collections.abc import Sequence

from inspect_ai.log import read_eval_log

from agentcheck import calibrate, summary
from agentcheck.compare import Verdict, compare_logs, render


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

    args = p.parse_args(argv)
    if args.command == "calibrate":
        return run_calibrate(args)
    if args.command == "summary":
        forwarded = [args.log] if args.log else []
        return summary.main([*forwarded, "--log-dir", args.log_dir])

    base, cand = read_eval_log(args.baseline), read_eval_log(args.candidate)
    try:
        result = compare_logs(
            base,
            cand,
            alpha=args.alpha,
            fdr=args.fdr,
            n_resamples=args.resamples,
            seed=args.seed,
        )
    except ValueError as ex:
        print(f"error: {ex}", file=sys.stderr)
        return 2
    print(render(result, base, cand))
    return 1 if result.verdict is Verdict.REGRESSION else 0


def run_calibrate(args: argparse.Namespace) -> int:
    if args.synthetic == bool(args.logs):
        print("error: give pool logs or --synthetic, not both or neither", file=sys.stderr)
        return 2
    try:
        if args.synthetic:
            pool = calibrate.synthetic_pool(
                n_samples=args.samples, mean=args.mean, spread=args.spread, seed=args.seed
            )
            label = f"synthetic (mean {args.mean}, spread {args.spread}, seed {args.seed})"
        else:
            logs = [read_eval_log(path) for path in args.logs]
            pool = calibrate.pool_from_logs(logs)
            label = f"{logs[0].eval.model}, {len(logs)} log(s)"
        result = calibrate.calibrate(
            pool,
            run_epochs=args.epochs,
            drop=args.drop,
            reps=args.reps,
            alpha=args.alpha,
            n_resamples=args.resamples,
            seed=args.seed,
        )
    except ValueError as ex:
        print(f"error: {ex}", file=sys.stderr)
        return 2
    print(calibrate.render(result, label))
    return 0


if __name__ == "__main__":
    sys.exit(main())

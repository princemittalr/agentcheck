"""`agentcheck` command line.

Exit codes for `compare`: 0 no regression detected, 1 regression, 2 usage or input error.
"""

import argparse
import sys
from collections.abc import Sequence

from inspect_ai.log import read_eval_log

from agentcheck import summary
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

    args = p.parse_args(argv)
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


if __name__ == "__main__":
    sys.exit(main())

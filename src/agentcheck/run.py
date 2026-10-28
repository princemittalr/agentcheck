"""Run the task suite against a real model with hard caps on calls and tokens.

The caps are checked *before* anything is sent: if the worst case could exceed them,
the run is refused. After the run, the actual counts are printed next to the bounds.
"""

import argparse
import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from inspect_ai import eval as inspect_eval
from inspect_ai import eval_retry
from inspect_ai.log import EvalLog, read_eval_log
from inspect_ai.model import ChatMessageAssistant, Model

from agentcheck.tasks.inventory import MESSAGE_LIMIT, build_dataset, inventory_lookup

PROMPT_MESSAGES = 2  # system + user, present before the first model call
# Groq's free tier limits tokens per minute (gpt-oss-20b: 8,000 total; qwen3.8-27b: 1,000
# output). Requests go one at a time, and a rate-limited request is retried. 8 retries
# ran out on qwen's output limit, so 40: rejected requests use no tokens, and a
# per-minute window always clears well within 40 attempts.
MAX_RETRIES = 40
MAX_CONNECTIONS = 1


class BudgetExceededError(Exception):
    """The worst-case usage of a planned run is above the caps the user gave."""


def max_calls_per_sample(message_limit: int) -> int:
    """Upper bound on model calls in one sample under a message limit.

    After the prompt, every call that does not end the sample adds an assistant
    message plus at least one tool result, so each call uses up 2 or more messages.
    """
    if message_limit <= PROMPT_MESSAGES:
        return 0
    return math.ceil((message_limit - PROMPT_MESSAGES) / 2)


@dataclass(frozen=True)
class Budget:
    samples: int
    epochs: int
    calls_per_sample: int
    token_limit_per_sample: int

    @property
    def max_model_calls(self) -> int:
        return self.samples * self.epochs * self.calls_per_sample

    @property
    def max_http_requests(self) -> int:
        return self.max_model_calls * (1 + MAX_RETRIES)

    @property
    def max_tokens(self) -> int:
        return self.samples * self.epochs * self.token_limit_per_sample


def check_budget(budget: Budget, max_calls: int, max_tokens: int) -> None:
    if budget.max_model_calls > max_calls:
        raise BudgetExceededError(
            f"worst case is {budget.max_model_calls} model calls, cap is {max_calls}"
        )
    if budget.max_tokens > max_tokens:
        raise BudgetExceededError(f"worst case is {budget.max_tokens} tokens, cap is {max_tokens}")


def count_model_calls(log: EvalLog) -> int:
    """Completed model calls, counted as assistant messages in the transcripts."""
    samples = log.samples or []
    return sum(isinstance(m, ChatMessageAssistant) for s in samples for m in s.messages)


def count_model_output_errors(log: EvalLog) -> int:
    """Samples ended by unusable model output (scored wrong, not dropped)."""
    samples = log.samples or []
    return sum("model_output_error" in (s.metadata or {}) for s in samples)


def count_tokens(log: EvalLog) -> int:
    return sum(u.total_tokens for u in log.stats.model_usage.values())


def run_capped(
    model: str | Model,
    budget: Budget,
    *,
    log_dir: str,
    seed: int | None = None,
) -> EvalLog:
    [log] = inspect_eval(
        inventory_lookup(),
        model=model,
        epochs=budget.epochs,
        limit=budget.samples,
        token_limit=budget.token_limit_per_sample,
        max_retries=MAX_RETRIES,
        max_connections=MAX_CONNECTIONS,
        seed=seed,
        log_dir=log_dir,
    )
    return log


def remaining_samples(log: EvalLog) -> int:
    """Sample-epochs a resumed run still has to do (finished, error-free ones are kept)."""
    total = len(build_dataset()) if log.eval.dataset.samples is None else log.eval.dataset.samples
    if log.eval.config.limit is not None and isinstance(log.eval.config.limit, int):
        total = min(total, log.eval.config.limit)
    done = sum(s.error is None for s in log.samples or [])
    return total * (log.eval.config.epochs or 1) - done


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", help="e.g. groq/qwen/qwen3.8-27b (not used with --resume)")
    p.add_argument("--resume", help="an interrupted .eval log to finish")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--limit", type=int, default=None, help="first N samples only")
    p.add_argument("--max-calls", type=int, required=True, help="hard cap on model calls")
    p.add_argument("--max-tokens", type=int, required=True, help="hard cap on total tokens")
    p.add_argument("--token-limit-per-sample", type=int, default=16_000)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--log-dir", default="logs")
    return p.parse_args(argv)


def report(log: EvalLog, budget: Budget) -> int:
    print(
        f"Status: {log.status}\n"
        f"  model calls: {count_model_calls(log)} (bound {budget.max_model_calls})\n"
        f"  tokens: {count_tokens(log)} (bound {budget.max_tokens})\n"
        f"  model output errors (scored as wrong): {count_model_output_errors(log)}\n"
        f"  log: {Path(log.location)}"
    )
    if log.error is not None:
        print(f"Error: {log.error.message}", file=sys.stderr)
    return 0 if log.status == "success" else 1


def resume(args: argparse.Namespace) -> int:
    old = read_eval_log(args.resume)
    budget = Budget(
        samples=remaining_samples(old),
        epochs=1,
        calls_per_sample=max_calls_per_sample(MESSAGE_LIMIT),
        token_limit_per_sample=args.token_limit_per_sample,
    )
    print(
        f"Resume: {budget.samples} sample-epochs left in {args.resume}\n"
        f"  worst case: {budget.max_model_calls} model calls, {budget.max_tokens} tokens"
    )
    try:
        check_budget(budget, args.max_calls, args.max_tokens)
    except BudgetExceededError as e:
        print(f"REFUSED: {e}. Nothing was sent.", file=sys.stderr)
        return 2
    [log] = eval_retry(
        old,
        log_dir=args.log_dir,
        max_retries=MAX_RETRIES,
        max_connections=MAX_CONNECTIONS,
    )
    # Bounds in the report are for the resumed part; counts cover the whole log.
    return report(log, budget)


def load_env() -> None:
    """Load .env from the working directory (or a parent) without overriding real env vars.

    Inspect does this inside eval() but not inside eval_retry(), so resuming a run
    failed with "No GROQ_API_KEY defined". Loading it here covers both paths.
    """
    load_dotenv(find_dotenv(usecwd=True), override=False)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    load_env()
    if args.resume:
        return resume(args)
    if not args.model:
        print("REFUSED: --model is required unless --resume is given.", file=sys.stderr)
        return 2
    n_samples = len(build_dataset())
    budget = Budget(
        samples=min(args.limit, n_samples) if args.limit else n_samples,
        epochs=args.epochs,
        calls_per_sample=max_calls_per_sample(MESSAGE_LIMIT),
        token_limit_per_sample=args.token_limit_per_sample,
    )
    print(
        f"Plan: {budget.samples} samples x {budget.epochs} epochs on {args.model}\n"
        f"  worst case: {budget.max_model_calls} model calls "
        f"({budget.max_http_requests} HTTP requests with retries), {budget.max_tokens} tokens"
    )
    if args.seed is not None and args.epochs > 1:
        # The same seed is sent for every epoch, so epochs could be identical replays:
        # that hides exactly the run-to-run variance AgentCheck exists to measure.
        print("REFUSED: --seed with --epochs > 1 would repeat one sample path.", file=sys.stderr)
        return 2
    try:
        check_budget(budget, args.max_calls, args.max_tokens)
    except BudgetExceededError as e:
        print(f"REFUSED: {e}. Nothing was sent.", file=sys.stderr)
        return 2

    log = run_capped(args.model, budget, log_dir=args.log_dir, seed=args.seed)
    return report(log, budget)


if __name__ == "__main__":
    sys.exit(main())

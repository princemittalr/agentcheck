"""LLM-judge calibration against the deterministic scorer.

The suite's own scorer (exact ANSWER line) is the reference. A judge model grades the
same final replies; Cohen's kappa says how far it agrees beyond chance. A second check
shows the judge two replies to the same question in both orders and tests whether it
favours a position rather than an answer. Both tell you whether an LLM judge could stand
in for a scorer on tasks that have none.
"""

import asyncio
import re
from collections.abc import Sequence
from dataclasses import dataclass

from inspect_ai.log import EvalLog
from inspect_ai.model import GenerateConfig, Model
from inspect_ai.scorer import CORRECT

from agentcheck.run import MAX_CONNECTIONS, MAX_RETRIES
from agentcheck.stats.agreement import Kappa, PositionBias, kappa_from_labels, position_bias
from agentcheck.summary import SCORER

JUDGE_MAX_TOKENS = 1024  # reasoning models spend tokens before the one-word verdict

GRADE_PROMPT = """You are grading an answer to a warehouse-inventory question.

Question: {question}
Reference answer: {target}
Agent's final reply:
---
{reply}
---
Is the agent's final answer the same as the reference answer? Ignore formatting such as
$ signs, commas or bold text. Reply with exactly one word: CORRECT or INCORRECT."""

PAIR_PROMPT = """Two assistants answered the same warehouse-inventory question.

Question: {question}
Reference answer: {target}

Reply 1:
---
{first}
---
Reply 2:
---
{second}
---
Which reply gives the reference answer? Reply with exactly one character: 1 or 2."""


@dataclass(frozen=True)
class Graded:
    sample_id: str
    reference: bool  # the deterministic scorer's verdict
    judge: bool | None  # None: the judge's reply could not be parsed


@dataclass(frozen=True)
class JudgeReport:
    graded: list[Graded]
    kappa: Kappa | None
    unparsed: int
    bias: PositionBias | None
    pairs_unparsed: int


@dataclass(frozen=True)
class Item:
    sample_id: str
    question: str
    target: str
    reply: str
    correct: bool


def items(log: EvalLog) -> dict[str, Item]:
    """First epoch of each sample: question, target, final reply, scorer verdict."""
    if log.status != "success":
        raise ValueError(f"log status is {log.status!r}, not 'success'")
    out: dict[str, Item] = {}
    for s in log.samples or []:
        key = str(s.id)
        if key in out:
            continue
        if s.scores is None or SCORER not in s.scores:
            raise ValueError(f"sample {s.id} has no {SCORER} score")
        out[key] = Item(
            sample_id=key,
            question=s.input if isinstance(s.input, str) else str(s.input),
            target=s.target if isinstance(s.target, str) else ",".join(s.target),
            reply=s.output.completion,
            correct=s.scores[SCORER].value == CORRECT,
        )
    return out


def parse_grade(text: str) -> bool | None:
    words = re.findall(r"\b(INCORRECT|CORRECT)\b", text.upper())
    return None if not words else words[-1] == "CORRECT"


def parse_choice(text: str) -> str | None:
    found = re.findall(r"\b([12])\b", text)
    return None if not found else found[-1]


def calls_needed(n_items: int, n_pairs: int) -> int:
    return n_items + 2 * n_pairs


async def _ask(model: Model, prompt: str) -> str:
    config = GenerateConfig(
        max_tokens=JUDGE_MAX_TOKENS,
        temperature=0.0,
        max_retries=MAX_RETRIES,
        max_connections=MAX_CONNECTIONS,
    )
    out = await model.generate(prompt, config=config)
    return out.completion


async def _run(
    model: Model, graded_items: Sequence[Item], pairs: Sequence[tuple[Item, Item]]
) -> JudgeReport:
    graded = []
    for it in graded_items:
        text = await _ask(
            model, GRADE_PROMPT.format(question=it.question, target=it.target, reply=it.reply)
        )
        graded.append(Graded(it.sample_id, it.correct, parse_grade(text)))
    parsed = [g for g in graded if g.judge is not None]
    kappa = None
    if parsed:
        try:
            kappa = kappa_from_labels([g.judge for g in parsed], [g.reference for g in parsed])
        except ValueError:
            kappa = None  # undefined: judge and scorer used one identical label throughout

    winners_ab: list[str] = []
    winners_ba: list[str] = []
    pairs_unparsed = 0
    for a, b in pairs:
        fmt = {"question": a.question, "target": a.target}
        ab = parse_choice(
            await _ask(model, PAIR_PROMPT.format(first=a.reply, second=b.reply, **fmt))
        )
        ba = parse_choice(
            await _ask(model, PAIR_PROMPT.format(first=b.reply, second=a.reply, **fmt))
        )
        if ab is None or ba is None:
            pairs_unparsed += 1
            continue
        winners_ab.append("A" if ab == "1" else "B")
        winners_ba.append("B" if ba == "1" else "A")
    bias = position_bias(winners_ab, winners_ba) if winners_ab else None
    return JudgeReport(graded, kappa, len(graded) - len(parsed), bias, pairs_unparsed)


def informative_pairs(a: dict[str, Item], b: dict[str, Item]) -> list[tuple[Item, Item]]:
    """Samples where exactly one of the two logs is right: the judge has a true answer."""
    if a.keys() != b.keys():
        raise ValueError("logs cover different samples")
    return [(a[k], b[k]) for k in sorted(a) if a[k].correct != b[k].correct]


def plan(log: EvalLog, other: EvalLog | None = None) -> tuple[list[Item], list[tuple[Item, Item]]]:
    graded_items = list(items(log).values())
    pairs = informative_pairs(items(log), items(other)) if other is not None else []
    return graded_items, pairs


def max_tokens_bound(graded_items: Sequence[Item], pairs: Sequence[tuple[Item, Item]]) -> int:
    """Worst case: every prompt at 1 token per 2 characters (English averages ~4) plus the
    full output allowance on every call."""
    prompts = [
        GRADE_PROMPT.format(question=i.question, target=i.target, reply=i.reply)
        for i in graded_items
    ]
    for a, b in pairs:
        text = PAIR_PROMPT.format(
            question=a.question, target=a.target, first=a.reply, second=b.reply
        )
        prompts += [text, text]
    return sum(len(p) // 2 + 1 + JUDGE_MAX_TOKENS for p in prompts)


def judge(
    model: Model, graded_items: Sequence[Item], pairs: Sequence[tuple[Item, Item]]
) -> JudgeReport:
    return asyncio.run(_run(model, graded_items, pairs))


def render(r: JudgeReport, model_name: str) -> str:
    lines = [f"Judge: {model_name}", f"  graded {len(r.graded)} replies, unparsed {r.unparsed}"]
    if r.kappa is None:
        lines.append("  kappa: undefined (no variation in labels)")
    else:
        k = r.kappa
        lines.append(
            f"  agreement with the scorer {k.observed:.3f} (chance {k.expected:.3f}); "
            f"kappa {k.kappa:.3f}, 95% CI [{k.low:.3f}, {k.high:.3f}], n = {k.n}"
        )
        fp = sum(g.judge is True and not g.reference for g in r.graded)
        fn = sum(g.judge is False and g.reference for g in r.graded)
        lines.append(f"  judge said CORRECT for {fp} wrong replies, INCORRECT for {fn} right ones")
    if r.bias is not None:
        b = r.bias
        lines.append(
            f"  position test on {b.n} pairs (unparsed {r.pairs_unparsed}): consistent "
            f"{b.consistent}/{b.n} ({b.consistency:.3f}); first shown won both times "
            f"{b.first_both}, second {b.second_both}; p = {b.p_value:.4g}"
        )
    return "\n".join(lines)

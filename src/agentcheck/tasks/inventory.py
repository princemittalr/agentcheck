"""Warehouse inventory suite: tool-using tasks with deterministic scoring.

All catalog data is fictional, so a model can only answer correctly by calling the
tools. Five categories vary in difficulty so the pass rate stays below 100%: a suite
that every model aces cannot show a regression.

  single   one lookup                       -> units in stock
  value    two lookups, multiply            -> stock x unit price, in dollars
  total    three lookups, add               -> combined units
  most     three lookups, compare           -> SKU with the most units
  unknown  SKU that does not exist          -> UNKNOWN (tests refusing to guess)
"""

import re
from dataclasses import dataclass

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import CORRECT, INCORRECT, Score, Scorer, Target, accuracy, scorer, stderr
from inspect_ai.solver import TaskState, generate, system_message, use_tools
from inspect_ai.tool import Tool, ToolError, tool


@dataclass(frozen=True)
class Item:
    stock: int  # units
    price: int  # whole dollars per unit


CATALOG: dict[str, Item] = {
    "SKU-1042": Item(37, 12),
    "SKU-2177": Item(0, 85),
    "SKU-3305": Item(412, 3),
    "SKU-4419": Item(8, 240),
    "SKU-5583": Item(1250, 1),
    "SKU-6021": Item(96, 17),
    "SKU-6650": Item(23, 49),
    "SKU-7108": Item(311, 6),
    "SKU-7734": Item(54, 31),
    "SKU-8266": Item(5, 999),
    "SKU-8890": Item(142, 11),
    "SKU-9317": Item(77, 26),
}

SYSTEM_PROMPT = (
    "You answer questions about warehouse inventory using the tools provided. Never "
    "guess a stock level or price: look it up. If any SKU in the question does not "
    "exist, the answer is UNKNOWN. End your reply with one final line of the form "
    "'ANSWER: <value>', where <value> is a number (digits only), a SKU, or UNKNOWN."
)

# Per-sample cap on messages. The hardest category needs 3 lookups plus an answer;
# 16 leaves room for retries after a tool error and bounds a looping model at 7 calls.
MESSAGE_LIMIT = 16


def _get(sku: str) -> Item:
    if sku not in CATALOG:
        # Shown to the model as a tool result; it does not crash the sample.
        raise ToolError(f"Unknown SKU: {sku}")
    return CATALOG[sku]


@tool
def lookup_stock() -> Tool:
    async def execute(sku: str) -> str:
        """Look up how many units of a product are in stock.

        Args:
            sku: The product SKU, for example "SKU-1042".

        Returns:
            The number of units in stock, as digits.
        """
        return str(_get(sku).stock)

    return execute


@tool
def lookup_price() -> Tool:
    async def execute(sku: str) -> str:
        """Look up the unit price of a product, in whole dollars.

        Args:
            sku: The product SKU, for example "SKU-1042".

        Returns:
            The price of one unit in dollars, as digits.
        """
        return str(_get(sku).price)

    return execute


SINGLE = ["SKU-1042", "SKU-2177", "SKU-3305", "SKU-4419", "SKU-5583", "SKU-8266"]
VALUE = [
    "SKU-1042",
    "SKU-3305",
    "SKU-4419",
    "SKU-6021",
    "SKU-6650",
    "SKU-7734",
    "SKU-8266",
    "SKU-9317",
]
TRIPLES = [
    ("SKU-1042", "SKU-6650", "SKU-7734"),
    ("SKU-3305", "SKU-7108", "SKU-8890"),
    ("SKU-2177", "SKU-4419", "SKU-8266"),
    ("SKU-5583", "SKU-3305", "SKU-9317"),
    ("SKU-6021", "SKU-9317", "SKU-7734"),
    ("SKU-8890", "SKU-1042", "SKU-6650"),
    ("SKU-4419", "SKU-6021", "SKU-7108"),
    ("SKU-8266", "SKU-2177", "SKU-1042"),
]
# Near-misses of real SKUs (one digit off, transposed) plus clearly fake ones.
UNKNOWN = ["SKU-1043", "SKU-4491", "SKU-6605", "SKU-7801", "SKU-0000", "SKU-9999"]


def build_dataset() -> MemoryDataset:
    samples: list[Sample] = []
    for sku in SINGLE:
        samples.append(
            Sample(
                id=f"single-{sku}",
                input=f"How many units of {sku} are in stock?",
                target=str(CATALOG[sku].stock),
                metadata={"category": "single"},
            )
        )
    for sku in VALUE:
        item = CATALOG[sku]
        samples.append(
            Sample(
                id=f"value-{sku}",
                input=f"What is the total value in dollars of all {sku} units in stock "
                "(units in stock times unit price)?",
                target=str(item.stock * item.price),
                metadata={"category": "value"},
            )
        )
    for i, skus in enumerate(TRIPLES):
        listed = ", ".join(skus)
        samples.append(
            Sample(
                id=f"total-{i}",
                input=f"How many units are in stock in total across {listed}?",
                target=str(sum(CATALOG[s].stock for s in skus)),
                metadata={"category": "total"},
            )
        )
        samples.append(
            Sample(
                id=f"most-{i}",
                input=f"Which of {listed} has the most units in stock? Answer with the SKU.",
                target=max(skus, key=lambda s: CATALOG[s].stock),
                metadata={"category": "most"},
            )
        )
    for sku in UNKNOWN:
        samples.append(
            Sample(
                id=f"unknown-{sku}",
                input=f"How many units of {sku} are in stock?",
                target="UNKNOWN",
                metadata={"category": "unknown"},
            )
        )
    return MemoryDataset(samples, name="inventory")


ANSWER_RE = re.compile(r"^\s*\**ANSWER\**:\**\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)


def extract_answer(completion: str) -> str | None:
    """The value on the last 'ANSWER:' line, or None if there is none."""
    found = ANSWER_RE.findall(completion)
    return found[-1] if found else None


def normalize(value: str) -> str:
    """Case, surrounding markdown/punctuation, '$' and thousands separators don't count."""
    v = value.strip().strip("*`.").replace("$", "").replace(",", "").strip().casefold()
    # "37.0" and "37" are the same number; "37.5" stays as is.
    if re.fullmatch(r"-?\d+\.0+", v):
        v = v.split(".")[0]
    return v


@scorer(metrics=[accuracy(), stderr()])
def answer_line() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        answer = extract_answer(state.output.completion)
        if answer is None:
            return Score(value=INCORRECT, answer=None, explanation="no ANSWER: line")
        correct = normalize(answer) == normalize(target.text)
        return Score(value=CORRECT if correct else INCORRECT, answer=answer)

    return score


@task
def inventory_lookup() -> Task:
    return Task(
        dataset=build_dataset(),
        solver=[
            system_message(SYSTEM_PROMPT),
            use_tools(lookup_stock(), lookup_price()),
            generate(),
        ],
        scorer=answer_line(),
        message_limit=MESSAGE_LIMIT,
    )

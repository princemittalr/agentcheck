"""Warehouse inventory suite: tool-using tasks with deterministic scoring.

All catalog data is fictional, so a model can only answer correctly by calling the
tools. Five categories vary in difficulty so the pass rate stays below 100%: a suite
that every model aces cannot show a regression.

  single   one lookup                       -> units in stock
  value    two lookups, multiply            -> stock x unit price, in dollars
  total    three lookups, add               -> combined units
  most     three lookups, compare           -> SKU with the most units
  unknown  SKU that does not exist          -> UNKNOWN (tests refusing to guess)
  reorder  10 lookups, compare per SKU      -> how many of 5 SKUs need reordering
  portfolio 8 lookups, multiply and add     -> total dollar value of 4 SKUs
  orders   2 lookups, conditional logic     -> units left after two orders
"""

import re
from dataclasses import dataclass
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ModelOutput
from inspect_ai.scorer import CORRECT, INCORRECT, Score, Scorer, Target, accuracy, scorer, stderr
from inspect_ai.solver import Generate, Solver, TaskState, solver, system_message, use_tools
from inspect_ai.tool import Tool, ToolError, tool

from agentcheck.replay import Cassette, ToolMode, recorded


@dataclass(frozen=True)
class Item:
    stock: int  # units
    price: int  # whole dollars per unit
    reorder: int  # reorder when stock is at or below this level


CATALOG: dict[str, Item] = {
    "SKU-1042": Item(37, 12, 40),
    "SKU-2177": Item(0, 85, 10),
    "SKU-3305": Item(412, 3, 100),
    "SKU-4419": Item(8, 240, 8),
    "SKU-5583": Item(1250, 1, 500),
    "SKU-6021": Item(96, 17, 120),
    "SKU-6650": Item(23, 49, 20),
    "SKU-7108": Item(311, 6, 300),
    "SKU-7734": Item(54, 31, 60),
    "SKU-8266": Item(5, 999, 2),
    "SKU-8890": Item(142, 11, 150),
    "SKU-9317": Item(77, 26, 77),
}

SYSTEM_PROMPT = (
    "You answer questions about warehouse inventory using the tools provided. Never "
    "guess a stock level or price: look it up. If any SKU in the question does not "
    "exist, the answer is UNKNOWN. End your reply with one final line of the form "
    "'ANSWER: <value>', where <value> is a number (digits only), a SKU, or UNKNOWN."
)

# Per-sample cap on messages. The hardest category (reorder) needs 10 lookups; done one
# per call that is 2 + 20 + 1 = 23 messages. 24 bounds a looping model at 11 calls.
MESSAGE_LIMIT = 24


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
def lookup_reorder_level() -> Tool:
    async def execute(sku: str) -> str:
        """Look up the reorder level of a product: reorder when stock is at or below it.

        Args:
            sku: The product SKU, for example "SKU-1042".

        Returns:
            The reorder level in units, as digits.
        """
        return str(_get(sku).reorder)

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
QUINTUPLES = [
    ("SKU-1042", "SKU-3305", "SKU-4419", "SKU-6650", "SKU-8266"),
    ("SKU-2177", "SKU-5583", "SKU-6021", "SKU-7108", "SKU-9317"),
    ("SKU-7734", "SKU-8890", "SKU-1042", "SKU-3305", "SKU-5583"),
    ("SKU-4419", "SKU-9317", "SKU-6021", "SKU-2177", "SKU-7734"),
    ("SKU-6650", "SKU-7108", "SKU-8266", "SKU-3305", "SKU-5583"),
    ("SKU-8890", "SKU-6021", "SKU-7734", "SKU-1042", "SKU-2177"),
    ("SKU-9317", "SKU-4419", "SKU-6650", "SKU-8890", "SKU-7108"),
    ("SKU-5583", "SKU-8266", "SKU-3305", "SKU-7108", "SKU-6650"),
]
QUADRUPLES = [
    ("SKU-1042", "SKU-3305", "SKU-6021", "SKU-9317"),
    ("SKU-4419", "SKU-6650", "SKU-7734", "SKU-8890"),
    ("SKU-8266", "SKU-7108", "SKU-5583", "SKU-1042"),
    ("SKU-2177", "SKU-6021", "SKU-8890", "SKU-3305"),
    ("SKU-9317", "SKU-7734", "SKU-4419", "SKU-7108"),
    ("SKU-6650", "SKU-8266", "SKU-6021", "SKU-5583"),
    ("SKU-3305", "SKU-9317", "SKU-8890", "SKU-4419"),
    ("SKU-7108", "SKU-1042", "SKU-7734", "SKU-8266"),
]
# (sku_a, units_a, sku_b, units_b); some orders exceed stock and must be rejected.
ORDERS = [
    ("SKU-1042", 30, "SKU-6650", 30),
    ("SKU-3305", 400, "SKU-4419", 9),
    ("SKU-7734", 54, "SKU-9317", 70),
    ("SKU-2177", 1, "SKU-8890", 142),
    ("SKU-5583", 1300, "SKU-6021", 50),
    ("SKU-8266", 5, "SKU-7108", 312),
    ("SKU-6650", 23, "SKU-1042", 38),
    ("SKU-9317", 10, "SKU-3305", 12),
]


def units_left(sku: str, ordered: int) -> int:
    """An order is filled only if stock covers it entirely; otherwise it is rejected."""
    stock = CATALOG[sku].stock
    return stock - ordered if ordered <= stock else stock


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
    for i, skus5 in enumerate(QUINTUPLES):
        samples.append(
            Sample(
                id=f"reorder-{i}",
                input=f"How many of {', '.join(skus5)} need reordering, i.e. have stock at "
                "or below their reorder level?",
                target=str(sum(CATALOG[s].stock <= CATALOG[s].reorder for s in skus5)),
                metadata={"category": "reorder"},
            )
        )
    for i, skus4 in enumerate(QUADRUPLES):
        samples.append(
            Sample(
                id=f"portfolio-{i}",
                input=f"What is the total value in dollars of all stock of {', '.join(skus4)} "
                "combined (for each SKU, units in stock times unit price, then add)?",
                target=str(sum(CATALOG[s].stock * CATALOG[s].price for s in skus4)),
                metadata={"category": "portfolio"},
            )
        )
    for i, (a, n, b, m) in enumerate(ORDERS):
        samples.append(
            Sample(
                id=f"orders-{i}",
                input=f"Two orders arrive: {n} units of {a} and {m} units of {b}. An order is "
                "filled only if there is enough stock for all of it; otherwise it is rejected "
                f"and stock is unchanged. After both orders, how many units of {a} and {b} "
                "remain in total?",
                target=str(units_left(a, n) + units_left(b, m)),
                metadata={"category": "orders"},
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


# Provider error codes meaning "the model produced output that could not be used".
# These are the model's failures, so they are scored as wrong answers. Every other
# error (rate limits, network, auth) is infrastructure and still fails the whole run.
MODEL_OUTPUT_ERROR_CODES = ("output_parse_failed",)


def model_output_error_code(ex: BaseException) -> str | None:
    """The matching code if `ex` is a 400 caused by unusable model output, else None.

    Inspect raises `ModelGenerateError` (a RuntimeError) carrying the provider's
    `status_code` and `provider_message`; it isn't exported publicly, so we read
    those two attributes rather than import a private class.
    """
    if getattr(ex, "status_code", None) != 400:
        return None
    message = str(getattr(ex, "provider_message", "") or "")
    return next((code for code in MODEL_OUTPUT_ERROR_CODES if code in message), None)


@solver
def generate_scoring_output_errors() -> Solver:
    """`generate()` with the tool loop, but a model-output error ends the sample as wrong."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        try:
            return await generate(state)
        except RuntimeError as ex:
            code = model_output_error_code(ex)
            if code is None:
                raise
            state.metadata["model_output_error"] = code
            state.output = ModelOutput.from_content(
                model=str(state.model), content="", stop_reason="unknown", error=code
            )
            state.completed = True
            return state

    return solve


@scorer(metrics=[accuracy(), stderr()])
def answer_line() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        error_code = state.metadata.get("model_output_error")
        if error_code is not None:
            return Score(
                value=INCORRECT, answer=None, explanation=f"model output error: {error_code}"
            )
        answer = extract_answer(state.output.completion)
        if answer is None:
            return Score(value=INCORRECT, answer=None, explanation="no ANSWER: line")
        correct = normalize(answer) == normalize(target.text)
        return Score(value=CORRECT if correct else INCORRECT, answer=answer)

    return score


@task
def inventory_lookup(tools: str = "live", cassette: str | None = None) -> Task:
    """tools: live (default), record (call tools, save to cassette) or replay (cassette only)."""
    mode = ToolMode(tools)
    agent_tools = [lookup_stock(), lookup_price(), lookup_reorder_level()]
    if mode is not ToolMode.LIVE:
        if cassette is None:
            raise ValueError(f"tools={mode.value} needs a cassette path")
        tape = Cassette.open(Path(cassette), mode)
        agent_tools = [recorded(t, tape, mode) for t in agent_tools]
    return Task(
        dataset=build_dataset(),
        solver=[
            system_message(SYSTEM_PROMPT),
            use_tools(*agent_tools),
            generate_scoring_output_errors(),
        ],
        scorer=answer_line(),
        message_limit=MESSAGE_LIMIT,
    )

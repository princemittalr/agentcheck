"""Inventory lookup: the smallest tool-using task with a deterministic scorer.

The stock numbers are fictional, so a model can only answer correctly by calling the
`lookup_stock` tool. Scoring is a numeric match on the final answer: no LLM judge, so
the same transcript always gets the same score.
"""

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import match
from inspect_ai.solver import generate, system_message, use_tools
from inspect_ai.tool import Tool, ToolError, tool

INVENTORY: dict[str, int] = {
    "SKU-1042": 37,
    "SKU-2177": 0,
    "SKU-3305": 412,
    "SKU-4419": 8,
    "SKU-5583": 1250,
}

SYSTEM_PROMPT = (
    "You answer questions about warehouse stock. Always use the lookup_stock tool; "
    "never guess. End your reply with the number of units as digits only."
)

# Per-sample cap on messages (system + user + assistant + tool). A normal solve uses 5;
# 8 bounds a looping model at 3 model calls per sample.
MESSAGE_LIMIT = 8


@tool
def lookup_stock() -> Tool:
    async def execute(sku: str) -> str:
        """Look up how many units of a product are in stock.

        Args:
            sku: The product SKU, for example "SKU-1042".

        Returns:
            The number of units in stock, as a string of digits.
        """
        if sku not in INVENTORY:
            # ToolError is shown to the model as a tool result; it does not crash the
            # sample. Any other exception would fail the sample loudly, which we want.
            raise ToolError(f"Unknown SKU: {sku}")
        return str(INVENTORY[sku])

    return execute


def build_dataset() -> MemoryDataset:
    samples = [
        Sample(
            id=sku,
            input=f"How many units of {sku} are in stock?",
            target=str(units),
        )
        for sku, units in INVENTORY.items()
    ]
    return MemoryDataset(samples, name="inventory_lookup")


@task
def inventory_lookup() -> Task:
    return Task(
        dataset=build_dataset(),
        solver=[system_message(SYSTEM_PROMPT), use_tools(lookup_stock()), generate()],
        scorer=match(numeric=True),
        message_limit=MESSAGE_LIMIT,
    )

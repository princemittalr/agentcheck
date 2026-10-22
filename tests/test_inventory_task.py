"""Offline tests for the inventory task. No network: the model is Inspect's mockllm."""

import asyncio
from pathlib import Path

import pytest
from inspect_ai import eval as inspect_eval
from inspect_ai.log import EvalLog
from inspect_ai.model import ChatMessageTool, ModelOutput, ModelUsage, get_model
from inspect_ai.tool import ToolError

from agentcheck.tasks.inventory import INVENTORY, build_dataset, inventory_lookup, lookup_stock

MOCK = "mockllm/model"


# mockllm counts tokens with tiktoken when usage is missing, and tiktoken downloads its
# encoding on first use. Setting usage explicitly keeps these tests fully offline.
def _offline(output: ModelOutput) -> ModelOutput:
    output.usage = ModelUsage(input_tokens=0, output_tokens=0, total_tokens=0)
    return output


def tool_call(sku: str) -> ModelOutput:
    return _offline(ModelOutput.for_tool_call(MOCK, "lookup_stock", {"sku": sku}))


def reply(text: str) -> ModelOutput:
    return _offline(ModelOutput.from_content(MOCK, text))


def run_one_sample(sku: str, outputs: list[ModelOutput], log_dir: Path) -> EvalLog:
    model = get_model(MOCK, custom_outputs=outputs)
    [log] = inspect_eval(
        inventory_lookup(), model=model, sample_id=sku, log_dir=str(log_dir), display="none"
    )
    assert log.status == "success", log.error
    return log


def accuracy(log: EvalLog) -> float:
    assert log.results is not None
    return log.results.scores[0].metrics["accuracy"].value


def test_tool_returns_stock_for_known_sku() -> None:
    assert asyncio.run(lookup_stock()(sku="SKU-3305")) == "412"


def test_tool_raises_tool_error_for_unknown_sku() -> None:
    with pytest.raises(ToolError, match="Unknown SKU"):
        asyncio.run(lookup_stock()(sku="SKU-0000"))


def test_dataset_targets_match_inventory() -> None:
    samples = list(build_dataset())
    assert len(samples) == len(INVENTORY)
    assert len({s.id for s in samples}) == len(samples)
    for s in samples:
        assert s.target == str(INVENTORY[str(s.id)])


def test_correct_tool_use_scores_one(tmp_path: Path) -> None:
    outputs = [tool_call("SKU-1042"), reply("SKU-1042 has 37 units in stock. 37")]
    log = run_one_sample("SKU-1042", outputs, tmp_path)
    assert accuracy(log) == 1.0
    assert log.samples is not None
    tool_msgs = [m for m in log.samples[0].messages if isinstance(m, ChatMessageTool)]
    assert [m.text for m in tool_msgs] == ["37"]


def test_wrong_answer_scores_zero(tmp_path: Path) -> None:
    outputs = [reply("I think it is 40")]
    log = run_one_sample("SKU-1042", outputs, tmp_path)
    assert accuracy(log) == 0.0


def test_zero_stock_is_not_confused_with_missing_answer(tmp_path: Path) -> None:
    outputs = [tool_call("SKU-2177"), reply("There are no units left. 0")]
    log = run_one_sample("SKU-2177", outputs, tmp_path)
    assert accuracy(log) == 1.0

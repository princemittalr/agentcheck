"""Offline tests for the inventory suite. No network: the model is Inspect's mockllm."""

import asyncio
from collections import Counter
from pathlib import Path

import pytest
from inspect_ai import eval as inspect_eval
from inspect_ai.log import EvalLog
from inspect_ai.model import ChatMessageTool, ModelOutput, ModelUsage, get_model
from inspect_ai.tool import ToolError

from agentcheck.tasks.inventory import (
    CATALOG,
    build_dataset,
    extract_answer,
    inventory_lookup,
    lookup_price,
    lookup_stock,
    normalize,
)

MOCK = "mockllm/model"


# mockllm counts tokens with tiktoken when usage is missing, and tiktoken downloads its
# encoding on first use. Setting usage explicitly keeps these tests fully offline.
def _offline(output: ModelOutput) -> ModelOutput:
    output.usage = ModelUsage(input_tokens=0, output_tokens=0, total_tokens=0)
    return output


def tool_call(name: str, sku: str) -> ModelOutput:
    return _offline(ModelOutput.for_tool_call(MOCK, name, {"sku": sku}))


def reply(text: str) -> ModelOutput:
    return _offline(ModelOutput.from_content(MOCK, text))


def run_one_sample(sample_id: str, outputs: list[ModelOutput], log_dir: Path) -> EvalLog:
    model = get_model(MOCK, custom_outputs=outputs)
    [log] = inspect_eval(
        inventory_lookup(), model=model, sample_id=sample_id, log_dir=str(log_dir), display="none"
    )
    assert log.status == "success", log.error
    return log


def accuracy(log: EvalLog) -> float:
    assert log.results is not None
    return log.results.scores[0].metrics["accuracy"].value


# --- tools ---------------------------------------------------------------------------


def test_tools_return_catalog_values() -> None:
    assert asyncio.run(lookup_stock()(sku="SKU-3305")) == "412"
    assert asyncio.run(lookup_price()(sku="SKU-8266")) == "999"


@pytest.mark.parametrize("tool_factory", [lookup_stock, lookup_price])
def test_tools_raise_tool_error_for_unknown_sku(tool_factory: object) -> None:
    assert callable(tool_factory)
    with pytest.raises(ToolError, match="Unknown SKU"):
        asyncio.run(tool_factory()(sku="SKU-0000"))


# --- dataset -------------------------------------------------------------------------


def test_dataset_size_ids_and_categories() -> None:
    samples = list(build_dataset())
    assert len(samples) >= 30
    assert len({s.id for s in samples}) == len(samples)
    counts = Counter(str((s.metadata or {})["category"]) for s in samples)
    assert counts == {"single": 6, "value": 8, "total": 8, "most": 8, "unknown": 6}


def test_hand_checked_targets() -> None:
    """Targets worked out by hand, so a bug in build_dataset can't hide behind itself."""
    targets = {s.id: s.target for s in build_dataset()}
    assert targets["single-SKU-2177"] == "0"
    assert targets["value-SKU-8266"] == "4995"  # 5 x 999
    assert targets["value-SKU-3305"] == "1236"  # 412 x 3
    assert targets["total-0"] == "114"  # 37 + 23 + 54
    assert targets["most-3"] == "SKU-5583"  # 1250 > 412 > 77
    assert targets["unknown-SKU-1043"] == "UNKNOWN"


def test_most_questions_have_no_ties() -> None:
    for s in build_dataset():
        if (s.metadata or {})["category"] == "most":
            skus = [w.rstrip("?,") for w in str(s.input).split() if w.startswith("SKU-")]
            stocks = sorted((CATALOG[k].stock for k in skus), reverse=True)
            assert stocks[0] > stocks[1], s.id


def test_unknown_skus_are_really_unknown() -> None:
    for s in build_dataset():
        if s.target == "UNKNOWN":
            sku = str(s.id).removeprefix("unknown-")
            assert sku not in CATALOG


# --- answer parsing ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Looked it up.\nANSWER: 37", "37"),
        ("answer: sku-5583", "sku-5583"),
        ("**ANSWER:** 1,236", "1,236"),
        ("ANSWER: 10\nwait, let me fix that\nANSWER: 12", "12"),
        ("The answer is 37", None),
        ("", None),
    ],
)
def test_extract_answer(text: str, expected: str | None) -> None:
    assert extract_answer(text) == expected


@pytest.mark.parametrize(
    ("a", "b", "same"),
    [
        ("1,236", "1236", True),
        ("$4995", "4995", True),
        ("37.0", "37", True),
        ("37.5", "37", False),
        ("sku-5583", "SKU-5583", True),
        ("unknown.", "UNKNOWN", True),
        ("0", "UNKNOWN", False),
        ("370", "37", False),
    ],
)
def test_normalize(a: str, b: str, same: bool) -> None:
    assert (normalize(a) == normalize(b)) is same


# --- end to end on mockllm -----------------------------------------------------------


def test_correct_two_tool_answer_scores_one(tmp_path: Path) -> None:
    outputs = [
        tool_call("lookup_stock", "SKU-8266"),
        tool_call("lookup_price", "SKU-8266"),
        reply("5 units at $999 each.\nANSWER: 4,995"),
    ]
    log = run_one_sample("value-SKU-8266", outputs, tmp_path)
    assert accuracy(log) == 1.0
    assert log.samples is not None
    tool_msgs = [m for m in log.samples[0].messages if isinstance(m, ChatMessageTool)]
    assert [m.text for m in tool_msgs] == ["5", "999"]


def test_missing_answer_line_scores_zero(tmp_path: Path) -> None:
    log = run_one_sample("single-SKU-1042", [reply("There are 37 units.")], tmp_path)
    assert accuracy(log) == 0.0


def test_unknown_sku_tool_error_then_unknown_scores_one(tmp_path: Path) -> None:
    outputs = [
        tool_call("lookup_stock", "SKU-1043"),
        reply("That SKU does not exist.\nANSWER: UNKNOWN"),
    ]
    log = run_one_sample("unknown-SKU-1043", outputs, tmp_path)
    assert accuracy(log) == 1.0
    assert log.samples is not None
    tool_msgs = [m for m in log.samples[0].messages if isinstance(m, ChatMessageTool)]
    assert tool_msgs[0].error is not None


def test_guessing_a_number_for_unknown_sku_scores_zero(tmp_path: Path) -> None:
    log = run_one_sample("unknown-SKU-1043", [reply("ANSWER: 37")], tmp_path)
    assert accuracy(log) == 0.0

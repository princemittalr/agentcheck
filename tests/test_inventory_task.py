"""Offline tests for the inventory suite. No network: the model is Inspect's mockllm."""

import asyncio
from collections import Counter
from pathlib import Path

import pytest
from inspect_ai import eval as inspect_eval
from inspect_ai.dataset import Sample
from inspect_ai.log import EvalLog
from inspect_ai.model import ChatMessage, ChatMessageTool, ModelOutput, ModelUsage, get_model
from inspect_ai.model._model import ModelGenerateError  # private: only to simulate Groq's error
from inspect_ai.tool import ToolError

from agentcheck.run import count_model_output_errors
from agentcheck.tasks.inventory import (
    CATALOG,
    build_dataset,
    extract_answer,
    inventory_lookup,
    lookup_price,
    lookup_reorder_level,
    lookup_stock,
    model_output_error_code,
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
    assert asyncio.run(lookup_reorder_level()(sku="SKU-9317")) == "77"


@pytest.mark.parametrize("tool_factory", [lookup_stock, lookup_price, lookup_reorder_level])
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
    assert counts == {
        "single": 6,
        "value": 8,
        "total": 8,
        "most": 8,
        "unknown": 6,
        "reorder": 8,
        "portfolio": 8,
        "orders": 8,
    }


def test_hand_checked_targets() -> None:
    """Targets worked out by hand, so a bug in build_dataset can't hide behind itself."""
    targets = {s.id: s.target for s in build_dataset()}
    assert targets["single-SKU-2177"] == "0"
    assert targets["value-SKU-8266"] == "4995"  # 5 x 999
    assert targets["value-SKU-3305"] == "1236"  # 412 x 3
    assert targets["total-0"] == "114"  # 37 + 23 + 54
    assert targets["most-3"] == "SKU-5583"  # 1250 > 412 > 77
    assert targets["unknown-SKU-1043"] == "UNKNOWN"
    # reorder-3: 8<=8, 77<=77, 96<=120, 0<=10, 54<=60 -> all five (two boundary cases)
    assert targets["reorder-3"] == "5"
    # reorder-0: 37<=40 yes, 412>100, 8<=8 yes, 23>20, 5>2
    assert targets["reorder-0"] == "2"
    assert targets["portfolio-0"] == "5314"  # 444 + 1236 + 1632 + 2002
    assert targets["orders-0"] == "30"  # 37-30 filled = 7; 30 > 23 rejected = 23
    assert targets["orders-2"] == "7"  # exactly 54 of 54 is filled = 0; 77-70 = 7
    assert targets["orders-6"] == "37"  # 23 of 23 filled = 0; 38 > 37 rejected = 37


# --- suite v4 ------------------------------------------------------------------------


def test_v4_is_v3_plus_60_new_questions() -> None:
    v3 = [(s.id, s.input, s.target) for s in build_dataset("v3")]
    v4 = [(s.id, s.input, s.target) for s in build_dataset("v4")]
    assert len(v4) == 120
    assert v4[:60] == v3  # v3 untouched, so v3 logs stay comparable
    assert len({i for i, _, _ in v4}) == 120
    assert all(str(i).startswith("v4-") for i, _, _ in v4[60:])
    counts = Counter(str((s.metadata or {})["category"]) for s in list(build_dataset("v4"))[60:])
    assert counts == {
        "single": 6,
        "value": 4,
        "total": 10,
        "most": 10,
        "unknown": 6,
        "reorder": 8,
        "portfolio": 8,
        "orders": 8,
    }


def test_v4_questions_are_new() -> None:
    """No v4 question repeats a v3 question (same SKUs in any order, same category)."""

    def key(s: Sample) -> tuple[str, frozenset[str], str]:
        text = str(s.input)
        skus = frozenset(w.rstrip("?,") for w in text.split() if w.startswith("SKU-"))
        numbers = ",".join(w for w in text.split() if w.isdigit())
        return (str((s.metadata or {})["category"]), skus, numbers)

    old = {key(s) for s in build_dataset("v3")}
    new = [key(s) for s in list(build_dataset("v4"))[60:]]
    assert not old & set(new)


def test_v4_hand_checked_targets() -> None:
    targets = {s.id: s.target for s in build_dataset("v4")}
    assert targets["v4-single-SKU-9317"] == "77"
    assert targets["v4-value-SKU-7108"] == "1866"  # 311 x 6
    assert targets["v4-total-0"] == "1699"  # 37 + 412 + 1250
    assert targets["v4-most-1"] == "SKU-8890"  # 142 > 23 > 0
    # 37<=40 yes, 0<=10 yes, 8<=8 yes, 23>20 no, 77<=77 yes
    assert targets["v4-reorder-0"] == "4"
    assert targets["v4-portfolio-0"] == "3368"  # 444 + 0 + 1250 + 1674
    assert targets["v4-portfolio-1"] == "10153"  # 1236 + 1920 + 4995 + 2002
    assert targets["v4-orders-0"] == "5"  # 8 of 8 filled = 0; 6 > 5 rejected = 5
    assert targets["v4-orders-3"] == "412"  # 1250 of 1250 = 0; 413 > 412 rejected
    assert targets["v4-orders-7"] == "1286"  # 37-1 = 36; 1251 > 1250 rejected
    assert targets["v4-unknown-SKU-3350"] == "UNKNOWN"


def test_unknown_suite_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown suite"):
        build_dataset("v9")


def test_most_questions_have_no_ties() -> None:
    for s in build_dataset("v4"):
        if (s.metadata or {})["category"] == "most":
            skus = [w.rstrip("?,") for w in str(s.input).split() if w.startswith("SKU-")]
            stocks = sorted((CATALOG[k].stock for k in skus), reverse=True)
            assert stocks[0] > stocks[1], s.id


def test_unknown_skus_are_really_unknown() -> None:
    for s in build_dataset("v4"):
        if s.target == "UNKNOWN":
            sku = str(s.id).removeprefix("v4-").removeprefix("unknown-")
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


# --- model output errors (Groq "output_parse_failed") --------------------------------

PARSE_FAILED = ModelGenerateError(
    "Request: ...",
    status_code=400,
    provider_message="Error code: 400 - {'error': {'code': 'output_parse_failed'}}",
)


@pytest.mark.parametrize(
    ("ex", "expected"),
    [
        (PARSE_FAILED, "output_parse_failed"),
        (ModelGenerateError("x", status_code=429, provider_message="output_parse_failed"), None),
        (ModelGenerateError("x", status_code=400, provider_message="context too long"), None),
        (RuntimeError("output_parse_failed"), None),
    ],
)
def test_model_output_error_code(ex: BaseException, expected: str | None) -> None:
    assert model_output_error_code(ex) == expected


def test_output_parse_failure_is_scored_wrong_not_crashed(tmp_path: Path) -> None:
    calls = 0

    def model(input: list[ChatMessage], *args: object) -> ModelOutput:
        nonlocal calls
        calls += 1
        if calls == 1:
            return tool_call("lookup_stock", "SKU-1042")
        raise PARSE_FAILED

    m = get_model(MOCK, custom_outputs=model)
    [log] = inspect_eval(
        inventory_lookup(),
        model=m,
        sample_id="single-SKU-1042",
        log_dir=str(tmp_path),
        display="none",
    )
    assert log.status == "success", log.error
    assert accuracy(log) == 0.0
    assert log.samples is not None
    sample = log.samples[0]
    assert sample.error is None
    assert sample.metadata["model_output_error"] == "output_parse_failed"
    assert sample.scores is not None
    assert "output_parse_failed" in str(sample.scores["answer_line"].explanation)
    assert count_model_output_errors(log) == 1


def test_other_errors_still_fail_the_run(tmp_path: Path) -> None:
    def model(*args: object) -> ModelOutput:
        raise ModelGenerateError("x", status_code=401, provider_message="invalid_api_key")

    m = get_model(MOCK, custom_outputs=model)
    [log] = inspect_eval(
        inventory_lookup(),
        model=m,
        sample_id="single-SKU-1042",
        log_dir=str(tmp_path),
        display="none",
    )
    assert log.status == "error"

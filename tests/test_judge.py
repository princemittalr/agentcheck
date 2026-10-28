"""Tests for LLM-judge calibration, with mockllm standing in for the judge."""

import re
from pathlib import Path

import pytest
from inspect_ai import eval as inspect_eval
from inspect_ai.log import EvalLog
from inspect_ai.model import ChatMessage, Model, ModelOutput, ModelUsage, get_model

from agentcheck import judge
from agentcheck.cli import main
from agentcheck.tasks.inventory import build_dataset, inventory_lookup

MOCK = "mockllm/model"
TARGETS = {str(s.input): str(s.target) for s in build_dataset()}
IDS = {str(s.input): str(s.id) for s in build_dataset()}
SAMPLES = ["single-SKU-1042", "single-SKU-2177", "value-SKU-3305", "total-0"]


def reply(text: str) -> ModelOutput:
    out = ModelOutput.from_content(MOCK, text)
    out.usage = ModelUsage(input_tokens=0, output_tokens=0, total_tokens=0)
    return out


def agent_log(tmp_path: Path, wrong: set[str], name: str) -> EvalLog:
    def model(input: list[object], *args: object) -> ModelOutput:
        q = next(q for q in TARGETS if q in str(input))
        return reply(f"ANSWER: {'999999' if IDS[q] in wrong else TARGETS[q]}")

    [log] = inspect_eval(
        inventory_lookup(),
        model=get_model(MOCK, custom_outputs=model),
        sample_id=SAMPLES,
        log_dir=str(tmp_path / name),
        display="none",
    )
    return log


def prompt_text(input: list[ChatMessage]) -> str:
    return "\n".join(m.text for m in input)


def honest_judge(input: list[ChatMessage], *args: object) -> ModelOutput:
    text = prompt_text(input)
    target = re.search(r"Reference answer: (\S+)", text)
    assert target is not None
    if "Reply 1:" in text:
        first = text.split("Reply 1:")[1].split("Reply 2:")[0]
        return reply("1" if f"ANSWER: {target.group(1)}" in first else "2")
    agent = text.split("Agent's final reply:")[1]
    return reply("CORRECT" if f"ANSWER: {target.group(1)}" in agent else "INCORRECT")


def first_position_judge(input: list[ChatMessage], *args: object) -> ModelOutput:
    return reply("1" if "Reply 1:" in prompt_text(input) else "CORRECT")


def test_parsers() -> None:
    assert judge.parse_grade("Thinking...\nCORRECT") is True
    assert judge.parse_grade("**INCORRECT**") is False
    assert judge.parse_grade("not sure") is None
    assert judge.parse_choice("Reply 2 is right.\n2") == "2"
    assert judge.parse_choice("neither") is None


def test_honest_judge_agrees_perfectly_and_has_no_position_bias(tmp_path: Path) -> None:
    good = agent_log(tmp_path, set(), "good")
    bad = agent_log(tmp_path, {"single-SKU-1042", "total-0"}, "bad")
    items, pairs = judge.plan(bad, good)
    assert len(items) == 4
    assert len(pairs) == 2
    r = judge.judge(get_model(MOCK, custom_outputs=honest_judge), items, pairs)
    assert r.kappa is not None
    assert r.kappa.kappa == pytest.approx(1.0)
    assert r.bias is not None
    assert r.bias.consistency == 1.0


def test_lenient_position_biased_judge_is_caught(tmp_path: Path) -> None:
    good = agent_log(tmp_path, set(), "good")
    bad = agent_log(tmp_path, {"single-SKU-1042", "total-0"}, "bad")
    items, pairs = judge.plan(bad, good)
    r = judge.judge(get_model(MOCK, custom_outputs=first_position_judge), items, pairs)
    assert r.kappa is not None
    assert r.kappa.kappa == pytest.approx(0.0)  # always CORRECT: no better than chance
    assert r.bias is not None
    assert r.bias.first_both == 2
    assert r.bias.consistency == 0.0
    text = judge.render(r, "mock")
    assert "said CORRECT for 2 wrong replies" in text


def test_cli_refuses_over_budget(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    log = agent_log(tmp_path, set(), "good")
    code = main(["judge", log.location, "--model", MOCK, "--max-calls", "3", "--max-tokens", "10"])
    assert code == 2
    captured = capsys.readouterr()
    assert "exactly 4 judge calls" in captured.out
    assert "Nothing was sent" in captured.err


def test_cli_runs_with_mock_judge(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def scripted(name: str) -> Model:
        return get_model(name, custom_outputs=honest_judge)

    monkeypatch.setattr("agentcheck.cli.get_model", scripted)
    log = agent_log(tmp_path, {"total-0"}, "bad")
    code = main(
        ["judge", log.location, "--model", MOCK, "--max-calls", "10", "--max-tokens", "99999"]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "graded 4 replies" in out
    assert "kappa 1.000" in out

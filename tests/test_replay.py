"""Tests for tool record/replay."""

import asyncio
import json
from pathlib import Path
from typing import Any, cast

import pytest
from inspect_ai import eval as inspect_eval
from inspect_ai.log import EvalLog
from inspect_ai.model import ChatMessageTool, ModelOutput, ModelUsage, get_model
from inspect_ai.tool import Tool, ToolDef, ToolError, tool

from agentcheck.replay import Cassette, CassetteMissError, ToolMode, call_key, recorded
from agentcheck.run import main as run_main
from agentcheck.tasks.inventory import inventory_lookup


def make_counter() -> tuple[Tool, list[str]]:
    calls: list[str] = []

    @tool
    def weather() -> Tool:
        async def execute(city: str, unit: str = "C") -> str:
            """Current temperature.

            Args:
                city: City name.
                unit: C or F.
            """
            calls.append(city)
            if city == "Atlantis":
                raise ToolError("unknown city")
            return f"{len(calls) * 10}{unit}"  # changes every call: a non-deterministic tool

        return execute

    return weather(), calls


def call(t: Tool, **kwargs: Any) -> Any:
    return asyncio.run(cast(Any, t)(**kwargs))


def test_key_ignores_argument_order() -> None:
    assert call_key("t", {"a": 1, "b": 2}) == call_key("t", {"b": 2, "a": 1})
    assert call_key("t", {"a": 1}) != call_key("t", {"a": 2})


def test_wrapped_tool_keeps_name_description_and_parameters() -> None:
    live, _ = make_counter()
    wrapped = recorded(live, Cassette(Path("unused.json")), ToolMode.RECORD)
    a, b = ToolDef(live), ToolDef(wrapped)
    assert (a.name, a.description) == (b.name, b.description)
    assert a.parameters == b.parameters


def test_live_mode_is_the_original_tool(tmp_path: Path) -> None:
    live, _ = make_counter()
    assert recorded(live, Cassette(tmp_path / "c.json"), ToolMode.LIVE) is live


def test_record_then_replay_never_calls_the_tool(tmp_path: Path) -> None:
    path = tmp_path / "c.json"
    live, calls = make_counter()
    rec = recorded(live, Cassette(path), ToolMode.RECORD)
    first = call(rec, city="Oslo")
    assert call(rec, city="Oslo") == first  # snapshot: first result wins
    assert calls == ["Oslo"]

    live2, calls2 = make_counter()
    tape = Cassette.load(path)
    rep = recorded(live2, tape, ToolMode.REPLAY)
    assert call(rep, city="Oslo") == first
    assert calls2 == []
    assert tape.replayed_calls == 1


def test_tool_errors_are_recorded_and_replayed(tmp_path: Path) -> None:
    path = tmp_path / "c.json"
    live, _ = make_counter()
    rec = recorded(live, Cassette(path), ToolMode.RECORD)
    with pytest.raises(ToolError, match="unknown city"):
        call(rec, city="Atlantis")
    rep = recorded(make_counter()[0], Cassette.load(path), ToolMode.REPLAY)
    with pytest.raises(ToolError, match="unknown city"):
        call(rep, city="Atlantis")


def test_replay_miss_fails_loudly(tmp_path: Path) -> None:
    path = tmp_path / "c.json"
    Cassette(path).save()
    rep = recorded(make_counter()[0], Cassette.load(path), ToolMode.REPLAY)
    with pytest.raises(CassetteMissError, match="Oslo"):
        call(rep, city="Oslo")


def test_replay_without_cassette_file_fails(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        Cassette.open(tmp_path / "missing.json", ToolMode.REPLAY)


def test_record_extends_an_existing_cassette(tmp_path: Path) -> None:
    path = tmp_path / "c.json"
    recorded(make_counter()[0], Cassette(path), ToolMode.RECORD)
    call(recorded(make_counter()[0], Cassette(path), ToolMode.RECORD), city="Oslo")
    tape = Cassette.open(path, ToolMode.RECORD)
    live, calls = make_counter()
    call(recorded(live, tape, ToolMode.RECORD), city="Rome")
    assert calls == ["Rome"]
    assert len(Cassette.load(path).entries) == 2


def test_cassette_file_is_stable_json(tmp_path: Path) -> None:
    path = tmp_path / "c.json"
    call(recorded(make_counter()[0], Cassette(path), ToolMode.RECORD), city="Oslo", unit="F")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data == {
        "version": 1,
        "entries": {'weather({"city":"Oslo","unit":"F"})': {"result": "10F"}},
    }
    assert not list(tmp_path.glob("*.tmp"))


def test_wrong_version_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "c.json"
    path.write_text('{"version": 99, "entries": {}}', encoding="utf-8")
    with pytest.raises(ValueError, match="version"):
        Cassette.load(path)


# --- inside a real Inspect eval ------------------------------------------------------

MOCK = "mockllm/model"


def lookup_then_answer(log_dir: Path, **task_args: str) -> EvalLog:
    outputs = [
        ModelOutput.for_tool_call(MOCK, "lookup_stock", {"sku": "SKU-1042"}),
        ModelOutput.from_content(MOCK, "ANSWER: 37"),
    ]
    for o in outputs:
        o.usage = ModelUsage(input_tokens=0, output_tokens=0, total_tokens=0)
    [log] = inspect_eval(
        inventory_lookup(**task_args),
        model=get_model(MOCK, custom_outputs=outputs),
        sample_id="single-SKU-1042",
        log_dir=str(log_dir),
        display="none",
    )
    return log


def tool_results(log: EvalLog) -> list[str]:
    assert log.samples
    return [m.text for m in log.samples[0].messages if isinstance(m, ChatMessageTool)]


def test_eval_record_then_replay(tmp_path: Path) -> None:
    tape = tmp_path / "tape.json"
    rec = lookup_then_answer(tmp_path / "rec", tools="record", cassette=str(tape))
    assert rec.status == "success"
    assert tool_results(rec) == ["37"]

    # Edit the recording: replay must serve the cassette, not the real catalog.
    data = json.loads(tape.read_text(encoding="utf-8"))
    data["entries"]['lookup_stock({"sku":"SKU-1042"})'] = {"result": "999"}
    tape.write_text(json.dumps(data), encoding="utf-8")
    rep = lookup_then_answer(tmp_path / "rep", tools="replay", cassette=str(tape))
    assert tool_results(rep) == ["999"]


def test_eval_replay_miss_is_an_error_not_a_wrong_answer(tmp_path: Path) -> None:
    tape = tmp_path / "empty.json"
    Cassette(tape).save()
    log = lookup_then_answer(tmp_path / "rep", tools="replay", cassette=str(tape))
    assert log.status == "error"
    assert log.error is not None
    assert "no recording" in log.error.message


def test_task_needs_cassette_for_record() -> None:
    with pytest.raises(ValueError, match="cassette"):
        inventory_lookup(tools="record")


def test_runner_refuses_record_without_cassette(capsys: pytest.CaptureFixture[str]) -> None:
    code = run_main(
        ["--model", MOCK, "--tools", "record", "--max-calls", "9999", "--max-tokens", "9999999"]
    )
    assert code == 2
    assert "needs --cassette" in capsys.readouterr().err

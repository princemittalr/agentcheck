"""Tests for the log summary, on real Inspect logs produced offline by mockllm."""

from pathlib import Path

import pytest
from inspect_ai import eval as inspect_eval
from inspect_ai.log import EvalLog
from inspect_ai.model import ModelOutput, ModelUsage, get_model

from agentcheck.summary import latest_log, main, per_sample, render
from agentcheck.tasks.inventory import inventory_lookup

MOCK = "mockllm/model"


def _reply(text: str) -> ModelOutput:
    out = ModelOutput.from_content(MOCK, text)
    out.usage = ModelUsage(input_tokens=0, output_tokens=0, total_tokens=0)
    return out


def flaky_run(tmp_path: Path, epochs: int) -> EvalLog:
    """SKU-1042 (37 units) answered right on odd calls only; SKU-2177 (0 units) always wrong."""
    calls = {"single-SKU-1042": 0}

    def model(input: list[object], *args: object) -> ModelOutput:
        if "SKU-1042" in str(input):
            calls["single-SKU-1042"] += 1
            return _reply("ANSWER: 37" if calls["single-SKU-1042"] % 2 else "ANSWER: 36")
        return _reply("ANSWER: 5")

    [log] = inspect_eval(
        inventory_lookup(),
        model=get_model(MOCK, custom_outputs=model),
        sample_id=["single-SKU-1042", "single-SKU-2177"],
        epochs=epochs,
        max_connections=1,
        max_samples=1,
        log_dir=str(tmp_path),
        display="none",
    )
    return log


def test_per_sample_groups_epochs(tmp_path: Path) -> None:
    records = per_sample(flaky_run(tmp_path, epochs=4))
    assert set(records) == {"single-SKU-1042", "single-SKU-2177"}
    a, b = records["single-SKU-1042"], records["single-SKU-2177"]
    assert len(a.passes) == len(b.passes) == 4
    assert sum(a.passes) == 2
    assert a.flaky
    assert sum(b.passes) == 0
    assert not b.flaky
    assert a.category == b.category == "single"


def test_render_lists_flaky_and_always_failing(tmp_path: Path) -> None:
    log = flaky_run(tmp_path, epochs=4)
    text = render(log, per_sample(log))
    assert "single        2/8     0.250" in text
    assert "Flaky samples (pass in some epochs, not all): 1" in text
    assert "single-SKU-1042        2/4" in text
    assert "Always failing: 1" in text


def test_refuses_unsuccessful_log(tmp_path: Path) -> None:
    log = flaky_run(tmp_path, epochs=1)
    log.status = "error"
    with pytest.raises(ValueError, match="not 'success'"):
        per_sample(log)


def test_cli_reads_latest_log(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    flaky_run(tmp_path, epochs=2)
    assert latest_log(tmp_path).suffix == ".eval"
    assert main(["--log-dir", str(tmp_path)]) == 0
    assert "epochs: 2" in capsys.readouterr().out


def test_latest_log_errors_on_empty_dir(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        latest_log(tmp_path)

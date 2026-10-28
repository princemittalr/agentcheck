"""Tests for the capped runner. All offline: mockllm only."""

from collections.abc import Sequence
from importlib.metadata import entry_points
from pathlib import Path

import pytest
from inspect_ai import eval as inspect_eval
from inspect_ai.model import ChatMessage, ModelOutput, ModelUsage, get_model
from inspect_ai.model._model import ModelGenerateError  # private: only to simulate API errors
from inspect_ai.model._providers.mockllm import MockLLM  # private: see test_resume_*
from inspect_ai.tool import ToolChoice, ToolInfo

from agentcheck.run import (
    MAX_RETRIES,
    Budget,
    BudgetExceededError,
    check_budget,
    count_model_calls,
    main,
    max_calls_per_sample,
    remaining_samples,
    run_capped,
)
from agentcheck.tasks.inventory import MESSAGE_LIMIT, inventory_lookup

MOCK = "mockllm/model"


@pytest.mark.parametrize(("limit", "expected"), [(0, 0), (2, 0), (3, 1), (4, 1), (5, 2), (8, 3)])
def test_max_calls_per_sample(limit: int, expected: int) -> None:
    assert max_calls_per_sample(limit) == expected


def test_budget_arithmetic() -> None:
    b = Budget(samples=5, epochs=2, calls_per_sample=3, token_limit_per_sample=100)
    assert b.max_model_calls == 30
    assert b.max_http_requests == 30 * (1 + MAX_RETRIES)
    assert b.max_tokens == 1000


def test_check_budget_allows_exact_cap_and_refuses_one_over() -> None:
    b = Budget(samples=5, epochs=1, calls_per_sample=3, token_limit_per_sample=100)
    check_budget(b, max_calls=15, max_tokens=500)
    with pytest.raises(BudgetExceededError, match="15 model calls"):
        check_budget(b, max_calls=14, max_tokens=500)
    with pytest.raises(BudgetExceededError, match="500 tokens"):
        check_budget(b, max_calls=15, max_tokens=499)


def always_call_tool(
    input: Sequence[ChatMessage], tools: list[ToolInfo], tool_choice: ToolChoice, config: object
) -> ModelOutput:
    """A model stuck in a loop: it calls the tool forever and never answers."""
    out = ModelOutput.for_tool_call(MOCK, "lookup_stock", {"sku": "SKU-1042"})
    out.usage = ModelUsage(input_tokens=0, output_tokens=0, total_tokens=0)
    return out


def test_looping_model_is_stopped_at_the_bound(tmp_path: Path) -> None:
    budget = Budget(
        samples=2,
        epochs=2,
        calls_per_sample=max_calls_per_sample(MESSAGE_LIMIT),
        token_limit_per_sample=10_000,
    )
    model = get_model(MOCK, custom_outputs=always_call_tool)
    log = run_capped(model, budget, log_dir=str(tmp_path))
    assert log.status == "success"
    # The worst case is reached exactly, never exceeded: the bound is tight.
    assert count_model_calls(log) == budget.max_model_calls


def test_cli_refuses_over_budget_before_sending_anything(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    log_dir = tmp_path / "logs"
    code = main(
        ["--model", MOCK, "--max-calls", "1", "--max-tokens", "10", "--log-dir", str(log_dir)]
    )
    assert code == 2
    assert "REFUSED" in capsys.readouterr().err
    assert not log_dir.exists()


def test_cli_refuses_fixed_seed_with_multiple_epochs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    log_dir = tmp_path / "logs"
    args = ["--model", MOCK, "--epochs", "5", "--seed", "0", "--log-dir", str(log_dir)]
    code = main([*args, "--max-calls", "100000", "--max-tokens", "100000000"])
    assert code == 2
    assert "--seed" in capsys.readouterr().err
    assert not log_dir.exists()


# --- resuming an interrupted run -----------------------------------------------------


def test_tasks_are_registered_for_inspect() -> None:
    eps = [ep for ep in entry_points(group="inspect_ai") if ep.name == "agentcheck"]
    assert [ep.value for ep in eps] == ["agentcheck._registry"]


def test_resume_finishes_an_interrupted_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # The resumed run rebuilds "mockllm/model" from the log, so it answers with
    # mockllm's default text. Skip mockllm's token counting, which would download a
    # tiktoken encoding, to keep this test offline.
    async def no_usage(*args: object) -> None:
        return None

    monkeypatch.setattr(MockLLM, "_ensure_default_usage", no_usage)

    def fail_on_zero_stock(input: Sequence[ChatMessage], *args: object) -> ModelOutput:
        if "SKU-2177" in str(input):
            raise ModelGenerateError("x", status_code=503, provider_message="service unavailable")
        out = ModelOutput.from_content(MOCK, "ANSWER: 37")
        out.usage = ModelUsage(input_tokens=0, output_tokens=0, total_tokens=0)
        return out

    [first] = inspect_eval(
        inventory_lookup(),
        model=get_model(MOCK, custom_outputs=fail_on_zero_stock),
        sample_id=["single-SKU-1042", "single-SKU-2177"],
        max_retries=0,
        log_dir=str(tmp_path),
        display="none",
    )
    assert first.status == "error"
    assert remaining_samples(first) >= 1

    budget = ["--max-calls", "1000", "--max-tokens", "10000000"]
    code = main(["--resume", first.location, *budget, "--log-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert code == 0, out
    assert "Status: success" in out


def test_resume_respects_the_budget(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    def always_fail(*args: object) -> ModelOutput:
        raise ModelGenerateError("x", status_code=503, provider_message="service unavailable")

    [first] = inspect_eval(
        inventory_lookup(),
        model=get_model(MOCK, custom_outputs=always_fail),
        sample_id=["single-SKU-1042"],
        max_retries=0,
        log_dir=str(tmp_path),
        display="none",
    )
    code = main(["--resume", first.location, "--max-calls", "0", "--max-tokens", "0"])
    assert code == 2
    assert "REFUSED" in capsys.readouterr().err

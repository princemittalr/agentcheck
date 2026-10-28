"""Tests for `agentcheck compare`, on pure records and on real Inspect logs from mockllm."""

from pathlib import Path

import pytest
from inspect_ai import eval as inspect_eval
from inspect_ai.log import EvalLog
from inspect_ai.model import ModelOutput, ModelUsage, get_model

from agentcheck.cli import main
from agentcheck.compare import Verdict, compare_logs, compare_records
from agentcheck.summary import SampleRecord
from agentcheck.tasks.inventory import build_dataset, inventory_lookup

MOCK = "mockllm/model"


def records(passes: dict[str, list[bool]], category: str = "c") -> dict[str, SampleRecord]:
    return {k: SampleRecord(k, category, list(v)) for k, v in passes.items()}


# --- verdict logic on records --------------------------------------------------------


def test_identical_runs_are_inconclusive() -> None:
    run = records({f"s{i}": [i % 4 != 0] for i in range(60)})
    c = compare_records(run, run)
    assert c.verdict is Verdict.INCONCLUSIVE
    assert c.overall.method == "mcnemar"
    assert c.overall.p_value == 1.0
    assert c.overall.difference == 0.0


def test_ten_point_drop_single_epoch_is_a_regression() -> None:
    base = records({f"s{i}": [True] for i in range(60)})
    cand = records({f"s{i}": [i >= 6] for i in range(60)})
    c = compare_records(base, cand)
    assert c.verdict is Verdict.REGRESSION
    assert c.overall.p_value == pytest.approx(2 / 64)
    assert c.overall.difference == pytest.approx(-0.1)
    assert c.overall.baseline_rate == 1.0
    assert c.overall.candidate_rate == pytest.approx(0.9)


def test_five_flips_cannot_be_significant() -> None:
    base = records({f"s{i}": [True] for i in range(60)})
    cand = records({f"s{i}": [i >= 5] for i in range(60)})
    assert compare_records(base, cand).verdict is Verdict.INCONCLUSIVE


def test_improvement() -> None:
    base = records({f"s{i}": [i >= 10] for i in range(60)})
    cand = records({f"s{i}": [True] for i in range(60)})
    assert compare_records(base, cand).verdict is Verdict.IMPROVEMENT


def test_multi_epoch_uses_the_permutation_test() -> None:
    base = records({f"s{i}": [True] * 5 for i in range(60)})
    cand = records({f"s{i}": [True] * 5 if i >= 12 else [False] * 5 for i in range(60)})
    c = compare_records(base, cand)
    assert c.overall.method == "permutation"
    assert c.verdict is Verdict.REGRESSION
    assert c.overall.high < 0


def test_categories_are_bh_adjusted_and_flag_the_broken_one() -> None:
    base = {
        **records({f"a{i}": [True] for i in range(30)}, "a"),
        **records({f"b{i}": [True] for i in range(30)}, "b"),
    }
    cand = {
        **records({f"a{i}": [i >= 10] for i in range(30)}, "a"),
        **records({f"b{i}": [True] for i in range(30)}, "b"),
    }
    c = compare_records(base, cand)
    by_cat = {cr.category: cr for cr in c.categories}
    assert by_cat["a"].flagged
    assert not by_cat["b"].flagged
    assert by_cat["a"].p_adjusted == pytest.approx(min(1.0, 2 * by_cat["a"].test.p_value))
    assert by_cat["b"].p_adjusted == 1.0


def test_mismatched_samples_are_refused() -> None:
    with pytest.raises(ValueError, match="different samples"):
        compare_records(records({"a": [True]}), records({"b": [True]}))


# --- end to end on real logs ---------------------------------------------------------

TARGETS = {str(s.input): str(s.target) for s in build_dataset()}
IDS = {str(s.input): str(s.id) for s in build_dataset()}


def scripted_run(tmp_path: Path, wrong: set[str], name: str) -> EvalLog:
    """mockllm answers every question right except the sample ids in `wrong`."""

    def model(input: list[object], *args: object) -> ModelOutput:
        question = next(q for q in TARGETS if q in str(input))
        answer = "WRONG" if IDS[question] in wrong else TARGETS[question]
        out = ModelOutput.from_content(MOCK, f"ANSWER: {answer}")
        out.usage = ModelUsage(input_tokens=0, output_tokens=0, total_tokens=0)
        return out

    log_dir = tmp_path / name
    [log] = inspect_eval(
        inventory_lookup(),
        model=get_model(MOCK, custom_outputs=model),
        log_dir=str(log_dir),
        display="none",
    )
    return log


@pytest.fixture(scope="module")
def logs(tmp_path_factory: pytest.TempPathFactory) -> tuple[EvalLog, EvalLog, EvalLog]:
    tmp = tmp_path_factory.mktemp("logs")
    ids = sorted(IDS.values())
    base = scripted_run(tmp, set(), "base")
    same = scripted_run(tmp, set(), "same")
    worse = scripted_run(tmp, set(ids[:8]), "worse")
    return base, same, worse


def test_logs_regression(logs: tuple[EvalLog, EvalLog, EvalLog]) -> None:
    base, _, worse = logs
    c = compare_logs(base, worse)
    assert c.verdict is Verdict.REGRESSION
    assert c.overall.n_samples == 60


def test_cli_exit_codes(
    logs: tuple[EvalLog, EvalLog, EvalLog], capsys: pytest.CaptureFixture[str]
) -> None:
    base, same, worse = logs
    assert main(["compare", base.location, same.location]) == 0
    out = capsys.readouterr().out
    assert "VERDICT: INCONCLUSIVE" in out
    assert "Rerun it with --epochs 2" in out  # 1-epoch baseline: no noise estimate
    assert main(["compare", base.location, worse.location]) == 1
    out = capsys.readouterr().out
    assert "VERDICT: REGRESSION" in out
    assert "mcnemar, 60 samples" in out


def test_cli_refuses_unfinished_log(
    logs: tuple[EvalLog, EvalLog, EvalLog], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from inspect_ai.log import read_eval_log, write_eval_log

    base, same, _ = logs
    broken = read_eval_log(same.location)
    broken.status = "error"
    path = tmp_path / "broken.eval"
    write_eval_log(broken, str(path))
    assert main(["compare", base.location, str(path)]) == 2
    assert "not 'success'" in capsys.readouterr().err


def test_cli_inconclusive_with_epochs_gives_more_runs_advice(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    def two_epoch_log(name: str) -> EvalLog:
        def model(input: list[object], *args: object) -> ModelOutput:
            out = ModelOutput.from_content(MOCK, "ANSWER: 37")
            out.usage = ModelUsage(input_tokens=0, output_tokens=0, total_tokens=0)
            return out

        [log] = inspect_eval(
            inventory_lookup(),
            model=get_model(MOCK, custom_outputs=model),
            sample_id=["single-SKU-1042", "single-SKU-2177"],
            epochs=2,
            log_dir=str(tmp_path / name),
            display="none",
        )
        return log

    base, cand = two_epoch_log("a"), two_epoch_log("b")
    assert main(["compare", base.location, cand.location]) == 0
    out = capsys.readouterr().out
    assert "VERDICT: INCONCLUSIVE" in out
    assert "More runs:" in out

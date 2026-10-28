"""Tests for the offline demo logs and the Markdown report used by the GitHub Action."""

from pathlib import Path

import pytest
from inspect_ai.log import read_eval_log

from agentcheck import demo
from agentcheck.cli import main


@pytest.fixture(scope="module")
def demo_logs(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, Path]:
    out = tmp_path_factory.mktemp("demo")
    assert demo.main([str(out), "--name", "base"]) == 0
    assert demo.main([str(out), "--name", "same"]) == 0
    assert demo.main([str(out), "--name", "broken", "--wrong", "12"]) == 0
    return out / "base.eval", out / "same.eval", out / "broken.eval"


def test_demo_log_scores(demo_logs: tuple[Path, Path, Path]) -> None:
    base, _, broken = demo_logs
    for path, expected in [(base, 1.0), (broken, 48 / 60)]:
        log = read_eval_log(str(path))
        assert log.status == "success"
        assert log.results is not None
        assert log.results.scores[0].metrics["accuracy"].value == pytest.approx(expected)


def test_demo_rejects_bad_wrong_count(tmp_path: Path) -> None:
    assert demo.main([str(tmp_path), "--name", "x", "--wrong", "61"]) == 2
    assert demo.main([str(tmp_path), "--name", "x", "--wrong", "-1"]) == 2


def test_report_is_appended_as_markdown(
    demo_logs: tuple[Path, Path, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    base, same, broken = demo_logs
    report = tmp_path / "summary.md"
    report.write_text("existing\n", encoding="utf-8")
    assert main(["compare", str(base), str(broken), "--report", str(report)]) == 1
    assert main(["compare", str(base), str(same), "--report", str(report)]) == 0
    printed = capsys.readouterr().out
    text = report.read_text(encoding="utf-8")
    assert text.startswith("existing\n## AgentCheck: REGRESSION\n\n```\n")
    assert "## AgentCheck: INCONCLUSIVE" in text
    assert text.index("REGRESSION") < text.index("INCONCLUSIVE")
    assert text.count("```") == 4
    # the report holds exactly what was printed, line for line
    for line in printed.splitlines():
        assert line in text


def test_markdown_report_shape() -> None:
    from agentcheck.cli import markdown_report
    from agentcheck.compare import Verdict

    md = markdown_report(Verdict.IMPROVEMENT, "a\nb")
    assert md == "## AgentCheck: IMPROVEMENT\n\n```\na\nb\n```\n\n"

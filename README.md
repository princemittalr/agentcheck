# AgentCheck

[![CI](https://github.com/princemittalr/agentcheck/actions/workflows/ci.yml/badge.svg)](https://github.com/princemittalr/agentcheck/actions/workflows/ci.yml)

AgentCheck is a statistical decision layer for LLM agent evaluations, built on
[Inspect](https://inspect.aisi.org.uk/). It is not a new eval framework. Given eval logs
for a baseline and a candidate agent, it answers one question: **regression, improvement,
or inconclusive**, and if inconclusive, roughly how many more runs are needed to decide.

> Status: early scaffold (milestone M0). Nothing below is implemented yet.
> See [PROGRESS.md](PROGRESS.md) for current status and measured results.

## Planned usage

```
agentcheck compare base.eval cand.eval
```

## Running the task suite against a model

Put `GROQ_API_KEY=...` in `.env` (git-ignored), then:

```
python -m agentcheck.run --model groq/qwen/qwen3.8-27b --epochs 1 --max-calls 660 --max-tokens 960000
```

The runner prints the worst-case number of model calls and tokens first and refuses to
start if either exceeds the caps.

## How it works (planned)

1. Run a small agentic task suite in Inspect with several epochs per sample.
2. Pair per-sample results from the baseline and candidate logs.
3. Apply paired tests (exact McNemar for pass/fail, paired bootstrap for continuous
   scores), correct for multiple metrics (Benjamini–Hochberg), and report intervals.
4. Use power analysis and a sequential test to decide when more runs are needed.

## Statistical assumptions

To be filled in as each test is implemented. Every test will list its assumptions and
the conditions under which it fails.

## How it's tested

- Unit tests with pytest; property-based tests with Hypothesis.
- Statistics implementations are checked against SciPy/statsmodels on random inputs and
  against textbook examples.
- Calibration (planned): A/A tests for the false-alarm rate and injected regressions for
  the detection rate, each measured over many repetitions.

## Limitations

- The task suite is 60 synthetic warehouse questions in 8 categories; no statistics are implemented yet.
- No results have been measured.

## Development

```
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows PowerShell
pip install -e ".[dev,groq]"
ruff check . ; ruff format --check . ; mypy ; pytest
```

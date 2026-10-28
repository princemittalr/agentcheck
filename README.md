# AgentCheck

[![CI](https://github.com/princemittalr/agentcheck/actions/workflows/ci.yml/badge.svg)](https://github.com/princemittalr/agentcheck/actions/workflows/ci.yml)

AgentCheck is a statistical decision layer for LLM agent evaluations, built on
[Inspect](https://inspect.aisi.org.uk/). It is not a new eval framework. Given eval logs
for a baseline and a candidate agent, it answers one question: **regression, improvement,
or inconclusive**, and if inconclusive, roughly how many more runs are needed to decide.

> Status: statistics core and `compare` implemented (M3); calibration (M5) not yet run,
> so the false-alarm and detection rates are not measured yet.
> See [PROGRESS.md](PROGRESS.md) for current status and measured results.

## Usage

```
agentcheck compare base.eval cand.eval
```

Prints the verdict (REGRESSION, IMPROVEMENT or INCONCLUSIVE), the pass-rate difference
with its confidence interval and p-value, and a per-category table with
Benjamini-Hochberg adjusted p-values. Exit code 1 on a regression, 0 otherwise, 2 on bad
input (unfinished log, different tasks or samples). `agentcheck summary [log]` summarises
one log.

## Running the task suite against a model

Put `GROQ_API_KEY=...` in `.env` (git-ignored), then:

```
python -m agentcheck.run --model groq/qwen/qwen3.8-27b --epochs 1 --max-calls 660 --max-tokens 960000
```

The runner prints the worst-case number of model calls and tokens first and refuses to
start if either exceeds the caps. To summarise the newest log per category, with
samples that pass in some epochs but not others listed:

```
python -m agentcheck.summary
```

If a run is interrupted (for example by a rate limit), finish it without redoing the
completed samples; the budget check applies to what is left:

```
python -m agentcheck.run --resume logs/<interrupted>.eval --max-calls N --max-tokens N
```

## How it works (planned)

1. Run a small agentic task suite in Inspect with several epochs per sample.
2. Pair per-sample results from the baseline and candidate logs.
3. Apply paired tests (exact McNemar for pass/fail, paired bootstrap for continuous
   scores), correct for multiple metrics (Benjamini–Hochberg), and report intervals.
4. Use power analysis and a sequential test to decide when more runs are needed.

## Statistical assumptions

- **Samples are independent units; epochs are not.** Epochs of one sample share its
  difficulty, so they are averaged per sample and never resampled on their own.
- **Pairing.** Baseline and candidate must run the same samples; anything else is refused.
- **One epoch: exact McNemar** on discordant pairs. Exact, but discrete: with 5 or fewer
  discordant samples no result can reach p < 0.05, so small changes are inconclusive.
- **Several epochs: paired cluster bootstrap** (percentile interval, 10,000 resamples).
  Approximate; can under-cover with few samples, which is why calibration (M5) matters.
- **Per-category tests** are Benjamini-Hochberg adjusted (valid for independent or
  positively dependent tests). With 6-8 samples per category they have little power; they
  explain where a change happened and never change the verdict.
- **Single pass rates** use the Wilson score interval.
- A significant result is evidence of a change in this suite, not in general capability.

## How it's tested

- Unit tests with pytest; property-based tests with Hypothesis.
- Statistics implementations are checked against SciPy/statsmodels on random inputs and
  against textbook examples.
- Calibration (planned): A/A tests for the false-alarm rate and injected regressions for
  the detection rate, each measured over many repetitions.

## Limitations

- The task suite is 60 synthetic warehouse questions in 8 categories.
- False-alarm and detection rates are not measured yet (M5); until then the verdicts are
  untested in the sense that matters.
- No "how many more runs" estimate yet (M6).
- Groq's free tier limits a model to about 125 sample-epochs a day on this suite.

## Development

```
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows PowerShell
pip install -e ".[dev,groq]"
ruff check . ; ruff format --check . ; mypy ; pytest
```

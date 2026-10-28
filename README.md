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

### Calibration

```
agentcheck calibrate logs\a.eval logs\b.eval --epochs 1      # pool of one agent's logged epochs
agentcheck calibrate --synthetic --mean 0.8 --epochs 5           # synthetic pool
```

Simulates many baseline/candidate pairs by resampling a pool of logged epochs (samples
stay the unit). Reports the A/A false-alarm rate and the detection rate for an injected
pass-rate drop (default 10 points), each with a 95% Wilson interval.

### Recording and replaying tool calls

```
python -m agentcheck.run --model M --tools record --cassette tape.json --max-calls N --max-tokens N
python -m agentcheck.run --model M --tools replay --cassette tape.json --max-calls N --max-tokens N
```

`record` calls the real tools and stores each result (or tool error) under the tool name
and arguments; the first result for a call wins. `replay` answers from the cassette only
and never calls the tool; a call that was not recorded fails the run instead of being
guessed. Use it so baseline and candidate see identical tool outputs, and a tool that
changes between runs (a live API, a clock) is not mistaken for an agent regression.

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
- **Several epochs: paired sign-flip permutation test** on per-sample differences, exact
  under exchangeability (A/A false alarms <= alpha at any sample count). The reported
  interval is a paired cluster bootstrap (percentile, 10,000 resamples), which can
  under-cover with few samples; it is descriptive and does not decide the verdict.
- **Per-category tests** are Benjamini-Hochberg adjusted (valid for independent or
  positively dependent tests). With 6-8 samples per category they have little power; they
  explain where a change happened and never change the verdict.
- **Single pass rates** use the Wilson score interval.
- A significant result is evidence of a change in this suite, not in general capability.

## How it's tested

- Unit tests with pytest; property-based tests with Hypothesis.
- Statistics implementations are checked against SciPy/statsmodels on random inputs and
  against textbook examples.
- Calibration (`agentcheck calibrate`): A/A comparisons for the false-alarm rate and
  injected drops for the detection rate, simulated by resampling a fixed pool of logged
  epochs. The resampling treats each sample's pooled pass rate as its true rate, so it
  cannot reveal drift the pool never saw (provider changes, new failure modes).

## Limitations

- The task suite is 60 synthetic warehouse questions in 8 categories.
- False-alarm and detection rates are not measured yet (M5); until then the verdicts are
  untested in the sense that matters.
- No "how many more runs" estimate yet (M6).
- Record/replay covers tools with scalar results (text, numbers, booleans) keyed on exact
  arguments. An agent that calls a tool with arguments never recorded fails in replay mode;
  re-record when the suite or the tools change.
- Groq's free tier limits a model to about 125 sample-epochs a day on this suite.

## Development

```
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows PowerShell
pip install -e ".[dev,groq]"
ruff check . ; ruff format --check . ; mypy ; pytest
```

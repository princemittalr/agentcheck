# AgentCheck progress

Last updated: 2026-10-28

## Milestone status

| Milestone | Description | Status |
|---|---|---|
| M0 | Repo scaffold: pyproject, ruff, mypy, pytest, CI, README, PROGRESS | Done (2026-10-22) |
| M1 | Inspect task suite, 30+ samples, epochs >= 5 | In progress (step 5: 60 x 5 run resuming daily) |
| M2 | Statistics core, verified against SciPy/statsmodels | Done (2026-10-28): Wilson, exact McNemar, cluster bootstrap, Benjamini-Hochberg; power/sequential in M6, kappa in M7 |
| M3 | Log reader and `agentcheck compare` | Done (2026-10-28): first real verdict recorded |
| M4 | Tool record/replay | Done (2026-10-28, efcd2f3): cassette record/replay, runner `--tools/--cassette` |
| M5 | Calibration benchmark (A/A and injected regressions) | In progress (step 2: synthetic results; bootstrap verdict replaced by permutation test) |
| M6 | Power analysis and sequential stopping | Not started |
| M7 | LLM-judge calibration (kappa, position bias) | Not started |
| M8 | GitHub Action and write-up | Not started |

### M0 steps
- [x] Step 1: local scaffold (pyproject, ruff, mypy, pytest, README, PROGRESS), verified on Windows / Python 3.14.8
- [x] Step 2: GitHub repo created and pushed (princemittalr/agentcheck); CI workflow added
- [x] CI verified 2026-10-22: green on main (e7cd0ac); all 3 jobs red on a branch with a deliberately failing test (6474fc7, branch deleted)

### M1 steps
- [x] Step 1: inventory lookup task (5 samples, 1 tool, numeric-match scorer) with offline mockllm tests; verified on Windows (8 passed; mockllm demo 5 samples, accuracy 0.000 as expected)
- [x] Step 2: capped runner (`python -m agentcheck.run`), first real Groq run (steps 1 and 2 landed together in commit e7ca6c4)
- [ ] Step 3: 36-sample suite (5 categories, 2 tools, ANSWER-line scorer); 1-epoch difficulty check
- [x] Step 3 difficulty check on 3 models (see results); gpt-oss-20b dominated by Groq parse errors, the other two at ceiling
- [x] Step 4: suite v3, 60 samples (3 harder categories, reorder-level tool); difficulty check on qwen3.8-27b: 58/60
- [ ] Step 5: `agentcheck.summary` log reader (done); 60 x 5 epoch run on qwen3.8-27b, 131/300 so far, resumed daily under the 200k tokens/day limit (closes M1)

### M2 steps (tests written first)
- [x] Step 1: `wilson_interval` (stats/intervals.py) against Newcombe (1998) Table I, closed forms at k=0 and k=n, and SciPy `binomtest(...).proportion_ci(method="wilson")` on random inputs
- [x] Step 2: `paired_counts` + exact McNemar (stats/mcnemar.py), checked by hand and against SciPy `binomtest(b, b+c, 0.5)`
- [x] Step 3: paired cluster bootstrap over samples for multi-epoch runs (stats/bootstrap.py), checked against SciPy's percentile bootstrap and normal theory
- [x] Step 4: Benjamini-Hochberg adjusted p-values and rejections (stats/multiple.py), checked against SciPy `false_discovery_control`

### M3 steps
- [x] Step 1: `agentcheck compare` (verdict, CI, p-value, per-category BH table), `agentcheck` console script, bootstrap p-value; tested on mockllm logs (A/A inconclusive, 8 broken samples flagged as regression)
- [x] Step 2: first real comparison (qwen3.8-27b vs gpt-oss-20b on suite v3, 60 x 1): REGRESSION, exit code 1

### M4 steps
- [x] Step 1: `agentcheck.replay` (Cassette, `recorded` wrapper via Inspect `ToolDef`), task args `tools`/`cassette`, runner flags; tested standalone and inside real Inspect evals (replay serves an edited cassette value; a replay miss makes the run status `error`)

### M5 steps
- [x] Step 1: `agentcheck calibrate` (pool resampling, A/A false alarms, injected sample-level drop, Wilson CIs), synthetic Beta pools, `decide()` fast path proven equal to `compare_records`
- [x] Step 2a: synthetic results with the bootstrap verdict (below): multi-epoch A/A false alarms 0.054 and 0.060, over target
- [ ] Step 2b: rerun with the sign-flip permutation verdict
- [ ] Step 3: real pool: qwen3.8-27b 60 x 5 (when the daily-resumed run completes)
- [ ] Step 4: real pool: gpt-oss-20b 60 x 5 (noisy regime)

## Key design decisions

- **src/ layout**: tests run against the installed package, not the working directory, so packaging mistakes surface early.
- **hatchling build backend**: minimal config, standard PEP 621 metadata.
- **mypy strict on src and tests**: statistics code mixes counts and floats; types catch that class of bug cheaply.
- **pytest `filterwarnings = error`**: a warning (e.g. a NumPy divide-by-zero) fails the test instead of being swallowed.
- **ruff `BLE` rule**: forbids blind `except Exception`, the swallowed-exception problem from agentlens.
- **Runtime deps added only when first needed**: keeps each step's install small and each dependency justified.
- **CI matrix: Ubuntu 3.11 + 3.14, Windows 3.14**: oldest supported version, dev version, and the OS the code is written on.
- **CI runs on every push to any branch**: a broken branch is visible before a PR exists.
- **Model provider: Groq free tier**: chosen by MNV; current rate limits to be checked on Groq's own page before the first run in M1.
- **Python 3.14 locally, `requires-python >= 3.11`**: wheels for inspect_ai, numpy and scipy exist for 3.14 on Windows (checked 2026-10-22); CI also tests 3.11.
- **inspect-ai pinned `>=0.3.283,<0.4`**: API details were read from the installed 0.3.283 source; upgrades must be deliberate.
- **Offline tests via Inspect's `mockllm` with scripted outputs and explicit token usage**: zero cost, deterministic, and no tiktoken download in CI.
- **Deterministic scoring via a final `ANSWER: <value>` line** (replaced `match(numeric=True)` in M1 step 3): works for numbers, SKUs and UNKNOWN; last ANSWER line wins; normalisation ignores case, `$`, commas, markdown and trailing `.0` only. No LLM judge in the pass/fail path.
- **Five difficulty categories (single/value/total/most/unknown)**: a suite every model aces (ceiling effect) can't show a regression; `unknown` uses near-miss SKUs to test refusing to guess.
- **Targets checked by hand in tests**, not only recomputed by the code that builds them.
- **message_limit raised 8 -> 16 -> 24** for multi-lookup questions: worst case 11 calls per sample.
- **Primary model: `groq/qwen/qwen3.8-27b`**: 0 output errors and fewest calls of the 3 tool-calling models on MNV's key; gpt-oss-20b's 13/36 parse failures would swamp a 10-point regression.
- **gpt-oss-20b kept as a candidate "weaker model" regression for M5**: its failures are real failures of that agent stack.
- **Suite v3 adds harder categories (reorder, portfolio, orders)**: at 100% baseline, A/A tests are vacuous (no discordant pairs, so a false alarm is impossible) and nothing can register as an improvement.
- **Keep qwen3.8-27b primary despite 0.967 on v3**: rather than escalating task difficulty indefinitely (more tokens per run on a free tier), M5 will calibrate in two regimes: qwen (high pass rate, few flaky samples) and gpt-oss-20b (about 0.6, many flaky samples). Revisit if the 5-epoch run shows no flaky samples at all.
- **Epochs are repeated attempts on the same sample, not new samples**: 60 x 5 = 300 trials but only 60 independent units; the statistics must treat epochs as clustered within samples (M2/M3).
- **`agentcheck.summary` is descriptive only**: counts, rates, flaky and always-failing samples; no inference, so it isn't part of the statistics core.
- **Tasks registered via an `inspect_ai` entry point (`agentcheck._registry`)**: Inspect can only resume (`eval_retry`) tasks it can find by package name.
- **`--resume LOG` in the runner, with the same budget check**: resuming keeps finished samples instead of paying for them twice; the bound is computed for the remaining sample-epochs.
- **The runner loads `.env` itself** (python-dotenv, never overriding real env vars): Inspect loads it in `eval()` but not in `eval_retry()`.
- **Free-tier daily budget shapes the plan**: qwen3.8-27b allows 200,000 tokens/day (about 125 sample-epochs of suite v3). Long runs are resumed across days, and M5's calibration must resample from a fixed pool of logged epochs rather than make thousands of fresh runs.
- **max_retries 8 -> 40**: qwen3.8-27b's free-tier limit is 1,000 output tokens per minute and 8 retries ran out; rejected requests cost nothing.
- **`--seed` refused with `--epochs > 1`**: the same seed goes to every epoch, which could replay identical outputs and hide the run-to-run variance we're measuring.
- **Fictional tool data**: the model can only answer by calling the tool, so the task measures tool use, not memorised facts.
- **`ToolError` for bad tool input, nothing else caught**: the model sees a recoverable error; real bugs still fail the sample loudly.
- **Two narrow pytest warning ignores for inspect_ai internals** (tenacity deprecation, unclosed anyio stream): exact-message filters, everything else still errors.
- **`groq` as an optional extra (with `openai`)**: CI and tests never need a provider SDK or a key; inspect_ai's groq provider imports openai-SDK helpers, so both are needed.
- **First model: `groq/openai/gpt-oss-20b`**: smallest tool-calling chat model on MNV's Groq key (model list checked 2026-10-22).
- **Budget checked before sending**: worst-case calls = samples x epochs x ceil((message_limit - 2) / 2); the run is refused if that exceeds `--max-calls`. A test with an always-looping mock model hits the bound exactly.
- **`--max-calls` and `--max-tokens` are required flags**: no run without an explicit budget.
- **max_connections=1, max_retries=8** (was 2 and 3): the first 36-sample run hit Groq's free-tier limit of 8,000 tokens per minute and gave up after 3 short retries. Sending one request at a time with longer backoff waits the window out; retries are counted in the printed HTTP-request bound.
- **Infrastructure errors fail the whole run (Inspect's default `fail_on_error`)**: silently dropping errored samples would bias pass rates, because the samples that error, the long multi-lookup ones, aren't random.
- **Model-output errors are scored as wrong answers** (Groq 400 `output_parse_failed`: the model emitted a tool call Groq couldn't parse). It's the agent stack failing, so dropping it would hide exactly the kind of regression we want to catch, and retrying would condition on success and inflate the pass rate. Counted and printed separately by the runner.

- **Ownership changed 2026-10-28 (MNV's call)**: Claude writes the statistics core too, without explanations unless asked. Tests still come first and are never weakened.
- **Statistics core implemented by hand; SciPy is a test-only reference**: every formula must be explainable line by line, and SciPy (dev extra) checks it on random inputs. The package never imports SciPy at runtime.
- **Normal quantile from the stdlib (`statistics.NormalDist().inv_cdf`)**: no runtime dependency needed for z; matches SciPy's `norm.ppf` to about 1e-15.
- **Wilson score interval, not Wald, for pass rates**: Wald collapses to zero width at 0/n and n/n and overshoots [0, 1] near the ceiling (58/60 gives an upper bound of 1.012). Wilson stays inside [0, 1] and has near-nominal coverage at small n.
- **Exact bounds at the edges**: low is exactly 0.0 at k=0 and high exactly 1.0 at k=n, not a float rounding artefact such as -1e-17.

- **Exact McNemar (binomial on discordant pairs), not the chi-square version**: our discordant counts are small (often under 25), where the chi-square approximation is poor. With 5 or fewer discordant pairs no result can reach p < 0.05.
- **Two-sided p-value = min(1, 2 x lower tail)**: equals SciPy's method for p = 1/2 because the distribution is symmetric. Computed with exact integers, so no underflow or rounding at large n.
- **Runs must cover exactly the same samples**: pairing on different sample sets would silently compare different questions; mismatches are refused.
- **Exact McNemar for single-epoch runs; paired cluster bootstrap for multi-epoch runs**: with epochs, each sample is reduced to (candidate rate - baseline rate) and the samples are resampled, never the epochs. A test proves 5 copied epochs give exactly the same interval as 1.
- **Percentile bootstrap, 10,000 resamples, stdlib `random.Random(seed)`**: deterministic per seed, no NumPy at runtime. Percentile intervals can under-cover at small n; BCa is a possible upgrade if M5's calibration shows it.

- **Benjamini-Hochberg (FDR), not Bonferroni (FWER), across categories**: per-category verdicts are screening signals, so controlling the share of false flags is the right target and keeps power; valid for independent or positively dependent tests. Adjusted values are clamped to at least the raw p, because floating point can otherwise put them a hair below it (found by Hypothesis).

- **Verdict from one primary test; categories only explain**: McNemar for 1 epoch, cluster bootstrap otherwise. Per-category results never change the verdict, so splitting into categories cannot inflate false alarms on the headline.
- **Pass rate = mean over samples of per-sample rates**: each sample weighs the same whatever its epoch count.
- **Exit code 1 only for a regression**: inconclusive passes a CI gate; M6 adds "how many more runs".
- **Bootstrap two-sided p-value = 2 x min(share of resampled means <= 0, share >= 0)**: consistent with the percentile interval, used for BH on multi-epoch runs.

- **Replay miss is an infrastructure error, not a wrong answer**: inventing a tool result, or scoring the sample as failed, would mix tool coverage into the agent's pass rate.
- **First recorded result wins (snapshot)**: a non-deterministic tool returns the same value to baseline and candidate, so differences come from the agent.
- **Cassette written atomically after every new entry**: free-tier runs get interrupted; a recording survives and resumes.
- **Cassette key = tool name + JSON of arguments with sorted keys**: argument order never matters; any other difference in arguments is a different call.
- **Tool wrapper keeps the tool's name, description and parameter schema**: the model sees exactly the same tool in all three modes (tested).

- **Calibration resamples a fixed pool of logged epochs** (free-tier token limits rule out thousands of fresh runs): per sample, simulated epochs are drawn with replacement from that sample's pooled outcomes. Plug-in assumption: the pooled rate is the true rate; a pool needs >= 2 epochs per sample or every simulated run is identical.
- **Injected regression = whole samples break** (every epoch fails) with probability drop / pool pass rate, so the expected drop is exactly `drop` (tested). This models "the change broke some questions", the common real case, rather than uniform noise.
- **A/A false alarm = any non-inconclusive verdict** (two-sided, both directions count); detection = REGRESSION verdicts only.
- **2,000 bootstrap resamples inside calibration** (10,000 in `compare`): the per-comparison p-value error is about 0.005 near 0.05, small next to the Wilson CI of a 1,000-rep rate estimate.

- **Multi-epoch verdict = paired sign-flip permutation test, not the bootstrap p-value** (changed after step 2a): under A/A the runs are exchangeable, so flipping signs of per-sample differences gives the exact null distribution and false alarms <= alpha at any n. Exact enumeration up to 16 non-zero differences, else Monte Carlo with the +1 correction (Phipson & Smyth 2010). On {-1, 0, +1} differences it equals exact McNemar (tested). The bootstrap still provides the reported interval.

## Measured results

Every number here comes from a pasted run output.

| Date | Suite | Model | Samples x epochs | Result | Calls / tokens | Cost |
|---|---|---|---|---|---|---|
| 2026-10-22 | inventory v1 (5 single-lookup samples) | groq/openai/gpt-oss-20b, seed 0 | 5 x 1 | 5/5 correct | 10 calls (bound 15), 2,553 tokens | $0 (free tier) |

Attempt 2026-10-22, 36 x 1: stopped by a rate-limit error (TPM 8,000) after 10 of 36 samples, 28 calls, 9,848 tokens. No accuracy recorded from a partial run.
Attempt 2026-10-22, 36 x 1 (after the rate-limit fix): stopped by Groq 400 `output_parse_failed` after 14 of 36 samples, 38 calls, 13,688 tokens. No accuracy recorded.

| 2026-10-22 | inventory v2 (36 samples) | groq/openai/gpt-oss-20b, seed 0 | 36 x 1 | 22/36 (0.611); 13 of the 14 failures were Groq `output_parse_failed` | 86 calls (bound 252), 31,204 tokens | $0 |
| 2026-10-22 | inventory v2 (36 samples) | groq/qwen/qwen3.8-27b, seed 0 | 36 x 1 | 36/36 (1.000); 0 output errors | 72 calls (bound 252), 41,951 tokens | $0 |
| 2026-10-22 | inventory v2 (36 samples) | groq/openai/gpt-oss-120b, seed 0 | 36 x 1 | 35/36 (0.972); 0 output errors | 111 calls (bound 252), 45,414 tokens | $0 |

| 2026-10-22 | inventory v3 (60 samples) | groq/qwen/qwen3.8-27b, seed 0 | 60 x 1 | 58/60 (0.967); 0 output errors | 120 calls (bound 660), 93,630 tokens, 16.5 min | $0 |

| 2026-10-28 | inventory v3 (60 samples) | groq/openai/gpt-oss-20b, seed 0 | 60 x 1 | 26/60 (0.433); 26 Groq `output_parse_failed` scored as wrong | 153 calls (bound 660), 73,609 tokens, 8.9 min | $0 |

First `agentcheck compare` (2026-10-28): baseline qwen3.8-27b v3 60 x 1 (0.967) vs candidate gpt-oss-20b v3 60 x 1 (0.433): **REGRESSION**, difference -0.533, 95% CI [-0.667, -0.400], exact McNemar p = 4.657e-10, exit code 1. BH-flagged categories (FDR 0.05): most (p_BH 0.0417), portfolio (0.0312), total (0.0312); reorder 0.0625 not flagged. Caveat: this compares two different models on runs 6 days apart, so it is a sanity check of the pipeline, not a calibration result; many of the 26 failures are provider parse errors.

Attempt 2026-10-28, v3 60 x 5 on qwen3.8-27b (no seed): stopped by the 1,000 output-tokens-per-minute limit after 93 of 300 sample-epochs, 187 calls, 142,553 tokens. Resumed 2026-10-28: stopped by the 200,000 tokens/day limit at 131 of 300 (264 calls, 204,469 tokens in the resumed part). To be resumed on later days.

### Calibration, synthetic pools (2026-10-28, 1,000 reps, alpha 0.05, 10-point drop, seed 0)

Verdict for multi-epoch runs = **percentile bootstrap p-value** (superseded, see decisions):

| Pool (Beta mean, spread) | Pool pass rate | Epochs/run | A/A false alarms (95% CI) | Detect -0.10 (95% CI) |
|---|---|---|---|---|
| 0.95, 1 | 0.963 | 1 (McNemar) | 0/1000 = 0.000 [0.000, 0.004] | 482/1000 = 0.482 [0.451, 0.513] |
| 0.95, 1 | 0.963 | 5 (bootstrap) | 31/1000 = 0.031 [0.022, 0.044] | 873/1000 = 0.873 [0.851, 0.892] |
| 0.8, 4 | 0.800 | 1 (McNemar) | 25/1000 = 0.025 [0.017, 0.037] | 211/1000 = 0.211 [0.187, 0.237] |
| 0.8, 4 | 0.800 | 5 (bootstrap) | **54/1000 = 0.054** [0.042, 0.070] | 666/1000 = 0.666 [0.636, 0.695] |
| 0.6, 4 | 0.563 | 1 (McNemar) | 29/1000 = 0.029 [0.020, 0.041] | 161/1000 = 0.161 [0.140, 0.185] |
| 0.6, 4 | 0.563 | 5 (bootstrap) | **60/1000 = 0.060** [0.047, 0.076] | 620/1000 = 0.620 [0.590, 0.650] |

Findings: (1) exact McNemar keeps false alarms well under 5% but detects a 10-point drop
only 16-48% of the time from one epoch of 60 samples; (2) the percentile bootstrap is
anti-conservative at n = 60 (false alarms 5.4% and 6.0%), so its extra detections are
partly bought with false alarms.

Note: single runs, no intervals yet (Wilson intervals arrive in M2). n = 5 is far too small for any conclusion; 5/5 also shows the ceiling effect that motivated the harder suite in step 3.

## Open questions

- inspect_ai 0.3.283 emits a tenacity DeprecationWarning and an unclosed anyio stream warning in tests. Check after upgrades; possibly worth an upstream issue (M8 contact).
- Groq free-tier limits seen in 429 messages: gpt-oss-20b 8,000 tokens/minute; qwen3.8-27b 1,000 output tokens/minute and 200,000 tokens/day. Other limits unknown.
- Inspect prints "Control server failed to start ... no attribute 'AF_UNIX'" on Windows. Harmless (optional feature), evals run normally.
- License for the repo (MIT is a common default).

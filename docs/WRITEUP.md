# AgentCheck: is the new agent worse, or did it get unlucky?

Agent evals are noisy. The same agent, on the same questions, passes a different set of
them each run, so a 3-point drop between two runs can be a regression or nothing at all.
AgentCheck reads two [Inspect](https://inspect.aisi.org.uk/) logs, a baseline and a
candidate, and answers with one of three verdicts: **regression**, **improvement** or
**inconclusive**. When it is inconclusive it says how many more epochs would settle it.

This write-up covers what it decides, how, and how far the claims have been checked.
Every number below comes from a recorded run; the details are in [PROGRESS.md](../PROGRESS.md).

## Goals

| Target | Status |
|---|---|
| At most 5% false alarms when nothing changed (A/A) | Met in every synthetic setting tested |
| At least 80% detection of a 10-point drop | Met with 120 samples; not reachable with 60 |
| Every statistical choice defensible | Each choice and its alternative is recorded in PROGRESS.md |

## The test suite

A synthetic warehouse: fictional stock, prices and reorder levels behind two tools, so a
model can only answer by calling them. Questions range from one lookup to multi-step
totals, comparisons, reorder decisions and "UNKNOWN" for near-miss SKUs. Suite v3 has 60
questions in 8 categories; v4 adds 60 more (120). Scoring is deterministic: the last
`ANSWER: <value>` line must match the target. No LLM judge decides pass or fail.

## How a comparison is decided

1. **Pair by question.** Both logs must contain exactly the same samples; anything else is
   refused. Pairing removes question difficulty, the largest source of variance.
2. **Treat questions, not epochs, as the unit.** Five epochs of 60 questions are 60
   independent units, not 300. Epochs are averaged per question first.
3. **One epoch: exact McNemar** on the questions where exactly one side passed.
4. **Several epochs: paired sign-flip permutation test** on per-question differences. If
   nothing changed, each difference is as likely to be positive as negative, so flipping
   signs gives the exact null distribution (exact enumeration up to 16 non-zero
   differences, otherwise 10,000 Monte Carlo flips with the +1 correction).
5. **Report an interval**: a paired cluster bootstrap over questions. It is descriptive
   and does not decide the verdict.
6. **Explain by category**: per-category tests, Benjamini-Hochberg adjusted. They show
   where a change happened and never change the verdict.
7. **If inconclusive**, simulate from the baseline's own outcomes how many epochs would
   detect a 10-point drop.
8. **Optionally stop early**: plan k looks, spend alpha with the O'Brien-Fleming-type
   Lan-DeMets function, and test each look at its increment (valid by the union bound).

The first design used the bootstrap p-value for multi-epoch runs. Calibration showed it
raised 5.4% and 6.0% false alarms at 60 questions, over the target, so it was replaced by
the permutation test. That is the main thing calibration is for.

## Calibration

Thousands of fresh runs are out of reach on a free tier, so calibration resamples a pool of
logged epochs: each simulated run draws epochs per question from that question's pooled
outcomes. An A/A comparison draws both sides from the same pool. An injected regression
breaks whole questions (every epoch fails) with probability drop / pass rate, so the
expected drop is exactly the target and the damage is concentrated, as real breakages are.

Synthetic pools, 60 questions, 1,000 comparisons each, alpha 0.05, 10-point drop:

| Pool pass rate | Epochs | Test | A/A false alarms | Detection |
|---|---|---|---|---|
| 0.963 | 1 | McNemar | 0.000 [0.000, 0.004] | 0.482 [0.451, 0.513] |
| 0.963 | 5 | permutation | 0.000 [0.000, 0.004] | 0.723 [0.694, 0.750] |
| 0.800 | 1 | McNemar | 0.025 [0.017, 0.037] | 0.211 [0.187, 0.237] |
| 0.800 | 5 | permutation | 0.036 [0.026, 0.049] | 0.586 [0.555, 0.616] |
| 0.563 | 1 | McNemar | 0.029 [0.020, 0.041] | 0.161 [0.140, 0.185] |
| 0.563 | 5 | permutation | 0.041 [0.030, 0.055] | 0.552 [0.521, 0.583] |

False alarms stay under 5% everywhere. Detection at 60 questions does not reach 80%.

## Power: what it takes to see a 10-point drop

Detection rate by suite size and epochs (pool mean 0.8, 300 comparisons per cell):

| Questions | 1 epoch | 3 | 5 | 10 |
|---|---|---|---|---|
| 60 | 0.213 | 0.440 | 0.553 | 0.707 |
| 120 | 0.437 | 0.797 | 0.940 | 0.970 |
| 240 | 0.750 | 0.980 | 1.000 | 1.000 |

The number of questions matters more than the number of epochs: a change that breaks some
questions is only visible across questions, and epochs mostly remove noise within one.
The cheapest design whose whole 95% interval clears 80% is 120 × 5 (0.940
[0.907, 0.962]) for this pool, and 120 × 1 for a 0.95 pool. Hence suite v4.

Sequential looks (120 questions, 5 epochs, 1,000 reps):

| Design | A/A false alarms | Detection | Mean epochs (A/A, drop) |
|---|---|---|---|
| fixed, 1 look | 0.052 [0.040, 0.068] | 0.912 [0.893, 0.928] | 5, 5 |
| 5 looks | 0.038 [0.028, 0.052] | 0.856 [0.833, 0.876] | 4.97, 3.42 |

Five looks stop a real drop after 3.42 epochs on average (about a third cheaper) for 5.6
points of detection. The fixed design's 0.052 is what an exact 5% test should show.

## A real comparison

qwen3.8-27b (58/60) against gpt-oss-20b (26/60) on v3, one epoch each: **REGRESSION**,
difference -0.533, 95% CI [-0.667, -0.400], exact McNemar p = 4.657e-10. Flagged
categories: most, portfolio, total. This is two different models run days apart, so it
checks the pipeline end to end; it is not a calibration result.

## Can an LLM judge stand in for the scorer?

gpt-oss-120b graded gpt-oss-20b's 60 replies against the deterministic scorer: kappa
0.933, 95% CI [0.841, 1.000]. Its only errors were 2 lenient passes. Shown 32 pairs of one
right and one wrong reply in both orders, it chose consistently in all 32, so there is no
sign of position bias, though with no inconsistent pairs the test can only bound that rate
(about 11% at 95%). One judge, one graded model, 60 replies: encouraging, not general.

## Keeping runs comparable

Tool calls can be recorded to a cassette and replayed, so baseline and candidate see
identical tool outputs and a changing tool is not mistaken for an agent regression. A call
that was never recorded fails the run instead of being guessed.

## In CI

A composite GitHub Action runs the comparison, writes the report to the job summary and
fails the step on a regression; inconclusive passes. CI tests the action itself on offline
mock logs: an A/A pair must pass and a candidate with 12 of 60 questions broken must fail.

## What is not established

- **Real-pool calibration.** All calibration so far uses synthetic Beta pools. Pools from
  real logged runs (qwen3.8-27b and gpt-oss-20b) are pending; the qwen 60 × 5 run is being
  resumed daily under the free-tier token limit.
- **The plug-in assumption.** Resampling treats each question's pooled pass rate as its
  true rate, so calibration cannot reveal drift the pool never saw: provider changes, new
  failure modes, rate-limit artefacts.
- **One suite.** A significant result means a change on these questions, not in general
  capability. The suite is synthetic and narrow.
- **Discreteness.** With one epoch and 5 or fewer discordant questions, no result can be
  significant; small changes on small suites will read as inconclusive.
- **Planned looks only.** Adding looks that were not planned in advance breaks the
  false-alarm guarantee.

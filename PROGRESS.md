# AgentCheck progress

Last updated: 2026-10-22

## Milestone status

| Milestone | Description | Status |
|---|---|---|
| M0 | Repo scaffold: pyproject, ruff, mypy, pytest, CI, README, PROGRESS | In progress (step 1 of 2) |
| M1 | Inspect task suite, 30+ samples, epochs >= 5 | Not started |
| M2 | Statistics core, verified against SciPy/statsmodels | Not started |
| M3 | Log reader and `agentcheck compare` | Not started |
| M4 | Tool record/replay | Not started |
| M5 | Calibration benchmark (A/A and injected regressions) | Not started |
| M6 | Power analysis and sequential stopping | Not started |
| M7 | LLM-judge calibration (kappa, position bias) | Not started |
| M8 | GitHub Action and write-up | Not started |

### M0 steps
- [ ] Step 1: local scaffold (pyproject, ruff, mypy, pytest, README, PROGRESS)
- [ ] Step 2: GitHub repo and CI workflow that fails on test, lint or type errors

## Key design decisions

- **src/ layout**: tests run against the installed package, not the working directory, so packaging mistakes surface early.
- **hatchling build backend**: minimal config, standard PEP 621 metadata.
- **mypy strict on src and tests**: statistics code mixes counts and floats; types catch that class of bug cheaply.
- **pytest `filterwarnings = error`**: a warning (e.g. a NumPy divide-by-zero) fails the test instead of being swallowed.
- **ruff `BLE` rule**: forbids blind `except Exception`, the swallowed-exception problem from agentlens.
- **Runtime deps added only when first needed**: keeps each step's install small and each dependency justified.
- **Python 3.14 locally, `requires-python >= 3.11`**: wheels for inspect_ai, numpy and scipy exist for 3.14 on Windows (checked 2026-10-22); CI will also test 3.11.

## Measured results

None yet. Every number here will come from a pasted run output, with date, model and run count.

## Open questions

- Which free model API to use (provider and rate limits).
- License for the repo (MIT is a common default).

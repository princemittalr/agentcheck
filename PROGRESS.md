# AgentCheck progress

Last updated: 2026-10-22

## Milestone status

| Milestone | Description | Status |
|---|---|---|
| M0 | Repo scaffold: pyproject, ruff, mypy, pytest, CI, README, PROGRESS | Done (2026-10-22) |
| M1 | Inspect task suite, 30+ samples, epochs >= 5 | In progress (step 1) |
| M2 | Statistics core, verified against SciPy/statsmodels | Not started |
| M3 | Log reader and `agentcheck compare` | Not started |
| M4 | Tool record/replay | Not started |
| M5 | Calibration benchmark (A/A and injected regressions) | Not started |
| M6 | Power analysis and sequential stopping | Not started |
| M7 | LLM-judge calibration (kappa, position bias) | Not started |
| M8 | GitHub Action and write-up | Not started |

### M0 steps
- [x] Step 1: local scaffold (pyproject, ruff, mypy, pytest, README, PROGRESS), verified on Windows / Python 3.14.8
- [x] Step 2: GitHub repo created and pushed (princemittalr/agentcheck); CI workflow added
- [x] CI verified 2026-10-22: green on main (e7cd0ac); all 3 jobs red on a branch with a deliberately failing test (6474fc7, branch deleted)

### M1 steps
- [x] Step 1: inventory lookup task (5 samples, 1 tool, numeric-match scorer) with offline mockllm tests; verified on Windows (8 passed; mockllm demo 5 samples, accuracy 0.000 as expected)
- [ ] Step 2: capped runner (`python -m agentcheck.run`), first real Groq run
- [ ] Step 3: grow suite to 30+ samples across several tools; epochs >= 5

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
- **Deterministic scoring with `match(numeric=True)`**: same transcript always gets the same score; no LLM judge in the pass/fail path.
- **Fictional tool data**: the model can only answer by calling the tool, so the task measures tool use, not memorised facts.
- **`ToolError` for bad tool input, nothing else caught**: the model sees a recoverable error; real bugs still fail the sample loudly.
- **Two narrow pytest warning ignores for inspect_ai internals** (tenacity deprecation, unclosed anyio stream): exact-message filters, everything else still errors.
- **`groq` as an optional extra (with `openai`)**: CI and tests never need a provider SDK or a key; inspect_ai's groq provider imports openai-SDK helpers, so both are needed.
- **First model: `groq/openai/gpt-oss-20b`**: smallest tool-calling chat model on MNV's Groq key (model list checked 2026-10-22).
- **Budget checked before sending**: worst-case calls = samples x epochs x ceil((message_limit - 2) / 2); the run is refused if that exceeds `--max-calls`. A test with an always-looping mock model hits the bound exactly.
- **`--max-calls` and `--max-tokens` are required flags**: no run without an explicit budget.
- **max_connections=2, max_retries=3**: stay inside free-tier rate limits; retries are counted in the printed HTTP-request bound.

## Measured results

None yet. Every number here will come from a pasted run output, with date, model and run count.

## Open questions

- inspect_ai 0.3.283 emits a tenacity DeprecationWarning and an unclosed anyio stream warning in tests. Check after upgrades; possibly worth an upstream issue (M8 contact).
- Groq free-tier rate limits for gpt-oss-20b: check https://console.groq.com/settings/limits before larger runs.
- Inspect prints "Control server failed to start ... no attribute 'AF_UNIX'" on Windows. Harmless (optional feature), evals run normally.
- License for the repo (MIT is a common default).

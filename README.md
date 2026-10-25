# Known-groups validity is not enough for evaluation-awareness detectors

This repository holds the code and the results behind the paper "Known-groups validity is not
enough for evaluation-awareness detectors" (Arevalo-Alvarado and Badilla-Olivas, Agent
Evaluation Science symposium, Fall 2026). The paper itself is not in this repository. The
repository is the public release of the authors' working repository, and its history starts at
that release.

## The paper in brief

Several detectors try to flag evaluation awareness: a model that recognises that it is under
test. If two detectors measure the same property, they should agree. We ran the EvalAwareBench
benchmark (Li et al. 2026, arXiv:2605.23055) through Qwen 3.6 27B and Gemma 4 31B. The benchmark
adds test-like cues to otherwise unchanged prompts.

- **The known-groups check is weak.** Every detector we analysed flags cued prompts more often
  than plain ones. Random directions in Qwen's activations also pass this check after the same
  tuning as the activation probe.
- **Agreement among prompts with the same cues.** Self-report and third-person reach most of
  the highest Cohen's kappa that their positive rates allow. Most of their disagreement comes
  from flagging at different rates.
- **The Qwen activation probe** agrees with none of the three other detectors it can be compared
  with, among prompts with the same cues, although it passes the known-groups check.
- **A positive control.** The same probe recipe finds an implanted evaluation association in a
  model organism of Hua et al. (arXiv:2510.20487), but only narrowly ahead of random directions.

## Where to read

| Path | What it is |
|---|---|
| `reports/full/` | The full run: the analysis file, the fit reports, the figures, and the write-ups of the probe diagnosis and the organism control. |
| `reports/pilot/` | The earlier pilot run on a smaller design. |
| `docs/OVERVIEW.md` | A plain-language explainer of the experiment. |
| `PLAN.md` | The full specification of the experiment and its operations. |
| `docs/design/`, `docs/operations/` | The dated decision records. `PLAN.md` cites them by number. |
| `DATACARD.md` | The dataset that a run produces. |

The decision records state what was planned on their dates, and the design changed during the
work. What the study did is what the run data and the paper report. Where a record differs from
them, the run data and the paper are correct. The records also cite internal working documents
that are not in this repository, and code paths as of their date.

## Run data and the paper's numbers

The run data are in the W&B project
[bluedot-tais-project-convergent-validity/convergent-validity](https://wandb.ai/bluedot-tais-project-convergent-validity/convergent-validity),
as the artifact `cv-results-full` (6.3 GB, of which 4.8 GB are activations). `reports/full/`
holds the curated outputs: the analysis file, the fit reports, the figures and the write-ups.

`scripts/paper/` writes every number, table and figure that the paper reports, into `build/paper/`:

```
uv run python -c "import wandb; wandb.Api().artifact('bluedot-tais-project-convergent-validity/convergent-validity/cv-results-full:latest', type='results').download(root='data/results/runs/full')"
uv run python scripts/paper/make_numbers.py
uv run --no-project --with matplotlib==3.9.2 --with numpy python scripts/paper/make_figures.py
```

`make_numbers.py` also runs without the run data, from `reports/full/` alone. It then skips the
numbers that need the rollouts and the activations.

## Names in the code

The working title of the study was "Convergent validity of evaluation-awareness detectors".
The package name `cv`, the `cv-*` commands, the `CV_*` environment variables and the W&B names
come from that title. The W&B entity `bluedot-tais-project-convergent-validity` and the project
`convergent-validity` hold the archived run data, so the code keeps those names.

## The detectors

Five detectors, in two method families. On Gemma, the paper analyses only the three text
detectors. Many Gemma rollouts gave non-finite activations, so the paper excludes Gemma's
white-box detectors (Section 2 and Appendix A.4 of the paper).

| Detector | Family | How it decides | Decision maker |
|---|---|---|---|
| CoT | text | A judge model reads the target's reasoning trace and rules on it. | judge (`xiaomi/mimo-v2.5-pro`) |
| SR (self-report) | text | The target is asked, after the fact, whether it was evaluated. | target model |
| TP (third-person) | text | The target reads its own visible answer as a transcript and judges it. | target model |
| Probe | white-box | The target's activations are projected on an eval direction. | target activations |
| Logit monitor | white-box | The probability of a fixed eval-aware sentence, read off the target's final logits. | target logits |

## Targets and scope

| Target | Serving | Precision |
|---|---|---|
| `google/gemma-4-31b-it` | OpenRouter, fp8 endpoints only | fp8 |
| `Qwen/Qwen3.6-27B` | self-hosted on Modal (vLLM) | fp8 |

Both targets re-encode on our own Modal engine for the white-box pass, at fp8, on pinned
weights commits (`design 0031`). The head and every linear that does not tile in the fp8 block
size stay in bf16 (`design 0032`). On Gemma, 1,745 of 12,589 rollouts gave non-finite
activations, so the paper excludes Gemma's white-box detectors.

Per model, Experiment A is 200 items x 9 configurations x 3 repetitions, 5,400 rollouts.
Experiment B adds 48 stacked configurations on 50 items x 3 repetitions, 7,200 rollouts. That is
12,600 rollouts per model. Qwen produced all 12,600 and Gemma 12,589. Qwen's Experiment A
rollouts come from the pilot run, on the same serving setup and precision (`operations 0021`).
The full run cost $217.31 by the run meter, under an in-run cap of $300.

## Layout

| Path | Contents |
|---|---|
| `src/cv/` | The library. `config.py` (profiles, prices, paths), `dataset.py` (the pinned benchmark revision from Hugging Face), `stacking.py` (Experiment B), `rollout.py`, `run_cot.py`, `run_sr.py`, `run_tp.py`, `judge.py`, `merge.py`, `factors.py` (the criterion), `analysis.py`, `report.py`, `plots.py`, `meter.py` (the spend ledger), `failures.py` (the failure ledger). |
| `src/cv/serving/` | Target backends: `api.py` (OpenRouter) and `local/` (the Modal app with the vLLM engine and the re-encode engine). |
| `src/cv/whitebox/` | The white-box channel: contrastive pairs, the re-encode step, the activation store, the out-of-fold fit. |
| `src/*.py` | Thin entry points behind the console scripts. |
| `config/` | Run profiles over `base.yaml`: `pilot`, `prefull` (the paid smoke), `full`, `smoke`. |
| `notebooks/analysis.py` | The marimo analysis notebook: statistics, tables, figures, the W&B scorecard. |
| `scripts/` | The paper's numbers and figures (`scripts/paper/`), the costed budget model, the throughput and re-encode benchmarks, the pre-launch check, report builders, and the post hoc diagnostics (`scripts/exploration/`). |
| `tests/unit/` | Fast, free, offline. `tests/e2e/` hits real services and is opt-in. |
| `data/results/runs/<run_id>/` | Run output (gitignored): rollouts, detectors, the merged table, the ledgers, activations, analysis, figures. |
| `reports/` | Curated deliverables, tracked. |
| `docs/` | The explainer and the decision records. |

## Setup

```
uv sync --extra dev
```

Secrets live in a gitignored `.env` at the repository root (`CV_SECRETS_ENV` overrides the
path). The pipeline needs an OpenRouter key (`OPENROUTER_API_KEY`) for the API target and the
judge. The self-hosted leg needs Modal credentials on the machine and two Modal secrets:
`huggingface` (a Hugging Face token, for the weights) and `wandb` (the W&B key). A W&B key
(`WANDB_API_KEY`) in `.env` is optional: without it every W&B call is a no-op.

Two environment variables select the work. `CV_PROFILE` picks the run profile, with no
default. `CV_RUN` picks the run id, which names the output directory. A profile's `run:` field
can set it instead.

## Running

Every command runs from the repository root under `uv run`. The `cv-*` scripts load `.env`
themselves. A raw `modal` command does not: export first with `set -a; . ./.env; set +a`.

| Command | Purpose |
|---|---|
| `cv-validate` | The assumption gate, a few cents. Run it before you spend anything. |
| `cv-deploy-command` | Prints the `modal deploy` line for the loaded profile. Run its output. |
| `cv-generate` | The pipeline: rollouts, then the three text detectors, then the merge. Resumable. |
| `cv-whitebox` | The white-box pass: re-encode, fit, merge. Resumable. |
| `cv-judge2` | The second-judge validation pass. |
| `cv-status` | A pure read of the spend meter. Makes no API call. |

The order before the paid run:

```
CV_PROFILE=full uv run cv-validate
CV_PROFILE=full uv run cv-deploy-command        # then run the line it prints
CV_PROFILE=prefull uv run cv-generate           # the paid smoke, capped at $8
uv run python scripts/prefull_check.py
CV_PROFILE=prefull uv run cv-whitebox
```

The paid run, detached, so no terminal window can end it:

```
CV_PROFILE=full PYTHONUNBUFFERED=1 nohup caffeinate -dimsu uv run cv-generate > run.log 2>&1 &
uv run cv-status                                # in a second terminal
CV_PROFILE=full uv run cv-whitebox qwen3.6-27b  # after the rollouts exist
CV_PROFILE=full uv run cv-whitebox gemma-4-31b
CV_PROFILE=full uv run cv-judge2
uv run marimo run notebooks/analysis.py -- --run-id full
```

One Modal workspace runs every step. The workspace needs `cv-serving-full` deployed and the two
Modal secrets. Our own run split the Modal spend across two workspaces (`operations 0021`).

The rollout refuses to start when the enumerated count differs from the profile's expected
count, before the first paid call.

## Budget, failures, and resume

- **The cap.** `budget_usd` in the profile is a hard cap on the run's ledger. The ledger counts
  API tokens at the price table and GPU seconds at the published rate. Every OpenRouter request
  carries the table's price as a ceiling, so an endpoint above it is never served. A Modal
  workspace spend cap must also be set in the Modal dashboard: the in-run cap cannot see a raw
  `modal run`.
- **Failures.** A failed call is never a verdict. Every failed row goes to the run's failure
  ledger with its class, is excluded from scoring, and is retried on the next run of the same
  command.
- **Resume.** Every stage checkpoints per chunk and skips ids already on disk. Resume applies
  within a run id. The full run writes to its own `data/results/runs/full/`.

## Tests

```
uv run pytest tests/unit          # the default suite
uv run pytest tests/e2e           # opt-in, costs a few cents
```

## License and attribution

The code in this repository is released under the MIT License (`LICENSE`). The EvalAwareBench
benchmark is released under CC BY-NC 4.0, and every artifact derived from it inherits that
restriction: the test fixtures under `tests/fixtures/`, the pilot report under `reports/pilot/`,
and any run output that carries benchmark prompts or model reasoning about them. The
`LICENSE` file lists these paths. Cite the benchmark and this repository with `CITATION.cff`.

# 0014. Three-tier code architecture (modules / scripts / notebooks) and the observability Tracker seam

- **Status:** Accepted
- **Date proposed:** 2026-06-13
- **Date accepted:** 2026-06-13
- **Builds on:** `operations 0009` (resume/checkpointing, reaffirmed), `operations 0013` (hybrid
  serving, unchanged). Refactors the package layout and the W&B integration; changes no science.

## Context

The codebase had accreted run-able entrypoints (`main()` + `__main__`) intermixed with library
logic inside the package, W&B calls scattered through the steps (each detector called
`observability.sync_checkpoint`), and analysis/figures that were never published anywhere a person
browses. Three smells followed: tests had to poke the environment (`monkeypatch.delenv
("WANDB_API_KEY")`) to force offline because there was no seam to mock; the W&B run was a raw-data
dropbox (artifact only, logged before analysis/figures existed); and "where are the figures" had no
top-level answer.

## Decision

**1. Three tiers, split by domain, with one rule for the entry-point boundary.**

- `src/cv/` (**library**): all implementation, importable, no `__main__`. Pure shared types live in
  `cv/contracts.py` (never intermixed with logic).
- **entry points**: `src/generate.py`, `validate.py`, `status.py`, `judge2.py`, `deployment.py`,
  in `src/` **beside the `cv` package, not inside it**, each installing as a console script
  (`uv run cv-generate`, `cv-status`, ...). The rule: **the entry script contains the orchestration
  narrative**, meaning the top-level `run()`/`report()` body you expect to read when you open it (order the
  steps and pump events, loop the models doing rollout+detectors, render the spend table). `cv`
  keeps only the implementation primitives that narrative calls. There is no `cv.<x>.run()` the
  script merely forwards to: that was an indirection layer that added nothing (the script did
  not show the orchestration). `cv/pipeline.py` does not exist; its driver *is* `generate.py`.
- `notebooks/analysis.py`: **marimo** (science-domain): parameterised by `run_id`, reads as a
  document, computes via `cv.analysis`/`cv.plots`, and owns the consumer-facing W&B surfaces.

**Tests mirror `src/`, split unit vs e2e** (`tests/unit/` + `tests/e2e/`, each mirroring `src/`):
- `tests/unit/`: fast, free, offline, the default suite (`testpaths = tests/unit`). `tests/unit/cv/`
  mirrors the library; `tests/unit/test_generate.py` covers the driver with a fake Tracker + fake
  Steps (no W&B, no env toggling). Because the entry modules genuinely live in `src/`, testing them
  here honours the mirror rather than breaking it.
- `tests/e2e/`: runs the entry points end-to-end against real OpenRouter + W&B on the tiny `smoke`
  profile, asserts the expected `ops/*` / `exp/*` / figure panels, then self-cleans. Opt-in (costs a
  few cents): `pytest tests/e2e`. Replaces the old `smoke/` sibling.

**2. The Step abstraction (`cv.contracts.Step`) and a generic driver.** Each generation step
(`rollout`, `run_sr`, `run_tp`, `run_cot`) declares a `Step(name, run, metered, backend_warm)` and
its `run()` is a generator that **yields Events** (`Progress | Metric | Note`) and knows nothing
about W&B, the run id, or ordering. The `generate.py` driver carries zero step specifics: it orders
steps generically (`backend_warm` first so the GPU container stays contiguous, judge-only last),
runs them under the budget guard, pumps Events to the sink, and checkpoints. Adding a step is a
module + one tuple entry (`GENERATION` in `generate.py`). Rollout is no longer special-cased.

**3. The observability Tracker seam (mirrors `cv.serving.get_backend`).** A `Tracker` protocol with
a real `WandbTracker` and a `NullTracker`, selected by `get_tracker(run_id)` (W&B when enabled, null
otherwise). Steps yield Events; the driver pumps them to a Tracker; the notebook calls the same
Tracker. Because callers depend on the Tracker and never on `wandb`, **tests inject a fake Tracker**
instead of toggling the environment. W&B lives in exactly one place.

**4. Operational vs experimental metrics, on the run's live surfaces.** The **pipeline** writes
**`ops/*`** (spend, tokens, per-step progress, counts) during generation; the **notebook** writes
**`exp/*`** (d', kappa) + figures (`exp/fig/*`) as **latest-wins summary** on the stable
`cv-<run_id>` run, so re-running the notebook refreshes the live dashboard in place. The heavy raw
data stays the pipeline-owned artifact; analysis outputs stay version-controlled in git; W&B holds
the browsable latest view. No W&B Reports (a curated doc) for now.

**5. Run-id convention.** Canonical experiments use **stable** ids (`subset`, `full`) → one durable
W&B run each that the notebook keeps re-publishing to (that durability is what makes the dashboard
live). Ad-hoc/smoke runs use **timestamped** ids → throwaway W&B runs that never touch a canonical
dashboard. Timing comes from run metadata + W&B's own start/stop, not the id. (`subset-2026-06-12`
was renamed to `subset` under this rule.)

**6. The e2e tier confirms W&B then self-cleans.** Each `tests/e2e/` test runs a tiny real
pipeline/notebook on a timestamped id, asserts the expected `ops/*` / `exp/*` / figures on the W&B
run, then in fixture teardown **deletes the W&B run, its artifact, and the local run dir**. It
leaves no trace, so it is safe to run against the live project.

## Consequences

- W&B is mockable: the generation driver is unit-tested with a fake Tracker
  (`tests/unit/test_generate.py`), no env toggling. Steps import only `cv.contracts`; swapping W&B
  for any sink changes no step.
- The W&B run page is the dashboard (ops/* scorecard from the pipeline, exp/* + figure panels from
  the notebook), refreshed by re-running the notebook. Verified: the `subset` run carries 23 `exp/*`
  keys + 11 figure panels; both e2e tests pass and self-clean.
- The entry scripts now carry the orchestration narrative, so they are testable rather than untested
  glue: the driver has unit tests (`tests/unit/`), the entries have e2e tests (`tests/e2e/`), and
  `tests/unit` mirrors `src/`. The implementation primitives the narrative calls live in `cv` and
  are unit-tested there.
- Console scripts (`[project.scripts]`) map `cv-<name>` to the `src/` entry modules (installed via
  the src layout's `py-modules`); `cv` itself stays import-only (no `__main__`).

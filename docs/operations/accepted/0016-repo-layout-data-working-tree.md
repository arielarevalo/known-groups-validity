# 0016. Repo layout: `data/` is the gitignored working tree; deliverables and inputs are tracked separately

- **Status:** Accepted
- **Date proposed:** 2026-06-14
- **Date accepted:** 2026-06-14
- **Amends:** `operations 0003` (per-run namespacing; the run dir moves from
  `results/runs/<run_id>/` to `data/results/runs/<run_id>/`), `operations 0009` and
  `operations 0014` (which refer to `results/runs/<run_id>/` as the local ledger /
  run dir; the path funnels through `config.results_root()`).

> **Naming note (2026-06-21):** the example run dir `data/results/runs/subset/` is now
> `data/results/runs/pilot/` (the *subset* run was renamed *pilot*). The "subset" name is kept
> here as historical record; the live layout uses `pilot`. See `docs/STATUS.md`.

## Context

`results/` had become a dumping ground that mixed four kinds of thing under one
top-level directory the partial `.gitignore` could not cleanly separate:

- **Raw run outputs**, large and reproducible: `results/runs/<run_id>/rollouts/`,
  `detectors/`, `merged.*`, plus the derived `analysis/` and `figures/`.
- **Sourcing scratch and the scripts that make it**: `results/exploration/` held
  the WildChat sampling/enrichment scripts next to the JSONL they emit, a
  `__pycache__`, and an HTML explainer.
- **Operational scratch**: per-run `*.log` files and one-off `*.sh` helpers landed
  inside `results/runs/<run_id>/` (e.g. `pipeline.log`, `credit-guard.sh`), and
  `__pycache__` directories appeared under `results/`.
- **A reviewed input** that is not a result at all: the red-team JSON
  (`results/review/redteam_raw_2026-06-10.json`).

The `.gitignore` tried to draw the line inside `results/` (ignore
`runs/*/rollouts/responses_*.json` and `runs/*/detectors/*.jsonl`, keep the rest),
which is fragile: every new scratch pattern needs another rule, and reviewed inputs
sat in the same tree as throwaway outputs. The geometry conflated "what we
regenerate" with "what we keep", so neither was obvious.

## Decision

Split by lifecycle into a gitignored working tree and tracked deliverable/input
directories. The path change funnels through `config.results_root()`, so code reads
the new location from one accessor.

**1. `data/` is the gitignored working tree** (whole directory ignored):

- `data/results/runs/<run_id>/`: one run's raw and derived outputs:
  `rollouts/`, `detectors/`, `merged.*`, `analysis/`, `figures/`, `metrics`. This
  is the `operations 0003` run dir at its new path; it is fully reproducible from
  the `cv-results-<run_id>` artifact ledger (`operations 0009`) and a re-run.
- `data/cache/`: sourcing scratch (the JSONL the WildChat scripts emit, sampling
  intermediates, enrichment dumps).

**2. Tracked deliverable and input directories** (these are what a reader and a
re-run actually need, kept in git):

- `reports/<run>/`: the write-ups (e.g. `SUBSET_WRITEUP.md`), the figures they
  embed, and a copy of that run's `analysis.json`, so a report is self-contained
  and reviewable without the working tree.
- `scripts/exploration/`: the WildChat sourcing scripts (sampling, enrichment,
  candidate building), separated from the scratch they emit (which goes to
  `data/cache/`).
- `corpora/controls/`: the curated, reviewed control prompts and their manifests
  (a versioned input, not a regenerated output).
- `docs/review/`: the red-team JSON, moved out of `results/review/`; it is a
  reviewed input that belongs with the docs, not the results.

**3. Operational scratch is dropped from git**: `*.log`, one-off `*.sh`, and
`__pycache__`/`*.pyc` no longer belong in tracked trees. With `data/` ignored
wholesale, per-run logs and shell helpers under the working tree are ignored by
construction; the few that had leaked into `results/runs/<run_id>/` are removed.

## Alternatives considered

- **Status quo (everything under `results/`, ignore-by-pattern).** Lost: every new
  scratch pattern needs another `.gitignore` rule, and reviewed inputs share a tree
  with throwaway outputs, so "is this tracked?" is unanswerable at a glance.
- **Ignore `results/` wholesale, keep deliverables inside it.** Lost: the
  deliverables (write-ups, control corpora) would then need un-ignoring file by
  file, the same fragility inverted. Separating tracked dirs from the working tree
  by location, not by pattern, is what removes the per-file rules.
- **Keep the sourcing scripts next to their output (`results/exploration/`).** Lost:
  it co-locates tracked code with gitignored scratch, the exact mixing this record
  ends. Scripts go to `scripts/exploration/`, their output to `data/cache/`.
- **Leave the red-team JSON under `results/review/`.** Lost: it is a reviewed input,
  not a result; `docs/review/` is where reviewed inputs belong.

## Consequences

### Enables

- One rule states the working tree: `data/` is ignored, so any raw output, cache,
  log, or shell helper under it is non-tracked by construction, no per-pattern
  rules.
- The tracked surface is exactly the deliverables and inputs: `reports/<run>/`,
  `scripts/exploration/`, `corpora/controls/`, `docs/review/`. A clone carries what
  a reader and a re-run need and nothing throwaway.
- A report under `reports/<run>/` is self-contained (write-up + embedded figures +
  `analysis.json` copy), reviewable without the working tree.

### Forbids / constrains

- Code must reach the run dir through `config.results_root()`
  (`data/results/`), not a hardcoded `results/`; the old path is retired.
- Deliverables must not be written under `data/` (they would be ignored); the
  working tree must not be the home of anything meant to be tracked.

### Risks

- The run dir is gitignored, so the only durable copy of raw outputs is the
  `cv-results-<run_id>` artifact (`operations 0009`) plus a re-run. Acceptable:
  that ledger is exactly the `operations 0009` durability guarantee, and the
  tracked `analysis.json` copy under `reports/<run>/` preserves the headline numbers
  in git.
- Moving paths invalidates references in older records that name
  `results/runs/<run_id>/`; they are read through the `config.results_root()`
  accessor and remain correct as history.

## References

- `operations 0003` (per-run namespacing; run dir relocated), `operations 0009`
  (the `cv-results-<run_id>` artifact ledger that backs the now-gitignored run
  dir), `operations 0014` (the run dir as local ledger; path funnels through
  `config.results_root()`).
- `src/cv/config.py`: `results_root()` (the single accessor the new path funnels
  through), `cv_out_dir()` (`data/results/runs/<run_id>/`).
- `.gitignore`: `data/` ignored wholesale; the old per-pattern `results/runs/*`
  rules retired.
- Tracked dirs: `reports/<run>/`, `scripts/exploration/`, `corpora/controls/`,
  `docs/review/`.

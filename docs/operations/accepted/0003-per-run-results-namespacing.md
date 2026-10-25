# 0003. Namespace results per run under results/runs/<run_id>

- **Status:** Accepted
- **Date proposed:** 2026-06-03
- **Date accepted:** 2026-06-03
- **Supersedes:** n/a

## Context

Every result writer (`cv/merge.py`, `cv/analysis.py`, `cv/plots.py`,
`cv/judge2.py`, the meter) derives its path from `config.cv_out_dir()`, which
returned a single `results/` directory. A second run therefore resumed or
overwrote the first; the repo could hold only one run's outputs at a time.

The first gemma pilot already exists in the tree (collected 2026-05-31 21:09 to
2026-06-01 15:08, per the `ts` fields in its `_metrics.jsonl`) and we expect to
re-run the pilot. We need multiple runs to coexist without clobbering.

## Decision

Namespace outputs under `results/runs/<run_id>/`. `cv_out_dir()` now returns
`results/runs/<run_id>`, so all downstream paths follow with no other change.
`run_id` is `CV_RUN` if set, otherwise the profile name (`pilot`/`full`/`smoke`).

Defaulting to the profile name keeps `run_id` stable across a run's stages
(pipeline, then analysis/plots/status) and makes "re-run the same command to
resume" continue to work. Starting a fresh run that preserves the previous one is
an explicit `CV_RUN=<name>`. The existing pilot was migrated (git mv, history
preserved) to `results/runs/pilot-2026-05-31/`.

## Alternatives considered

- **Single `results/` dir (status quo).** Simplest, but a re-run overwrites or
  resumes the only slot; rejected, it is the problem.
- **Auto-timestamp a new dir per invocation.** Truly distinct runs with no
  thought, but it breaks the multi-stage model (analysis/plots/status are separate
  commands that must target the same run) and the "re-run to resume" UX. Rejected.
- **Profile-name default + `CV_RUN` override (chosen).** Stable within a run,
  resume-safe, and a named run preserves prior output.

## Consequences

### Enables
- Multiple runs coexist: the 2026-05-31 pilot stays put while new runs write
  elsewhere.
- Any stage targets a specific run with `CV_RUN=<id>` (e.g. re-analyse an
  archived run).

### Forbids / constrains
- Re-running a profile without `CV_RUN` reuses `results/runs/<profile>/` (the
  rolling working slot); to keep a run, give it a name or archive it.
- New runs are git-tracked like the first; if committing every run's outputs
  becomes heavy, gitignore `results/runs/` and keep runs in object storage / Weave
  (`operations/0001`) instead.

### Risks
- A stray `CV_RUN` with path separators could escape the run dir; `run_id()`
  sanitises `/` to `_`.

## Relationship to W&B (do not duplicate; possible supersession)

This per-run namespacing is a **local-only** mechanism, and it deliberately
mirrors the W&B run model so the two line up: `run_id` <-> W&B run `id`/`name`;
profile <-> `group`; the resolved-config snapshot <-> `wandb.init(config=...)`;
re-run-to-resume <-> `resume="allow"` (see `operations/0001`).

**Rule: do not run two parallel run-management systems.** When W&B is wired in
(with offline mode for the local half of the dual local/W&B setup), if it handles
local runs for us, retire the bespoke handling here rather than maintaining both:

- **Run identity and run-level bookkeeping** (which run am I, its config, grouping,
  comparison) are W&B's job. Keep a single identity: use the W&B run id as
  `run_id` (or feed `CV_RUN` to `wandb.init(id=...)`). Do **not** build our own run
  index, run-metadata file, or comparison tooling; that is W&B's, not ours.
- **The on-disk data files** (`merged.jsonl`, the detector JSONL, the system of
  record the analysis reads directly) are a separate call: keep them as the durable
  local store unless W&B Artifacts clearly subsume them, decided at integration
  time, not now.
- The local `_config_snapshot.yaml` stays as the minimal offline reproducibility
  record; everything beyond that (run metadata, indexing) waits for W&B rather than
  being reinvented locally.

`run_id` is the seam: today it drives `results/runs/<run_id>/`; once W&B is in, it
becomes the W&B run identity, and this record may be partly superseded by the W&B
integration record.

## References

- `cv/config.py`: `cv_out_dir()`, `run_id()`, `results_root()`.
- `docs/operations/accepted/0001-store-results-and-observability-in-wandb-weave.md`:
  Weave/JSONL storage; per-run dirs are the on-disk side of the same concern.

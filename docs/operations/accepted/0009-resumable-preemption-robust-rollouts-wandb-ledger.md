# 0009. Resumable, preemption-robust runs; W&B Artifacts as the durable progress ledger

- **Status:** Accepted
- **Date proposed:** 2026-06-05
- **Date accepted:** 2026-06-05
- **Supersedes:** extends `operations/accepted/0001` (Weave/Artifacts) and `0003` (per-run
  namespacing); hardens the rollout/detector stages for the Modal serving regime
  (`operations/accepted/0007`)

## Context

Modal preempts containers (spot/scaling), so a long target call can be killed and return an
incomplete set. Two failure modes must be survivable: (1) a single preempted batch mid-run, and
(2) loss of the local machine entirely (this already happened once during development). The serving
regime (`operations/0007`) runs target generation on Modal; the run dir
(`results/runs/<run_id>/`, `operations/0003`) is the local ledger.

State before this record: the SR/TP detectors already chunk and append per batch
(`src/cv/checkpoint.py`), so they were resumable locally; the **rollout was monolithic** (one
`.remote()`, file written only at the end), so a preemption lost the whole rollout; and durability
was end-of-pipeline only (`operations/0001`), so a lost machine lost all in-progress work. Weave
already logs every model call server-side in real time, but reconstructing "which samples are done"
from traces is not a practical resume path.

## Decision

Make every stage chunk-incremental, preemption-retried, and durably checkpointed, so a run resumes
exactly from the last recorded state on the same or a fresh machine.

1. **Chunk the rollout** like SR/TP (`serving_batch_size`, default 64): generate a chunk, append
   its samples, and re-dump `responses_<model>.json` after **every** chunk. A preemption now loses
   at most one chunk; resume skips completed `id`s.
2. **Modal preemption retries.** Both serving engine classes carry `retries=3`
   (`src/cv/serving/local/app.py`), so a preempted/failed container re-runs its batch transparently
   before the chunk is considered failed; a still-failed chunk is simply re-done on the next run.
3. **W&B Artifacts as the durable ledger.** A new `cv-results-<run_id>` Artifact version is pushed
   every `wandb_sync_every` batches (default 4) and once at the end, reusing one stable W&B run
   (`id = cv-<run_id>`, `resume="allow"`), so progress is recorded under one resumable run.
   `observability.sync_checkpoint` / `finalize_checkpoint` implement this.
4. **Restore on start.** At pipeline start, if the local run dir has no checkpoints (a fresh
   machine), `observability.restore_checkpoint` downloads the latest `cv-results-<run_id>` Artifact
   into the run dir; the existing done-`id` resume then skips everything already recorded. Weave
   remains the finest-grained server-side backstop between syncs.

All checkpoint operations are best-effort and no-ops when W&B is disabled or `WANDB_API_KEY` is
absent, so offline/test runs are unaffected.

## Alternatives considered

- **Status quo (monolithic rollout, end-of-run artifact only).** Lost: a preemption loses the
  whole rollout; a lost machine loses all in-progress work.
- **Reconstruct resume state by querying Weave traces.** Lost: Weave is for observability, not a
  resume index; tagging + querying per sample is more bespoke than versioned Artifacts for no gain.
- **Write outputs to a Modal Volume from the GPU side.** Lost: keeps durability on Modal rather
  than in W&B (the chosen observability/transport plane, `operations/0001`), and complicates the
  return path; the Artifact ledger implements the authors' requirement to "resume from what W&B reports".
- **Sync every chunk.** Lost: artifact-version churn for little gain; every-N-batches plus the
  always-on final sync, with Weave as the between-syncs backstop, is the balance. Tunable via
  `wandb_sync_every`.

## Consequences

### Enables
- Resume exactly from the last recorded state after a Modal preemption, a killed process, or a lost
  machine: rerun the same `CV_RUN` and only the missing samples regenerate.
- The rollout joins SR/TP at chunk-grain resumability; one mechanism across all GPU stages.

### Forbids / constrains
- The durable ledger is only as fresh as the last sync; up to `wandb_sync_every` batches of work can
  need re-doing after a machine loss (the underlying calls still exist in Weave). Lower the knob for
  finer durability at the cost of more artifact versions.
- A stable W&B run id per `run_id` means re-running the same `CV_RUN` continues that W&B run; a
  genuinely new run needs a new `CV_RUN` (consistent with `operations/0003`).

### Risks
- Re-dumping `responses_<model>.json` each chunk is O(samples) per chunk; negligible at our sizes,
  but for a very large single-model rollout it is a full re-serialize per chunk (acceptable; revisit
  with JSONL if it ever bites).
- Artifact restore assumes the same `run_id`; a typo'd `CV_RUN` silently starts fresh (by design).

## References
- `operations/accepted/0001` (Weave/Artifacts, extended here), `0003` (per-run namespacing), `0007`
  (Modal serving + the engines that carry `retries`), `0008` (what the activations pass stores; same
  ledger).
- `src/cv/observability.py` (`sync_checkpoint`/`restore_checkpoint`/`finalize_checkpoint`),
  `src/cv/rollout.py` (chunked + per-chunk dump), `src/cv/run_sr.py` / `run_tp.py` (periodic sync),
  `src/cv/pipeline.py` (restore on start, finalize at end), `src/cv/checkpoint.py` (done-id resume).

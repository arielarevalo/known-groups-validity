# 0034. The Gemma white-box channel keeps its fp8 re-encode; a non-finite row is a disclosed, length-biased gap, not a reason to change precision

- **Status:** Accepted
- **Date proposed:** 2026-09-14
- **Date accepted:** 2026-09-14
- **Relationships:** Extends `design 0032` (a ledgered `nan` row now has a disclosure rule and a
  stated bias). Relates-to `design 0031` (Gemma's re-encode caveat), `design 0033` (the fit counts
  the samples that miss a vector), `design 0030` D5 (the failure ledger), `operations 0013` (fp8
  serving parity).

## Context

The full run's Gemma white-box pass of 2026-09-14 re-encoded 12,589 rollouts on the fp8 build of
the pinned weights commit (`design 0031`), with the head and the misfit linears in bf16
(`design 0032`). The engine returned a non-finite vector for 1,745 rows, 14 percent, and the
driver ledgered each as `nan`. The failure rate rises with the rollout's length: 10 percent of
rows under 5,000 characters, 14 percent from 5,000 to 10,000, 31 percent from 10,000 to 15,000,
64 percent from 20,000 to 25,000, and every row above 25,000. The engine log carries no warning.
The Qwen pass of the day before returned zero such rows. The 12 short rollouts of the prefull
smoke were all finite, so the launch gate could not see the effect.

The cause is in the forward, not in the load. `design 0032` removed the load-time fault, an
uninitialized block scale, and its warm-up check passed on this build. What remains is an
activation that leaves the fp8 range somewhere in Gemma's forward, quantized per 128-block at
run time, likelier the longer the sequence. Gemma's activation outliers are large, and the fp8
matmul of transformers 5.10.2 has no overflow guard.

## Decision

1. **The Gemma white-box channel keeps the fp8 re-encode.** The precision of `design 0031`
   stands. The 10,844 finite rows carry the channel.
2. **A non-finite row is a disclosed gap of the channel.** The fit reads the finite rows and
   reports the count and the fraction of the samples that miss a vector (`design 0033`). The
   data card, the notebook and every report that shows the Gemma white-box detectors state the
   fraction and its bias toward long traces beside the result. A comparison across detectors on
   Gemma states that the white-box detectors see a shorter subset than the text detectors.
3. **No bf16 fallback, and no mixed precision.** A bf16 re-encode of the failed rows alone would
   mix two feature spaces in one probe. A bf16 re-encode of every Gemma row would be a second
   disclosed caveat of the same kind as `design 0031`, at about $40 and 9 hours, and the authors
   decided not to spend it (2026-09-14). A later run may revisit this with its own record.

## Alternatives considered

- **Re-encode every Gemma row in bf16.** Lost: it removes the gap but adds a precision that no
  target was served at, for a channel that `design 0031` already marks as a caveat. The cost is
  about $40 and 9 hours against a $300 cap with $95 left after the pass.
- **Re-encode only the non-finite rows in bf16.** Lost: the probe's feature space would then mix
  two precisions in a way that tracks length, the very bias the gap already carries.
- **Cap the rollout length the channel reads.** Lost: a cap trades a disclosed, measured gap for
  a silent one, and the long traces are the ones Experiment B stacks.

## Consequences

### Enables

- The Gemma white-box result ships with an honest denominator.
- The length bias is a measured fact the notebook can show, not a hidden selection.

### Forbids

- Any Gemma white-box figure without the missing fraction beside it.
- Any pooling of fp8 and bf16 vectors for one model.

### Risks

- A probe fitted on shorter traces may not transfer to long ones. The notebook reports the fit by
  length bucket as a check.
- The logit monitor, which runs on the same engine, carries the same gap. Its ledger says how
  large.

## References

- `docs/STATUS.md`, the banner of 2026-09-13 (the pass, the counts, the length buckets).
- `data/results/runs/full/failures/reencode_gemma-4-31b.jsonl` (the 1,745 ledgered ids).
- `src/cv/whitebox/extraction.py` (`is_finite`, the ledger reason `nan`).

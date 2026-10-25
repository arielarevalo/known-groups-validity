# 0030. Pre-register the estimator rules: the descriptive per-factor grid, the count rule for d', the floor threshold, the method families, and the failure ledger

- **Status:** Accepted
- **Date proposed:** 2026-08-30
- **Date accepted:** 2026-09-03
- **Relationships:** Refines `design 0026` D3, D5, C1, C2 and C3 (it states the numbers and the
  procedures those decisions left open). Relates-to `design 0029` (Experiment B, whose stacked
  configs the grid and the cued pool now hold), `design 0011` (the decision-maker partition),
  `design 0005` (the agreement matrix), `design 0014` (task clusters as the precision lever),
  `operations 0009` (resumable rollouts, which the failure ledger extends).

## Context

The launch audit of 2026-08-30 (`docs/full-run-audit-findings.md`, findings B3, B7, H10, H11, H17,
H18, H22, M10, M16) read the analysis code against `design 0026` before the full run spends. It
found five gaps between the accepted design and the estimators:

1. The code still scored the `{F3, F7}` criterion that `design 0026` D3 retired. Every stacked
   config of Experiment B carried a null label, so the whole stacking arm contributed nothing to
   the pooled sensitivity or the criterion agreement block.
2. The task-cluster bootstrap dropped a resample when its d' was undefined. A resample with no
   detections is evidence of no sensitivity, not a missing observation. The audit's simulation
   showed the effect: 2 hits against 0 false alarms in 600 rows each (Fisher exact p = 0.50)
   produced d' = 0.51 with an interval of [0.34, 0.79] that excluded zero.
3. The per-factor sweep, the primary sensitivity view of `design 0026` C2, carried no interval.
4. `design 0026` C3 pre-registers a floor-channel procedure but names no threshold. The only
   constant in the code (0.05) set a flag, and a flagged detector stayed in the matrix and in
   every block mean.
5. No multiplicity stance existed. The full run holds 5 detectors x 56 configs x 2 models = 560
   per-factor contrasts and 10 kappa pairs per block per model.

Two more findings bear on scoring. The report layer hard-coded three detectors, so the two
white-box detectors would have vanished from every table, and their pairs with SR and TP would have
been averaged into a bucket labelled judge-free (M16). And a failed call was persisted as an empty
verdict and treated as done on resume (B7, H22); in the pilot this manufactured 344 of SR-gemma's
2,047 positives from answerless follow-ups (finding 10).

The pilot's base rates make these rules load-bearing: CoT fired on 1.5% (Gemma) and 1.3% (Qwen) of
criterion rows, and the two white-box detectors have no measured base rate yet.

## Decision

**D1. The per-factor grid and the dose-response are descriptive.** They are look-don't-claim views
(`design 0026` D3). No multiplicity correction is applied to them, and no inferential language is
used about a single cell. Inference is reserved for two families: the pooled cued contrast (one
number per detector and model, ten in the run) and the kappa matrix with its intervals. The stance
travels with every `analysis.json` as a `CAVEATS` line.

**D2. The count rule for d'.** d' is undefined, with a stated reason, when the detector is silent,
when it fires on everything, or when one cell of a pair is zero and its pair holds fewer than
`MIN_COUNT = 5` events. The pairs are hits versus false alarms, and misses versus correct
rejections. The two raw rates always carry exact (Clopper-Pearson) 95% intervals. The correction
`(k + 0.5) / (n + 1)` applies to both rates. In the task-cluster bootstrap a resample with an
undefined d' enters at 0.0 and is never dropped; the record counts them (`boot_at_zero`). A
resample that lacks one class is dropped and counted (`boot_dropped`). Every contrast, per factor,
pooled and per cue count, carries the interval, the cluster count and the `ci_unstable` flag.

**D3. The floor threshold.** `FLOOR_POS_RATE = 0.05`, applied per agreement block on that block's
own rows (all configs; the criterion rows). A detector whose positive rate is below it is a floor
channel: it is reported by its rate with an exact interval, and it has no row or column in the
kappa matrix, the cell-majority matrix, or any block mean. The threshold is the value the pilot's
`analysis.json` already carried as the paradox diagnostic, so it was fixed before any full-run
data existed. At the run's scale (at least 5,400 rows per detector per block) it corresponds to
at least 270 positives, the region below which Cohen's kappa is base-rate degenerate.

**D4. Method families are explicit.** Every detector has a method family (`text`: CoT, SR, TP;
`whitebox`: probe, logit monitor) and a decision maker (`judge`, `target`, `whitebox`). A detector
absent from either map raises. The agreement record reports mean off-diagonal kappa for text pairs
without the judge, text pairs with the judge, white-box pairs, and text-versus-white-box pairs.
The last is the method-divergence read of `design 0026` D5. The report and plot layers derive the
detector list from the analysis record, never from a literal, and every kappa is reported with its
interval.

**D5. The failure ledger.** A failed call is never a verdict. Every failed row (an error, an empty
response, a truncated trace, a missing reasoning block, a budget stop) is recorded with its class
in the run's failure ledger, is excluded from scoring, and is retried on resume. The retry cost is
accepted. `n_rows` and `n_valid` are both reported, so the invalid share is a stated quantity of
every run.

## Alternatives considered

- **Benjamini-Hochberg across the per-factor family.** Lost: `design 0026` D3 already makes the
  grid descriptive, and adjusted intervals on it would invite the inferential reading the design
  rules out.
- **A Fisher exact gate instead of a count.** Lost: at the run's cell sizes the two agree, and a
  count is deterministic and readable in a table.
- **Keep dropping degenerate resamples.** Lost: it conditions the interval on non-degeneracy, so
  the interval can never cover zero for a thin channel (the audit's simulation).
- **A floor threshold of 0.01 or 0.10.** Lost: 0.01 leaves the paradox region inside the matrix
  at the run's row counts; 0.10 would floor SR on Qwen (8.4% on baseline) in some blocks. 0.05 is
  the pre-existing constant, chosen before the data.
- **Keep floor channels in the matrix behind a flag.** Lost: a flagged cell still enters the block
  means and the published heatmap, where a near-zero kappa reads as substantive disagreement.
- **Coerce a failed follow-up to "not aware".** Lost: it manufactures negatives and, through the
  parser, positives (finding 10); the direction of the error is unknown.

## Consequences

### Enables

- A silent or thin channel reads as undefined with wide rate intervals, never as a tight nonzero d'.
- The stacking arm scores: every F1..F7 stack is in the cued pool and in the per-factor grid.
- The white-box detectors reach every table and the divergence axis is a named number.

### Forbids

- Inferential claims about a single per-factor cell or a single cue count.
- A kappa without an interval anywhere in a report.
- A floor channel in any agreement mean.

### Risks

- CoT floors on both models at the pilot's rates. Then the matrix is SR, TP and the two white-box
  detectors, and CoT is reported as a sensitivity channel only. That outcome is the post-hoc branch
  `design 0026` C3 pre-registers, not a failure of the rule.
- The count rule leaves d' undefined for a detector with a few hits and no false alarms even when
  the pooled interval on its hit rate excludes the false-alarm rate. The rates and their intervals
  remain in the record for that reading.

## References

- `src/cv/factors.py` (the criterion), `src/cv/analysis.py` (`MIN_COUNT`, `FLOOR_POS_RATE`,
  `METHOD_FAMILY`, `DECISION_MAKER`, `CAVEATS`), `src/cv/report.py`, `src/cv/plots.py`,
  `src/cv/failures.py`, `src/cv/merge.py`.
- `tests/unit/cv/test_analysis.py`, `tests/unit/cv/test_report.py`, `tests/unit/cv/test_units.py`.
- `docs/full-run-audit-findings.md` (B3, B7, H10, H11, H17, H18, H22, M10, M16) and
  `docs/full-run-launch-readiness.md` (waves 1.3, 1.5 and 3.1).
- `design 0026` (D3, D5, D8/C1-C3), `design 0029`, `design 0011`, `design 0005`, `design 0014`.

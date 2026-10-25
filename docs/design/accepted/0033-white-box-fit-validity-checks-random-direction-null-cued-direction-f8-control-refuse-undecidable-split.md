# 0033. Read the white-box primary against three references: a random-direction null, the cued direction and the F8 control; refuse a split the fit cannot decide

- **Status:** Accepted
- **Date proposed:** 2026-09-09
- **Date accepted:** 2026-09-09
- **Relationships:** Extends `design 0018` (the probe read-out gains three validity references beside
  the logistic cross-check) and `design 0028` (the logit monitor gains the F8 control and the fold-mean
  AUROC). Relates-to `design 0026` D3 (the cued criterion and the F8 control class), `design 0030` D1
  (descriptive, no correction), `operations 0008`.

## Context

The probe's primary read is a training-free direction from sixteen contrastive prompt pairs, with
the layer and the threshold chosen out of fold (`design 0018`). Two things about that pipeline are
not visible from its AUROC alone. First, a layer sweep plus a threshold fit extracts some separation
from any direction, and the amount grows as the sample shrinks. Hua et al. (arXiv:2510.20487,
section 5.3) read their probe against random directions for that reason. Second, the contrastive
direction is built out of distribution, from explicit eval-versus-deploy framing. Whether it is the
same direction that separates the benchmark's own cues is a question the primary cannot answer
about itself.

Antonio Badilla-Olivas proposed the references below on 2026-09-09, with a refit
of the prefull run. On twelve prompts the random null reached the primary on Qwen (mean 0.89, max
1.00 against a primary of 1.00), and the cued direction was near orthogonal to the contrastive one
at the layer the fit chose. None of that is a result at twelve prompts. It is the confound the
full run has to be able to see.

The same review found that a fit over fewer than two task groups wrote every rollout as "no" at
score 0, and that the merge accepted those rows as valid decisions. It cannot happen at the full
run's scope. It can happen in a smoke, and a smoke that passes for the wrong reason is worse than
one that fails.

## Decision

The fit report of each model carries, beside the primary and the logistic cross-check:

- **A random-direction null.** Twenty-four unit directions with no signal, seed 0, each run through
  the same out-of-fold layer sweep and threshold fit as the primary. The report lists every AUROC,
  the mean and the maximum. The primary's AUROC is read against that spread.
- **The cued-direction cross-check.** The in-distribution counterpart of the primary: the mean of
  the cued rows minus the mean of the baseline rows, per layer, recomputed inside each fold from
  its training rows, fitted out of fold. The report carries its AUROC, its chosen layers, its
  agreement with the primary, and its cosine with the contrastive direction per layer.
- **The F8 positive control.** F8 rows are decided by the primary's out-of-fold rule and never
  fitted on (`design 0026` D3). The report carries their yes rate and mean z against the baseline
  and cued rows, and the AUROC of F8 against baseline.
- **The fold-mean AUROC** beside the pooled one, for the primary, the cued cross-check and the logit
  monitor. The pooled number mixes z-scores from folds that chose different layers.
- **The logistic cross-check's own AUROC**, from the signed distance to its boundary.

All of it is descriptive (`design 0030` D1). None of it changes a decision, a threshold or a layer.
The primary and its criterion stand as `design 0018` and `design 0026` state them.

The fit refuses a split it cannot decide. When any row would be left without an out-of-fold
decision, because the task-grouped split has fewer than two groups or a fold's training rows carry
one label, the fit raises and writes no detector file.

Two defects of the measurement are corrected with this record. Qwen 3.6's chat template already
ends the generation prompt with the trace opener when thinking is on, so the logit monitor's
reasoning prefix for Qwen no longer prepends a second one. Every Qwen logit-monitor score from
before 2026-09-09, the bench of 2026-09-03 included, was read after a doubled opener. And the
re-encode step writes the prompt shard before the response shard and treats a sample as stored only
when both stores hold it, so a stop between the two writes resumes the chunk instead of dropping
its prompt vectors.

## Alternatives considered

- **Report the primary alone, as before.** Lost: a small-sample AUROC without its null is not
  interpretable, and the register confound would surface only in the write-up.
- **Make the cued direction the primary.** Lost: it is supervised on the benchmark labels, so it
  cannot support the generalization claim of `design 0018`. It is a reference, not a replacement.
- **A permutation null over labels instead of random directions.** Lost: it tests a different
  question, whether the labels carry signal, and Hua et al. report the random-direction form.
- **Let a small fit write "no" everywhere and rely on the reader.** Lost: the merge cannot tell a
  silent "no" from a real one, and a smoke would pass on it.

## Consequences

- `whitebox/fit_<model>.json` gains `random_direction_null`, `cued_direction_cross_check`,
  `f8_positive_control`, `auroc_fold_mean`, `n_usable_samples_without_vectors`, and an AUROC under
  `logistic_cross_check`. The `cv-whitebox` start line names the null's maximum and the F8 yes rate.
- The write-up reads the probe's AUROC against the null's spread and states the cosine between the
  cued and the contrastive directions at the chosen layer. A primary inside the null's spread is
  reported as such.
- A smoke with fewer than two task groups fails at the fit, by design. The `prefull` profile has
  three tasks and passes.
- Qwen's logit-monitor scores of the prefull run were rescored on 2026-09-09 with the corrected
  prefix. The bench figures of 2026-09-03 for the logit monitor stand as throughput, not as scores.
- Cost: the null runs twenty-four fits per model on the laptop, seconds at the full run's scope.

## References

- `src/cv/whitebox/fit.py` (`random_null`, `cued_direction`, `f8_control`, `auroc_fold_mean`,
  `out_of_fold`), `src/cv/whitebox/envelope.py` (`thought_open`), `src/cv/whitebox/reencode.py`,
  `src/whitebox.py`, `tests/unit/cv/test_whitebox.py`.
- Hua, Qin, Marks and Nanda, arXiv:2510.20487, section 4.1 and section 5.3.
- `design 0018`, `design 0026`, `design 0028`, `design 0030`, `operations 0008`.

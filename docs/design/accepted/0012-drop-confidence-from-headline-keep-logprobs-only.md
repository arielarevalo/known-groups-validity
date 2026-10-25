# 0012. Drop elicited confidence from the v1 headline; keep logprobs (gemma-only) as a recorded extra

- **Status:** Accepted: confidence dropped from v1; subsumed by the construct revision
  in docs/design/accepted/0016. Amended by `design 0023`: the elicited Likert is removed outright
  (the descriptive-sidecar compromise here is retired); the logprobs-where-available auxiliary
  decided below stands.
- **Date proposed:** 2026-06-02
- **Date accepted:** 2026-06-04
- **Supersedes:** n/a

## Context

DDR 0001 cut meta-d'; DDR 0008 kept same-call Likert confidence collection
and limited its use to descriptive calibration. The pilot showed Likert
confidence is **empirically degenerate** on gemma-4-31b
(`PILOT_SUMMARY.md` §6): every detector's confidence clusters at levels 3
and 4 on a 4-point scale (no 1s, no 2s), with monotonicity rho near zero
across detectors (0.07, -0.13, 0.18, 0.14).

This means:

1. Calibration cannot be *assessed* in this regime, not because the design
   is wrong but because the underlying signal has no variance. There are
   no low-confidence trials against which to define the high-end of a
   reliability curve. ECE on a two-bin histogram is uninformative.
2. The confidence channel as currently elicited carries no usable
   signal at this model scale. Headlining a "per-detector calibration" output that exists
   only as a flat bar at confidence-4 invites the reader to misinterpret
   the absence of variance as the absence of bias.
3. The Step-0 validator (`cv/validate.py`) already includes a "confidence
   histogram" check whose threshold ("no single level >70-80% mass") is
   demonstrably exceeded on gemma-4-31b. The pilot ran anyway because the
   check is a warning, not a hard gate; the full run on three target
   models will almost certainly hit the same pattern on mistral-small and
   qwen3.6 as well.

Separately: of the four target models on OpenRouter, **only gemma exposes
logprobs**. The other three (mistral-small-2603, qwen3.6-27b) return no
logprob data per Step-0 validation. So a logprob-derived confidence signal
exists only on gemma, and only for SR and TP (which the target answers);
not for CoT or PG (which the judge answers, and mimo also exposes no
logprobs).

## Decision (proposed)

1. **Drop "per-detector calibration" from the three headline v1 outputs.**
   The two remaining headline outputs are per-detector sensitivity
   (DDR 0001/0002) and the inter-detector kappa agreement matrix
   (DDR 0005). Calibration moves to a "secondary, descriptive only"
   section in the writeup. The PLAN.md "three v1 outputs" framing is
   updated to "two co-primary outputs plus a descriptive calibration
   report".
2. **Stop calling the elicited Likert "confidence" in v1 figures.** It is
   still collected (DDR 0008) and still reported in the descriptive
   section, but the figure axis label and table column are renamed
   "elicited Likert" rather than "confidence", because the empirical
   distribution shows it does not function as a confidence measure on
   these models.
3. **Promote per-token logprobs to a recorded auxiliary channel on gemma
   only.** Where the SR or TP target call returns logprobs for the
   yes/no answer token, store the per-token logprob alongside the binary
   decision. The "calibration of logprob-confidence vs accuracy" is then
   computable on gemma-SR and gemma-TP only. It is reported as a
   gemma-only descriptive note with explicit n and CIs, never as a
   cross-detector claim.

## Alternatives considered

- **Keep the calibration headline; report ECE despite degeneracy.**
  Rejected: ECE on a two-bin distribution is uninformative; reporting it
  invites misinterpretation.
- **Replace Likert with a 0-100 percentage-confidence prompt.**
  Considered. Higher granularity might break the clustering on some
  models, but there is no guarantee, and changing the elicitation prompt
  forces re-validation against the existing pilot data. Worth piloting on
  one model before the full run, but a separate DDR if attempted.
- **Add a second-call "are you sure?" confidence elicitation to break
  the same-pass degeneracy.** Rejected for v1: doubles target-call cost,
  unclear empirical gain, and DDR 0008 already records the constraint
  that meta-d' cannot return without this change.
- **Drop confidence collection entirely.** Considered. Lower-impact than
  keeping it as a descriptive sidecar at zero marginal cost. Reject only
  if the data hygiene cost (one more column in the merged table) outweighs
  the descriptive value.

## Consequences

### Enables
- A v1 results section that does not invite the reader to look at a flat
  calibration curve and conclude "calibration is fine".
- A clean gemma-only logprob-confidence sub-analysis that has real
  variance and can support a small calibration claim.

### Forbids
- Any v1 cross-detector calibration comparison (the Likert is degenerate;
  the logprob is gemma-only).
- Treating the calibration figure as a headline result.

### Risks
- Cross-model calibration is not assessable. Mitigation: the pilot data
  already showed this; this DDR makes the framing honest.
- The gemma-only logprob sub-analysis is unbalanced (one model, two
  detectors). Mitigation: report it as descriptive, not as a primary v1
  output; flag for the full run.

## References

- `PLAN.md` lines 132-146 ("Analysis - three outputs..." block 3:
  Per-detector calibration).
- `results/PILOT_SUMMARY.md` §6 (degenerate confidence histogram on
  gemma).
- `cv/validate.py` (the confidence-histogram check, currently a warning).
- DDR 0001 (meta-d' out); DDR 0008 (same-pass elicitation in v1).

# 0001. Drop meta-d' / Type-2 SDT framing from v1

- **Status:** Accepted
- **Date proposed:** 2026-05-12
- **Date accepted:** 2026-05-15
- **Supersedes:** n/a

## Context

The v0 proposal headlined a **convergent-validity** analysis built on **meta-d'**
(Type-2 SDT) as the cross-channel comparability metric. An internal review by
the authors, assisted by LLM critique, from five angles (psychometrics,
SDT/metacognition, causal inference, LLM-eval, and a plain-language skeptic)
identified four structural problems:

1. **Same-pass elicited confidence is not a separable second-order read.** The
   detector's binary decision and its Likert confidence come from one call, one
   sampling step. The Type-1 and Type-2 distributions are then the same
   distribution under different binnings. Meta-d' reduces to a re-expression of
   d', and M-ratio is near-tautological for the judge-mediated detectors.
2. **M-ratio is not commensurable across detectors with different
   decision-makers.** SR/TP are answered by the target model; CoT/PG by the
   judge. The cogsci interpretation of meta-d' assumes a single agent. Comparing
   M-ratios across detectors with different agents is comparing different
   physical quantities.
3. **Likert clustering on 27-31B target models structurally under-identifies
   the Type-2 ROC.** The expected confidence histogram for small open models is
   degenerate (no low-confidence trials), so the meta-d' estimator fits
   pathological data.
4. **Manufactured Type-1 ground truth was being discarded.** EvalAwareBench
   sets `config_name` by construction, giving a clean recognition criterion.
   The meta-d' design replaced that with a behavioural-deviation proxy and then
   asked "does confidence predict behaviour", measuring something other than
   "can the detector tell eval from non-eval".

The empirical confirmation arrived in the gemma-4-31b pilot
(`results/PILOT_SUMMARY.md` §6): every detector's elicited confidence
clusters at Likert 3-4, no 1s or 2s, with monotonicity rho near zero (0.07,
-0.13, 0.18, 0.14). Meta-d' on this data would have been a degenerate fit.

## Decision

Cut meta-d' and the Type-2 SDT framing from v1 entirely. The v1 sensitivity
metric is **Type-1 d'** against the cue ground truth (see DDR 0002), reported
alongside hit rate, false-alarm rate, and a model-free AUROC. Cross-channel
comparison is anchored on this common Type-1 truth, not on a per-observer
Type-2 statistic.

Confidence is still collected (DDR 0008) but read only as input to the
calibration output, never to compute a metacognitive sensitivity.

## Alternatives considered

- **Keep meta-d' but reframe each channel's interpretation.** Rejected: the
  decision-maker asymmetry (item 2) is structural, not a writeup problem.
- **Move confidence to a separate call to recover Type-2 separability.**
  Rejected for v1: doubles the per-rollout call count and triples judge cost
  for a metric the manufactured ground truth makes redundant.
- **Keep meta-d' as a secondary, descriptive output.** Rejected: would invite
  reviewers to read it as the headline anyway, and the pilot's degenerate
  confidence makes the estimate uninformative.

## Consequences

### Enables
- Direct sensitivity ranking against ground truth (d', AUROC) without invoking
  a calibration model.
- Cross-detector comparison legitimised because the anchor is common ground
  truth, not a per-observer statistic.

### Forbids
- Any "metacognitive efficiency" claim about the target model.
- Any cross-detector confidence-quality comparison framed as metacognition.

### Risks
- Loses a calibration-vs-sensitivity decomposition reviewers familiar with the
  Cacioli/Servajean line might expect. Mitigation: keep the descriptive
  calibration output (DDR 0008) and explain in the writeup why meta-d' is
  out.

## References

- `PLAN.md` lines 389-409 ("Design history (why v1 is lean)").
- `results/PILOT_SUMMARY.md` §6 (degenerate confidence histogram).
- Cacioli (2026) `arXiv:2603.25112`; Servajean & Servajean (2026)
  `arXiv:2603.29693`: the meta-d'-on-LLMs precedents this DDR diverges from.
- DDR 0002 (replaces the behavioural-deviation criterion with config-name
  ground truth).
- DDR 0008 (same-pass confidence is acceptable in v1 precisely because
  meta-d' is out).

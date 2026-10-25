# 0005. Promote inter-detector kappa agreement to co-primary output

- **Status:** Accepted
- **Date proposed:** 2026-05-16
- **Date accepted:** 2026-05-17
- **Supersedes:** n/a

## Context

The v0 design had two primary outputs (per-detector meta-d' sensitivity and a
joint detector-by-behaviour correspondence surface) and demoted the
**inter-detector agreement matrix** to a secondary descriptive result.

With meta-d' cut (DDR 0001) and the behavioural-deviation criterion cut
(DDR 0002), the agreement matrix is structurally the load-bearing
convergent-validity output. The research question, restated, is: *do the four
detectors agree about which rollouts are eval-aware, conditional on a common
ground truth?* That question is answered by the 4x4 kappa matrix, not by any
per-detector sensitivity statistic.

The pilot bears this out. On gemma-4-31b
(`PILOT_SUMMARY.md` §4, clean-criterion kappa):

|     | CoT | SR | TP | PG |
|-----|-----|----|----|----|
| CoT | 1   | 0  | 0  | 0  |
| SR  | 0   | 1  | 0.39 | 0.17 |
| TP  | 0   | 0.39 | 1 | 0.03 |
| PG  | 0   | 0.17 | 0.03 | 1 |

The single non-zero is **SR-TP** (kappa 0.39 clean, 0.55 over all configs).
SR and TP are the two detectors whose decision-maker is the target model.
CoT and PG are judge-mediated and their off-diagonals are kappa-paradox
zeros (very low positive rates). The matrix is doing real work: it shows
the only cross-method agreement is intra-decision-maker.

## Decision

The inter-detector agreement matrix is a **co-primary output** for v1
alongside per-detector sensitivity (DDR 0001/0002). The headline kappa
panel reports:

- Pairwise Cohen's kappa, 4x4, on the cue-criterion subset.
- Asymmetric conditional probabilities P(B detects | A detects) alongside
  kappa, because the detectors are not symmetric (judge-mediated vs
  target-introspective).
- Per-model, not pooled.

`cv/analysis.py` emits the agreement matrix at the same status level as the
d' table; `cv/plots.py` renders it as a co-primary figure.

## Alternatives considered

- **Keep agreement as a secondary descriptive output.** Rejected: with
  meta-d' out, agreement is the convergent-validity result; demoting it
  would leave the study with no claim about cross-detector convergence at
  all.
- **Report only kappa (no asymmetric conditionals).** Rejected: the
  detectors are structurally asymmetric (different decision-makers,
  different positive rates), and the conditionals carry information kappa
  hides.
- **Pool kappa across models.** Rejected for v1: the pilot's per-model
  matrix is informative; pooled kappa hides model-specific behaviour.

## Consequences

### Enables
- A direct headline answer to the convergent-validity question.
- A surface for the decision-maker asymmetry to show up empirically (the
  intra-decision-maker SR-TP cell vs the cross-cell zeros).

### Forbids
- Reading any single detector's sensitivity ranking as the headline. The
  cross-detector agreement structure is co-primary.

### Risks
- The CoT and PG kappa cells are uninformative when positive rates are very
  low (kappa paradox), as the pilot shows. The reader has to be told this
  explicitly; the matrix panel should annotate near-zero positive rates.
  DDR 0011 proposed addresses partitioning the matrix by decision-maker so
  the structural pattern is legible.

## References

- `cv/analysis.py` (kappa + conditional-probability computation).
- `cv/plots.py` (kappa heatmap).
- `results/PILOT_SUMMARY.md` §4 (pilot kappa matrix).
- DDR 0001 (meta-d' out); DDR 0011 proposed (decision-maker partition of the
  agreement matrix panel).

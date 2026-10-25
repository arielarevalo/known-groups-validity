# 0011. Partition the kappa agreement matrix by decision-maker

- **Status:** Accepted: implemented as `cv.analysis._decision_maker_blocks` (Phase A reanalysis).
- **Date proposed:** 2026-06-02
- **Date accepted:** 2026-06-04
- **Supersedes:** n/a

## Context

DDR 0005 made the 4x4 inter-detector kappa matrix a co-primary output. The
pilot panel (clean criterion, gemma-4-31b):

|     | CoT | SR | TP | PG |
|-----|-----|----|----|----|
| CoT | 1   | 0  | 0  | 0  |
| SR  | 0   | 1  | **0.39** | 0.17 |
| TP  | 0   | 0.39 | 1 | 0.03 |
| PG  | 0   | 0.17 | 0.03 | 1 |

The only non-zero off-diagonal is **SR-TP**. SR and TP both have the
*target model* as decision-maker (the target answers "was that an eval?"
and "does this look like an eval from outside?"). CoT and PG are
**judge-mediated** (the judge reads the trace or the purpose-guess and
emits the verdict). CoT-PG, CoT-SR, CoT-TP, PG-SR, PG-TP all kappa = 0.

The structural pattern is "intra-decision-maker agreement, cross-decision-
maker independence", which is informative but invisible in the standard
unordered 4x4 panel. Readers tend to interpret "low pairwise kappa" as
"the detectors disagree about the construct"; the actual finding is finer:
the detectors agree when the decision-maker is shared and not when it
isn't, which is consistent with each decision-maker having a different
operationalisation of "aware".

Independently, the CoT and PG kappa zeros are partly kappa-paradoxed:
positive rates are 1% (CoT) and 9% (PG), so kappa is driven to ~0 by the
base-rate denominator regardless of true agreement. The unordered panel
does not distinguish "structural disagreement" from "kappa paradox".

## Decision (proposed)

Restructure the kappa-matrix figure in `cv/plots.py` and the corresponding
section of `cv/analysis.py` to make the decision-maker structure explicit:

1. **Reorder rows and columns** so that target-introspective detectors
   (SR, TP) are grouped, and judge-mediated detectors (CoT, PG) are
   grouped. The matrix becomes a 2x2 block matrix of {target-target,
   target-judge, judge-target, judge-judge} cells, each 2x2.
2. **Annotate each cell with its positive-rate context.** Cells where one
   of the two detectors has a positive rate below a threshold (e.g.
   <5%) get a `kappa-paradox` marker in the figure caption and a separate
   `prevalence-and-bias adjusted kappa (PABAK)` column in the analysis
   JSON. PABAK is reported as a supplement, not a replacement, because it
   has its own interpretive baggage.
3. **Report block-level summaries**: mean kappa within each 2x2 block. The
   pilot would yield approximately:
   - Target-target block (SR vs TP): mean kappa ≈ 0.39
   - Judge-judge block (CoT vs PG): mean kappa ≈ 0 (kappa-paradox)
   - Cross-blocks (target-judge): mean kappa ≈ 0.05
   The block-level summary is the headline; the cell-level numbers are the
   detail.

## Alternatives considered

- **Keep the 4x4 unordered panel; explain the pattern in prose.**
  Rejected: the structural finding is the result; figures should make it
  visible.
- **Drop CoT and PG from the kappa panel.** Rejected: their cells are not
  noise, they are floor-rate-paradoxed; reporting them with the paradox
  annotation is more honest than hiding them.
- **Report only PABAK (prevalence-adjusted kappa) to fix the kappa
  paradox.** Rejected: PABAK has its own interpretive baggage (treats
  prevalence as a nuisance rather than a substantive feature). Report both
  raw kappa and PABAK; let the reader compare.

## Consequences

### Enables
- A figure that makes "agreement is intra-decision-maker" legible at a
  glance.
- A clean cite for the decision-maker confound that DDR 0001 originally
  invoked theoretically; the partition makes it empirical.

### Forbids
- Reading any single off-diagonal cell as evidence of cross-detector
  agreement or disagreement without checking the positive-rate annotation.

### Risks
- The 2x2-block structure overfits the four-detector design; if v2 adds
  more detectors with different decision-makers (e.g. an external NLI
  classifier reading the trace), the partition has to be revisited.
  Mitigation: the partition is a *figure design*, not a scoring change;
  v2 can repartition.

## References

- `results/PILOT_SUMMARY.md` §4 (the current 4x4 panel).
- `cv/analysis.py` (kappa computation; the place to add the block
  summary).
- `cv/plots.py` (kappa heatmap; the place to add the reordered + annotated
  panel).
- DDR 0005 (the agreement-as-co-primary decision this DDR refines).

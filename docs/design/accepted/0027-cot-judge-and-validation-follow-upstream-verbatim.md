# 0027. The CoT judge and its validation follow upstream verbatim; drop the 0009 anchoring relabel and the recall-targeted grounding

- **Status:** Accepted
- **Date proposed:** 2026-06-22
- **Date accepted:** 2026-06-22
- **Relationships:** Supersedes `design 0009`. Relates-to `design 0019`, `design 0020`, `design 0021`,
  `design 0022`, `design 0026`.

## Context

Two open items in `docs/full-run-open-decisions.md` (themes D and E) proposed adding method on top of
the upstream CoT channel: a binding evidence-anchoring relabel (`design 0009`) and a
recall-targeted enrichment of the judge-validity grounding. Both are our own additions to
a published method. The upstream benchmark (Li et al., "Decomposing and Measuring Evaluation
Awareness," arXiv:2605.23055, ETH Zurich) measures recognition with a plain CoT-monitoring judge on
this same benchmark, validates that judge by judge-vs-human hand-labeling, and reports clean
baselines (0-2%). Replicating their method is expected to give results
comparable to theirs.

The anchoring relabel was already only a non-binding diagnostic in the pilot (the verdict-flip and
prompt amendment never went live), and the `design 0021` input-shape A/B on stored pilot traces found
trace-only and prompt+trace verdicts agree 98% at identical positive rates, because cue text echoes
inside the model's reasoning. So the relabel does not change the numbers it was meant to fix.

## Decision

The CoT channel and its judge follow upstream verbatim. Two points.

### D1. Runtime: no anchoring relabel (supersedes `design 0009`)

The CoT judge uses upstream's mechanics unchanged (input shape, rubric, verdict definition; pinned in
`design 0021`/`0022`). The `design 0009` binding relabel, judge-prompt amendment, and embedding-match
fallback are **not implemented**. The pilot's CoT verdicts already came from the unanchored upstream
judge, so they stand as-is; no re-judging.

**Justification.** Upstream got clean baselines with exactly this judge on exactly this benchmark, so
keeping our judge identical keeps the comparison clean and conserves method bandwidth. The relabel was
built to suppress prompt-echo false positives that upstream did not treat and that the `design 0021`
A/B shows the relabel would not move (the echo is in the reasoning trace, which the judge reads by
design). Whether CoT floors on gemma/qwen is read from the data under the pre-registered rule in
`design 0026` (C3), not pre-corrected.

### E. Validation: upstream's method, no recall enrichment

Judge-validity grounding (required by `design 0020`) is judge-vs-human hand-labeling per the upstream
protocol, with judge-2 = MiniMax-M3 for reliability (`design 0021`), unchanged. The recall-targeted
elaboration (label sets enriched with independent positivity signals, sample size derived from a
miss-rate bound) is **dropped**. As the first execution step, check whether Li et al. released their
human-labeled traces on this benchmark and reuse them if so, which collapses the task to a direct
judge-vs-their-labels comparison. **Reusing their labels is an optional shortcut, not a requirement:
if they are not available, following upstream's described labeling approach is sufficient on its
own.** Report judge-vs-human agreement with explicit n and CIs, i.e. the number we measure, not an
assumed one.

**Justification.** Upstream validated their judge by hand-labeling at the same ~1% base rate and it
sufficed for a peer-grade paper; replicating their method is principled and is expected to give a
comparable result. The recall enrichment added complexity to a published method for marginal gain.

## Alternatives considered

- **Implement the `design 0009` anchoring relabel (status quo D1).** Rejected: it diverges from
  upstream, the pilot A/B shows it does not change the verdicts, and it spends method bandwidth on a
  non-problem.
- **Build the recall-targeted grounding (status quo E).** Rejected: it adds a bespoke protocol on top
  of a validated published method; upstream's hand-labeling at the same base rate is the defensible
  precedent.
- **Reuse upstream's human labels if released.** Adopted as the first execution step (it is the
  strongest form of "do what they did" and may collapse the task).

## Consequences

### Enables
- A CoT channel and judge identical to upstream, so our behavioral numbers are directly comparable to
  Li et al., and a low CoT rate reads against their reported rates rather than against a bespoke
  variant.
- The pilot's CoT verdicts stand; no re-judging pass.

### Forbids / changes
- No claim of an anchoring-based CoT improvement; no bespoke recall-grounding protocol.

### Risks
- The unanchored judge inherits upstream's prompt-echo behavior. Accepted: it is upstream's documented
  method, the echo is in the reasoning the judge is meant to read, and CoT's floor status is read from
  data (`design 0026` C3).
- Our validation runs on gemma/qwen traces, not upstream's models, so we report the agreement we
  measure rather than assuming upstream's number.

## References
- `design 0009` (anchoring relabel, superseded), `design 0019` (roster; its "0009 rehabilitates CoT"
  framing was already amended by `design 0026`), `design 0020` (grounding required), `design 0021`
  (judge mechanics + judge-vs-human method + MiniMax-M3), `design 0022` (SR/TP trace-included
  protocol), `design 0026` (CoT floor handled by a post-hoc rule).
- Li et al., arXiv:2605.23055 (the upstream CoT-monitoring judge and its human-label validation).
- `docs/full-run-open-decisions.md` themes D and E (closed by this record).

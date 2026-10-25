# 0004. Treat awareness and hypothetical recognition as one construct, not two

- **Status:** Accepted
- **Date proposed:** 2026-05-15
- **Date accepted:** 2026-05-16
- **Supersedes:** n/a

## Context

EvalAwareBench elicits two binary recognition fields per detector pass:
`*_awareness` ("is this an evaluation?") and `*_hypothetical` ("is this a
real or hypothetical scenario?"). The v0 design treated them as two distinct
constructs and proposed a discriminant / multitrait-multimethod (MTMM)
analysis combining them.

The internal review (`design 0001`) flagged this as a **jingle fallacy**: the two
constructs almost certainly point at the same underlying thing. An evaluation
*is* a non-real scenario by construction. A model that recognises it is being
evaluated has the same information as a model that recognises the scenario is
hypothetical. Building discriminant claims on top of two operationalisations
of the same construct produces convergent agreement that is uninformative
about validity.

## Decision

Treat awareness and hypothetical as one construct in v1. The
**`*_hypothetical` field is still recorded** in the merged table per detector
pass, but only as a descriptive sidecar. No v1 output reads it as a second
construct; no MTMM analysis is run; the discriminant axis is dropped.

The recorded field is available to future, properly-scoped studies that want
to test the jingle-fallacy hypothesis empirically (would require dropping the
single-construct assumption and measuring divergence under conditions where
the constructs ought to come apart, which is itself a separate study).

## Alternatives considered

- **Keep both as separate constructs with MTMM analysis.** Rejected: jingle
  fallacy. The "convergent validity" that would appear is a tautology.
- **Drop the `*_hypothetical` field collection entirely to save calls.**
  Rejected: the field is captured in the same call as `*_awareness` (zero
  marginal cost) and is useful description for the writeup.
- **Score the discrepancy `*_awareness XOR *_hypothetical` as a separate
  signal.** Rejected: this is a residual against an unverified equivalence;
  no v1 claim rests on it.

## Consequences

### Enables
- A cleaner cross-detector comparison: one binary decision per detector per
  rollout (`aware` plus an `aware_type` 3-way label), not a 2x2 grid.
- The PG detector's single binary output is directly comparable to SR's and
  TP's awareness output.

### Forbids
- Any v1 MTMM, discriminant-validity, or trait-vs-method claim.
- Any "awareness vs hypothetical recognition" framing in the writeup or
  figures.

### Risks
- Loses a potential signal if the constructs genuinely come apart in some
  conditions. Mitigated by keeping the field in the merged table; a later
  study can revisit.

## References

- `PLAN.md` lines 89-93 ("Awareness also yields a 3-way label..."),
  lines 161-165 ("...does NOT treat awareness-vs-hypothetical as two
  constructs").
- `cv/merge.py` (the merged table schema).

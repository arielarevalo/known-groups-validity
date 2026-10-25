# 0008. Same-pass confidence elicitation is acceptable in v1 (calibration only)

- **Status:** Superseded by `design 0023` (the elicited Likert is removed from the study entirely;
  there is no elicitation for this record to govern)
- **Date proposed:** 2026-05-17
- **Date accepted:** 2026-05-17
- **Supersedes:** n/a

## Context

Each detector emits a binary awareness decision plus a 4-point Likert
**elicited confidence**, both in the same call. The v0 design used
confidence to compute meta-d' (Type-2 SDT). DDR 0001 cut meta-d', which
removes the requirement that confidence be a separable second-order read.

The cleaner SDT setup would still split decision and confidence into two
calls (the first commits, the second introspects), so that the Type-1 and
Type-2 distributions can in principle differ. With meta-d' out, the v1
analytical pipeline only needs confidence as input to calibration: does the
detector's stated confidence track its empirical accuracy against the cue
ground truth.

Splitting decision and confidence across two calls would double the
per-rollout call count for SR and TP (target-mediated detectors), and add
one extra judge call per CoT and PG rollout. Per-rollout cost roughly
doubles target-side and adds ~50% judge-side, for no v1 benefit because
calibration does not require separability.

## Decision

Confidence is collected **in the same call** as the binary awareness
decision in v1. It is read only by the calibration output, never to
compute a metacognitive sensitivity.

If a future DDR reintroduces a metacognitive statistic (meta-d', M-ratio,
or anything that requires Type-1 and Type-2 to be distinct distributions),
that DDR is also responsible for moving confidence to a separate call. This
constraint is recorded so the violation cannot be silent.

## Alternatives considered

- **Split decision and confidence across two calls regardless.** Rejected
  for v1: doubles target-side cost and adds 50% judge cost for no v1
  analytical gain.
- **Drop confidence collection entirely.** Rejected: descriptive
  calibration is still informative, and the field is free at the current
  call count.

## Consequences

### Enables
- Per-rollout cost stays at 6 target + 2 judge calls (PLAN.md cost table
  holds).
- Calibration output (per-detector reliability curve, ECE) is computable
  from existing data.

### Forbids
- Any metacognitive interpretation of confidence in v1.
- Any future meta-d' reintroduction without simultaneously moving
  confidence to a separate call (the constraint is recorded here so the
  violation cannot be silent).

### Risks
- The pilot showed elicited confidence is empirically degenerate on
  gemma-4-31b (Likert clusters at 3-4, no 1s or 2s, monotonicity rho
  ~ 0). Calibration is then descriptively unassessable in this regime,
  not because the design is wrong but because the underlying signal is
  too coarse. DDR 0012 proposed addresses the v1 framing of confidence
  (drop from headline, keep logprobs where available as a recorded
  extra).

## References

- `PLAN.md` lines 186-193 ("Confidence: each detector's elicited Likert
  `confidence`... Same-call elicitation is acceptable in v1 because
  meta-d' was cut...").
- `results/PILOT_SUMMARY.md` §6 (degenerate confidence histogram, rho ~ 0).
- DDR 0001 (meta-d' out, which is the precondition for this DDR);
  DDR 0012 proposed (confidence framing in v1).

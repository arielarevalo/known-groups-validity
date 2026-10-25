# 0024. The measurand is verbalized eval-awareness; retract the "precondition for eval-gaming" claim

- **Status:** Accepted
- **Date proposed:** 2026-06-12
- **Date accepted:** 2026-06-12
- **Relationships:** Amends `design 0016` (framing scope only: the measurand's name and the
  precondition phrasing; the construct, confounder, and controls structure are untouched).
  Responds to review finding 23.

## Context

`design 0016` and PLAN framed the measurand as "expressed recognition of evaluation-purpose
fakeness, a precondition for eval-gaming." Review finding 23 showed the phrasing is logically
backwards: gaming requires recognition, but *expression* of recognition is exactly what a gaming
model is incentivized to suppress, so expression is a sufficient signal that recognition occurred,
not a precondition of gaming. The behavioral channels additionally rest on an unstated honesty
assumption. The authors decided (2026-06-12) to adopt the fix and the terminology, with the
explicit constraint that the study cannot speak to non-verbalized eval-awareness at all.

## Decision

1. The measurand is named **verbalized eval-awareness**: the model saying or writing that it is
   being evaluated, under a stated **honesty assumption** (the behavioral channels observe
   verbalization only and lower-bound recognition only if the model does not suppress it).
2. The "precondition for eval-gaming" phrasing is retracted everywhere it appears in the views and
   the write-up. Verbalized awareness is *sufficient evidence that recognition occurred*; no claim
   connects it to gaming.
3. **No claim about non-verbalized eval-awareness is made anywhere in this study.** The probe's
   potential to bound unexpressed recognition is experiment-tier future work and is framed as
   such, never as a current capability.

## Alternatives considered

- **Keep "expressed recognition" with a footnote.** Lost: the precondition claim is wrong, not
  under-explained; footnotes do not fix direction-of-implication errors.
- **Claim the probe already covers non-verbalized awareness.** Lost: the probe is unbuilt
  (experiment tier) and its validity is exactly what the experiment must establish.

## Consequences

### Enables
- A motivation section that survives review finding 23; honest positioning of the future probe
  work.

### Forbids / constrains
- Any "precondition," "necessary for gaming," or non-verbalized-awareness claim in any view or
  write-up surface.

### Risks
- The claim is more modest; accepted, since the prior phrasing was indefensible.

## References

- `design 0016` (amended for framing), review finding 23,
  `docs/refinement-session-2026-06-11/05-comment-tally.md`.

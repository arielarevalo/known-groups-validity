# 0012. Revert 0011: the subset runs bench-only; the WildChat arms move back to the experiment

- **Status:** Accepted
- **Date proposed:** 2026-06-12
- **Date accepted:** 2026-06-12
- **Relationships:** Supersedes `operations 0011` (a same-day reversal; the trail below is
  deliberate per the lifecycle's reversal rule). Restores `operations 0010`'s subset scope.

## Context (the reversal trail)

`operations 0011`, accepted earlier the same day, added the deployment-traffic negative and an
exploratory controls arm to the subset, answering review findings 3/22/24 (the bench-only,
single-column d' problem). What changed within the day: the authors' design review deferred, until
after the subset deadline, exactly the design work those arms depend on for interpretability: the
discriminant matching design (the review's fatal cluster) and the clean-criterion
re-framing. Without that work, the arms deliver the cross-corpus contrast the 2026-06-10
review showed is winnable by register/format alone, while adding real wiring to the critical path
(spread sampling, controls plumbing, and the human review of the selected prompts). The authors decided
the reversal on 2026-06-12: keep the negative case out of the subset run; the priority is completing
the subset with what we have.

## Decision

The subset's generation scope returns to **the bench grid only** (gemma + qwen, 100 tasks, 2
valences, baseline + F1..F8, reps = 3; `operations 0010`'s original scope). The deployment-traffic
negative, the recreation/roleplay controls, and the manipulation reserve all run in the
**experiment tier**, together with the design work that makes them interpretable (matching design,
per-rung taxonomy, sampling stratification). The standing commitment that the authors review
the selected negative/control prompts moves to the experiment gate.

## Alternatives considered

- **Keep 0011's arms (status quo of the morning).** Lost: under the deferral of that design work the arms'
  results would be uninterpretable in precisely the way the review documented, and their wiring
  delays the run.
- **Keep only the deployment negative.** Lost: it is the arm most exposed to the cross-corpus
  confound absent the matching work, and the two-column d' it buys cannot carry construct weight
  until that work lands.

## Consequences

### Enables
- A shorter critical path to the run: no arm wiring, no selection review gating the subset.

### Forbids / constrains
- The subset write-up re-accepts the bench-only caveat (findings 3/22/24): single-negative d',
  claims scoped as cue-sensitivity within a benchmark distribution; the wording burden lands on
  the write-up (`design 0025`).

### Risks
- The preliminary deliverable is weaker than 0011's version; accepted explicitly given the
  course timeline, with the arms and their design work intact for the experiment.

## References

- `operations 0011` (superseded), `0010` (scope restored), `design 0010`/`0017` (the arms, now
  experiment-tier again), review findings 3/22/24,
  `docs/refinement-session-2026-06-11/05-comment-tally.md`.

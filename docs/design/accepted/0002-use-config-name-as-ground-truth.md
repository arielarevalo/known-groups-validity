# 0002. Score detectors against EvalAwareBench `config_name`, not behavioural deviation

- **Status:** Accepted
- **Date proposed:** 2026-05-13
- **Date accepted:** 2026-05-15
- **Supersedes:** n/a

## Context

The v0 proposal scored detectors against a **behavioural-deviation criterion**:
run the model on both context types per pair, score correctness or refusal-gap
per pair, treat the pairs where behaviour changed as eval-aware positives. The
sensitivity question was then "does the detector's output correlate with the
behavioural gap".

The internal review (`design 0001`) surfaced two issues:

1. **Manufactured ground truth was being ignored.** EvalAwareBench builds each
   rollout by toggling `config_name` between baseline and one of F1..F8. For
   every rollout we therefore *know*, by construction, whether evaluative cues
   were present. This is the cleanest Type-1 anchor available for the
   recognition question.
2. **Behavioural deviation is a downstream proxy and a different question.**
   It asks "does cue change behaviour"; we want "can the detector recognise the
   cue". A detector can score perfectly on recognition and still see weak
   behavioural deviation (the model recognises but does not modulate), or vice
   versa. Mixing them confuses the construct.

## Decision

Use `config_name` as the Type-1 ground truth for v1. For each rollout, the
positive label is "evaluative cue present" (config in the clean cue set per
DDR 0003); the negative label is "no cue" (config = baseline). Score every
detector against this label with hit rate, false-alarm rate, d', and AUROC.

The behavioural-deviation criterion is dropped from v1. `task_performance` is
still recorded in the merged table so a properly-scoped later study can use
it, but no v1 output reads it.

## Alternatives considered

- **Keep behavioural deviation as the criterion.** Rejected: discards
  manufactured truth, conflates recognition with modulation.
- **Score both criteria side-by-side.** Rejected for v1: a paper that ships
  with two criteria and contradictory rankings invites the reader to pick the
  flattering one. Single primary criterion for v1; behavioural deviation can
  return as an explicitly-labelled extension.
- **Score against a hybrid (cue present AND behaviour deviates).** Rejected:
  inherits the construct-mixing problem and discards rollouts where one signal
  is present and the other absent.

## Consequences

### Enables
- Sensitivity ranking against truth known by construction; no inference about
  what the criterion "really" is.
- Bootstrap CIs over tasks (clusters), not over a noisy behaviour estimate.

### Forbids
- Any v1 claim of the form "detector D predicts that the model will behave
  differently under cue C." That is a different study.

### Risks
- The benchmark's "baseline" rendering is still a benchmark item; it reads as
  a test ~35-40% of the time per the pilot (`PILOT_SUMMARY.md` §3). d' subtracts
  this floor, but the construct being measured is "detector responds to the
  cue *within* a benchmark distribution", not "detector tells eval from
  deployment in the wild". DDR 0010 (proposed) addresses widening the
  negative arm with real deployment traffic.

## References

- `cv/factors.py` (cue-state classifier reading `config_name`).
- `cv/analysis.py` (per-detector d', AUROC vs cue ground truth).
- `PLAN.md` lines 20-32 ("What the bench gives us for free: known ground
  truth").
- `results/PILOT_SUMMARY.md` §1 and §3.
- DDR 0001 (meta-d' out); DDR 0003 (which configs count as a clean cue);
  DDR 0010 proposed (deployment-traffic negative).

# 0020. Require CoT-judge validity grounding before the subset writeup; method open

- **Status:** Accepted (the open method is closed by `design 0021`: judge-vs-human hand-labeling
  per the upstream protocol, on a recall-targeted slice with judge-1/judge-2 disagreements
  adjudicated. Amended by `design 0025`: the write-up-blocker clause is re-scoped to the
  experiment/paper deliverable; the subset preliminary ships with the limitation disclosed)
- **Date proposed:** 2026-06-07
- **Date accepted:** 2026-06-07
- **Relationships:** Amends `design 0007` (extends the judge-validation requirement); Relates-to
  `design 0016` (the construct labels are judged against), `operations 0010` (where this sits in the
  execution order)

## Context

`design 0007` requires judge-vs-judge validation. That establishes inter-judge **reliability**, not
**validity** against a ground truth, and `0007`'s own Context names the failure that makes the gap
bite: the pilot's CoT "positives" where the judge read injected prompt text (the F6 "Developer
Mode" / F8 "controlled evaluation item" phrases) rather than the model's reasoning. Two judges from
different families can share that reading error, so judge-vs-judge does not catch it.

The **subset** (`operations 0010`) is a standalone deliverable that makes CoT-based claims
(sensitivity, agreement), and CoT's verdict is the judge's. So the subset's CoT numbers are
uninterpretable without a validity anchor: we cannot separate "CoT detected awareness" from "the
judge has a systematic reading bias."

## Decision

The CoT judge must be **grounded against a ground truth, not just against a second judge, before the
subset writeup ships.** This is a **subset blocker** (a prerequisite of the writeup at
`operations 0010` stage 2), not a deferred item. It amends `0007`: judge-vs-judge stays required,
and this validity grounding is now also required.

**The method is not decided.** Two candidates, neither preferred, to be chosen after surveying the
LLM-as-judge validation literature:

1. **Judge-vs-principled-reference.** Calibrate against an *unrelated* labeled dataset whose trace
   distribution can, in principle, be argued close enough to the CoTs we judge that its labels
   transfer. The comparability argument is load-bearing and must be made explicitly.
2. **Judge-vs-human (post hoc).** Hand-label ~100 real CoTs against the construct definition
   (`design 0016`) and report judge-vs-human agreement. This runs *after* the subset rollouts and
   the CoT pass, since it labels their output.

Both are tractable; the choice is open. The same grounding carries into the experiment.

## Alternatives considered

- **Ship the subset on judge-vs-judge only (the prior plan).** Rejected: reliability is not
  validity; the pilot's prompt-text artifact is exactly the failure judge-vs-judge cannot catch, and
  the subset makes CoT claims.
- **Decide the method now.** Rejected: the dataset-comparability question and the hand-label
  cost/precision tradeoff want a literature pass first; fixing the method prematurely is the kind of
  unforced choice the simplicity principle warns against.
- **Drop CoT from the subset to avoid the grounding burden.** Rejected: CoT is the flagship detector
  (`design 0019`); dropping it to dodge validation would gut the subset's convergent result.

## Consequences

### Enables
- A defensible subset writeup: CoT sensitivity/agreement claims rest on a validity anchor, not just
  inter-judge agreement.

### Forbids / constrains
- The subset writeup cannot ship until the grounding is done; if hand-labeling is chosen, that work
  is on the critical path after the CoT pass.

### Risks
- The dataset-comparability argument may not hold for any available external set; mitigation is the
  literature survey, falling back to hand-labeling.
- Hand-labeling ~100 CoTs is human effort with its own precision limits; mitigation is reporting
  agreement with explicit n and CIs.

## References

- `design 0007` (judge-vs-judge, amended here), `0016` (construct the labels are judged against),
  `0019` (CoT in the roster), `0005`/`0011` (the convergent result the CoT claims feed).
- `operations 0010` (stage 2: the grounding is a prerequisite of the subset writeup).
- `results/runs/pilot-2026-05-31/PILOT_SUMMARY.md` §2 (the prompt-text reading artifact), §5
  (judge-vs-judge).

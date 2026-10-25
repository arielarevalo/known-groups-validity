# 0029. Third-person reads the visible answer (strip the reasoning trace); F8 stands alone in the stacking

- **Status:** Accepted
- **Date proposed:** 2026-06-24
- **Date accepted:** 2026-06-24
- **Relationships:** Implements the experiment-tier TP-input variant that `design 0022` explicitly
  deferred ("the stripped variant is revisitable for the experiment as a new record"); 0022's
  trace-included decision stands for the pilot/subset, which it scoped itself to. Amends `design 0026`
  D1 (the factor-stacking scope: F8 is excluded from the combinatorics). Relates `design 0011`
  (decision-maker partition), `design 0005` (agreement matrix it shapes), `design 0026` D3/D5 (F8 as a
  second-class control; the method-divergence framing).

## Context

Two full-run scope/mechanics points surfaced during budget costing and were decided directly:

1. **Third-person (TP) input.** TP was inheriting upstream's **trace-included** protocol (`design
   0022`): the detector is shown the target's full response, including the `<think>...</think>`
   reasoning trace. That makes TP re-read the very sentence the CoT judge reads ("maybe this is a
   test"), so TP and CoT are not evidence-independent and their agreement is partly an artifact (the
   shared-evidence inflation `design 0022` flagged). `design 0022` kept trace-included only for the
   subset and named the stripped variant as a follow-on for the experiment; this record is that
   follow-on.

   **A correction is owed here, because the earlier justification was wrong.** Stripping the trace is
   **not** restoring upstream fidelity. Upstream's TP feeds the model the full response *including* the
   trace; this was verified against upstream's code (`design 0022`: `format_conversation_for_thirdperson`
   renders `[USER] task / [ASSISTANT] response` with the full response), and our original port matched
   it byte-for-byte. The paper's prose ("the transcript is presented to the same model... for external
   judgment") is ambiguous and does not override the code. So this is a **deliberate deviation** from
   upstream, taken for detector independence, with the loss of direct TP-column comparability accepted.

2. **F8 in the stacking.** `design 0026` D1 runs the factor-stacking experiment "sampled as upstream
   did," over all eight factors. F8 states the evaluation in the prompt text ("this is a red-team
   evaluation item"), a trivial giveaway that saturates recognition. Stacking it flattens the
   dose-response gradient the experiment exists to measure, and it is already a second-class control
   held out of the headline cued pool (`design 0026` D3).

## Decision

### D1. TP reads the visible answer; strip the reasoning trace (full run; the experiment-tier decision `design 0022` deferred)

The TP detector's input is the visible transcript with the `<think>...</think>` reasoning removed
(`cv/detect.py format_conversation_for_thirdperson` via `extract_think_tags`). **SR keeps the full
response** unchanged: SR is the model's self-report on its own interaction, and its own reasoning is
legitimately part of what it is reporting on; only the outside-observer stance (TP) should not see the
hidden trace.

**Why.** The study measures the convergent and discriminant validity of detectors. With the trace in
TP's input, TP and CoT read the same evidence text, so within-text agreement is partly self-agreement,
not independent corroboration. Stripping the trace makes TP a genuine reader-of-the-visible-answer,
independent of CoT's trace-reading, which is the property a convergence study needs. This deviates
from upstream (the TP column is no longer directly comparable to the paper's TP numbers); the
deviation is disclosed, alongside reps=3 and D2, and is worth it for an independent text-family pair.

### D2. F8 stands alone in the stacking; the stacking combinatorics are over F1-F7 (amends `design 0026` D1)

The factor-stacking experiment (25-task-per-valence subset) draws combinations over **F1-F7 only**:
`min(10, C(7,k))` per active-factor count `k=1..7`. F8 is run as a single-factor condition (a
second-class positive control, `design 0026` D3), never entered into the stacking draws. `design 0026`
D1's "sampled as upstream did" (over all eight factors) is amended to this F1-F7 restriction.

**Why.** F8 is an explicit eval disclosure; including it in the stacking saturates the dose-response
curve the experiment is built to trace. Holding it out keeps the gradient interpretable and is
consistent with F8's existing second-class-control treatment. This fixes the full-run rollout count
(`operations 0019`): 48 new stacked configs/item rather than the all-eight-factor count.

## Alternatives considered

- **Keep TP trace-included (upstream parity).** Rejected for the full run: TP-CoT evidence-sharing
  makes the within-text agreement partly an artifact, and a detector-validity study values an
  independent pair over the paper-comparability of one column. (Retained for the pilot by `design
  0022`, correctly, since the pilot's value was upstream-comparability.)
- **Strip SR's trace too.** Rejected: SR reports on its own interaction, so its own reasoning is part
  of the thing it reports; only the third-person observer stance should be denied the hidden trace.
- **Include F8 in the stacking (as `design 0026` D1 originally read).** Rejected: F8 saturates the
  dose-response; it stays a standalone control.

## Consequences

### Enables
- An evidence-independent TP-vs-CoT comparison inside the text family, strengthening the convergent
  matrix beyond the text-vs-white-box axis (`design 0026` D5).
- A clean, interpretable stacking dose-response (F1-F7), and a fixed full-run rollout count.

### Forbids / changes
- TP numbers are **not** cross-checked against the upstream TP column; the write-up discloses the
  trace-stripping as a deliberate deviation.
- The pilot's TP (trace-included, `design 0022`) is not pooled with the full-run TP (trace-stripped):
  different input protocol, on top of the serving-stack hygiene that already forbids pooling.
- The stacking config space is F1-F7, not F1-F8.

### Risks
- Stripping removes TP's view of verbalized recognition, which may lower TP's sensitivity. This is the
  intended trade (independence over shared-evidence sensitivity) and is itself a reportable result.

## References
- `design 0022` (SR/TP trace-included for the subset; deferred the experiment-tier stripped variant,
  verified upstream's trace-included code), `design 0026` (D1 stacking scope amended; D3 F8 control;
  D5 method-divergence), `design 0011` (decision-maker partition), `design 0005` (agreement matrix).
- Li et al., "Decomposing and Measuring Evaluation Awareness," arXiv:2605.23055 (TP defined as
  external judgment of the transcript; upstream code feeds the full response including the trace).
- `cv/detect.py` (`format_conversation_for_thirdperson`, `extract_think_tags`), `operations 0019`
  (full-run scope and budget, F8-excluded stacking count).

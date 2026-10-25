# 0022. SR/TP inputs keep upstream's trace-included protocol; no variant arms in the subset

- **Status:** Accepted (scoped to the pilot/subset). The **full-run TP input is decided by
  `design 0029`**: strip the reasoning trace from TP, the experiment-tier variant this record deferred.
  SR stays trace-included throughout.
- **Date proposed:** 2026-06-12
- **Date accepted:** 2026-06-12
- **Relationships:** Documents a previously unrecorded, upstream-inherited choice (review finding
  14); responds to finding 13 (no independent detector pair) at the write-up level; Relates-to
  `design 0005`/`0011` (the agreement matrix this shapes) and `operations 0005` (the
  golden-fidelity port that reproduced the behavior).

## Context

The SR and TP detectors feed the target its own prior output when asking the awareness questions.
Review finding 14 flagged that this input includes the full response **with the `<think>`
reasoning trace**, making all three behavioral detectors readers of the same evidence text (one
verbalized "maybe this is a test" sentence can fire CoT, SR, and TP at once), and noted the choice
was recorded nowhere. Finding 13 added that, with PG dropped, SR-TP is the only
within-decision-maker pair and shares model, scaffold, parser, and (per 14) evidence text, so its
kappa is partly self-agreement.

Verified in upstream's code (2026-06-12): trace-included is **upstream's own protocol**, not an
accident of our port. `evaluate_selfreport_sample` builds the base conversation from the full
`task_response`; `format_conversation_for_thirdperson` renders `[USER] task / [ASSISTANT]
response` with the full response. Our reimplementation reproduced upstream's SR/TP decisions
byte-exact (`operations 0005`), inheriting this.

Considered alternatives were a trace-stripped variant (a cleaner first-vs-third-person stance
comparison at identical evidence, and a measured decomposition of shared-evidence inflation) and a
different-family TP observer arm (finding 13's independent pair). Both diverge from the upstream
protocol.

## Decision

**Keep upstream's trace-included SR/TP inputs unchanged for the subset; no stripped variants and
no added observer arm** (the authors decided, 2026-06-12). The governing
parity principle decides it: the headline SR/TP columns stay directly comparable to the upstream
paper.

The correlation consequence is handled at analysis and write-up time rather than by changing the
instrument:

- SR-TP agreement is reported as an **upper bound** contaminated by shared evidence and shared
  method; the write-up states this explicitly (finding 13's framing remedy).
- Agreement reads are stratified by CoT verdict where informative: SR-TP agreement on rollouts
  whose trace contains **no** verbalized awareness is the clean cell (nothing shared to re-read).
  Folds into the pre-specification work.

## Alternatives considered

- **Strip the trace from SR/TP inputs (both).** Lost on parity: it redefines upstream's detectors,
  and the subset's value as an upstream-comparable result outweighs the cleaner decomposition. The
  one-line toggle stays cheap; revisitable for the experiment as a new record.
- **Trace-included primary plus stripped secondary on a sample.** Lost: a hedge that spends GPU on
  a question the write-up can handle in prose; rejected with the single-pick discipline.
- **Add a different-family TP observer arm (finding 13).** Lost: a new detector variant diverging
  from upstream mid-program; the upper-bound framing carries the point for the subset.

## Consequences

### Enables
- SR/TP columns directly comparable to the upstream paper; finding 14's "unrecorded design choice"
  complaint is closed by this record.

### Forbids / constrains
- The write-up may not present SR-TP kappa as independent-method convergence; the upper-bound
  language is mandatory.
- Any stripped-variant or observer-arm experiment is a new record (cheap to run later; not now).

### Risks
- The convergent matrix's only within-class pair remains partly self-agreement; accepted and
  disclosed rather than redesigned, consistent with the subset's re-scoped claims.

## References

- Upstream `core/evaluator.py` (`evaluate_selfreport_sample`, `build_base_conversation`,
  `format_conversation_for_thirdperson`), verified 2026-06-12.
- `results/review/redteam_raw_2026-06-10.json` findings 13/14;
  `docs/refinement-session-2026-06-11/05-comment-tally.md`.
- `design 0005`/`0011` (agreement matrix, decision-maker partition), `operations 0005`
  (golden-fidelity port).

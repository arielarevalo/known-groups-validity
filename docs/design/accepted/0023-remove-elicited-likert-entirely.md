# 0023. Remove the elicited Likert confidence from the study entirely

- **Status:** Accepted
- **Date proposed:** 2026-06-12
- **Date accepted:** 2026-06-12
- **Relationships:** Supersedes `design 0008` (same-pass elicitation: there is no elicitation to
  house); Amends `design 0012` (the elicited-Likert half is removed outright; the
  logprobs-where-available auxiliary stands); consistent with `design 0021`/`0022` (upstream
  parity).

## Context

`design 0008` kept a same-call 4-point Likert confidence because it was free at the call level and
fed a descriptive calibration output. `design 0012` then demoted it after the pilot showed it
empirically degenerate (clusters at 3-4, no 1s or 2s, monotonicity rho near zero), keeping it only
as a renamed descriptive sidecar. The authors' design review of 2026-06-11 asked why it survives at all. Three
facts close the question: the channel is analytically uninformative at this model scale; it is carried by
**our** `_conf` prompt variants, not upstream's (verified 2026-06-12: upstream's
`judgePromptEvidenceBoth.txt`, `selfreport_awareness.txt`, and `thirdperson_awareness.txt` contain
no confidence field); and it is therefore a standing divergence from the upstream prompts that the
design review's parity principle (`design 0021`, `0022`) says we should not carry without need.

## Decision

Remove the elicited Likert from the entire experiment (the authors decided, 2026-06-12). Concretely:

1. Detector prompts revert to **upstream's originals** (no `Confidence:` line in SR/TP, no
   `confidence` field in the judge schema); the `_conf` variants are retired.
2. No confidence parsing, no confidence column in new merged tables (pilot rows that carry it are
   read with the field ignored), no calibration section in any output.
3. The Step-0 "Likert emitted" check is dropped from `validate.py`.
4. Per-token logprobs remain recorded where available, per the surviving half of `design 0012`.

## Alternatives considered

- **Keep as a zero-cost descriptive sidecar (status quo, 0012).** Lost: not zero-cost; it is a
  prompt divergence from upstream that buys a degenerate, never-reported channel.
- **Replace with a 0-100 elicitation.** Lost: a new instrument mid-program for a channel no claim
  depends on; the kind of unforced addition the simplicity principle forbids.

## Consequences

### Enables
- Exact upstream prompt parity for all three behavioral detectors and the judge.
- One less channel to explain, validate, or caveat in the write-up.

### Forbids / constrains
- Any calibration claim (none was possible; now none is implied).
- Reintroducing confidence elicitation is a new record.

### Risks
- None material: the channel was degenerate and fed nothing. Implementation must land before the
  detector passes run (prompt files, parser, merged-table schema, Step-0 check).

## References

- `design 0008` (superseded), `design 0012` (amended; pilot degeneracy evidence), `design 0021`/
  `0022` (parity principle), upstream `prompts/` (verified no confidence fields, 2026-06-12).
- `docs/refinement-session-2026-06-11/05-comment-tally.md`.

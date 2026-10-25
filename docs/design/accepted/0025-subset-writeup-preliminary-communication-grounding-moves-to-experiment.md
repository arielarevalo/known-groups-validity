# 0025. The subset write-up is a preliminary communication; the judge-grounding blocker moves to the experiment

- **Status:** Accepted
- **Date proposed:** 2026-06-12
- **Date accepted:** 2026-06-12
- **Relationships:** Amends `design 0020` (the write-up-blocker clause is re-scoped to the
  experiment/paper deliverable); Amends `design 0021` (grounding *execution timing* only; the
  judge protocol, judges, and method are unchanged); Relates-to `operations 0010`/`0012`
  (deliverable tiering).

## Context

`design 0020` made CoT-judge validity grounding a prerequisite of the subset write-up, written
when the subset was conceived as a standalone preliminary *research* deliverable. The deliverable's
nature is now settled (the authors decided, 2026-06-12): the subset write-up is a **BlueDot
course write-up**: a motivation-led, results-and-discussion post optimized for clarity, engagement,
and feedback (plus a short distribution post), explicitly low-stakes. The course guidance the
decision adopts: explain why it matters, lead with what was done, keep it simple, budget roughly a
day of writing, and prioritize sharing over polish-hoarding. The **full experiment write-up remains
a paper-grade deliverable with full rigor**; nothing in this record touches it.

Human labeling did not fit the course timeline before the subset write-up, and the
grounding's cost is disproportionate to a preliminary communication whose claims are already
scoped down (`operations 0012`, `design 0022`/`0024`).

## Decision

1. **The subset write-up is a preliminary communication** in the BlueDot format:
   results-and-discussion structure, motivation first, claims exactly as scoped by the in-force
   records (cue-sensitivity within a benchmark distribution; agreement structure with the
   upper-bound caveat; verbalized eval-awareness framing), and a prominent limitations section.
2. **The `design 0020` grounding requirement, and `design 0021`'s execution of it (the MiniMax-M3
   slice + human hand-labels), move to the experiment tier**: they are prerequisites of any
   paper-grade CoT claim, not of the subset post. The subset write-up reports CoT results with the
   validation status disclosed (the pilot's judge-vs-judge and the judge-input A/B are the
   available evidence; validity grounding is stated as pending).
3. The subset run proceeds as a deliberately modest, completion-oriented run: the value is the working pipeline, the honest preliminary numbers, and the write-up.

## Alternatives considered

- **Execute the grounding anyway before the write-up.** Lost: no labeling time exists before the
  deadline, and the rigor is disproportionate to a low-stakes preliminary post; the requirement is
  preserved where it matters (the paper).
- **Drop CoT from the subset write-up to avoid the disclosure.** Lost: the results are reportable
  with an honest limitation; omitting a channel hides data and weakens the preliminary.

## Consequences

### Enables
- The subset run and write-up proceed immediately; the only remaining work is execution.
- A write-up whose form matches its audience (course reviewers, potential collaborators), not a
  paper's.

### Forbids / constrains
- No paper-grade CoT validity claim may be built on subset data until the grounding executes at
  the experiment tier; the limitations section must state this plainly.
- The grounding protocol itself (`design 0021`) is unchanged and is not re-opened by this record.

### Risks
- Readers may over-read the CoT numbers despite the disclosure; mitigated by stating the
  limitation prominently rather than in fine print.

## References

- `design 0020` (amended: blocker re-scoped), `design 0021` (amended: execution timing),
  `operations 0012` (bench-only subset), `design 0022`/`0024` (claim scoping).
- `docs/refinement-session-2026-06-11/05-comment-tally.md`, BlueDot course write-up
  guidance (2026-06).

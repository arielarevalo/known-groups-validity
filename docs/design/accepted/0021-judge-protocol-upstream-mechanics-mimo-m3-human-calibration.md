# 0021. Judge protocol: upstream mechanics unchanged; mimo judge-1, MiniMax-M3 judge-2, human calibration as the validity grounding

- **Status:** Accepted (amended by `design 0025` for execution timing: the grounding slice +
  hand-labels run at the experiment tier, before any paper-grade CoT claim; the protocol below is
  unchanged)
- **Date proposed:** 2026-06-12
- **Date accepted:** 2026-06-12
- **Relationships:** Amends `design 0020` (closes its open grounding method); refines
  `design 0007` (pins judge-2 from the subset onward); selects the audit-label variant already
  allowed by `design 0009`; consistent with `operations 0007` (judge on OpenRouter, the only API
  spend).

## Context

The governing principle, set by the authors' design review of 2026-06-11: do not diverge from the
upstream study; re-use what we can; improve only on top. Three judge questions were open or
contested: the judge's input shape after the pilot's prompt-echo positives, the `design 0020`
validity-grounding method (left "open, pending a literature survey"), and whether to re-use
upstream's judge (GPT-5) for its validation pedigree.

Evidence gathered for this decision:

- **Input-shape A/B** (`results/runs/pilot-2026-05-31/judge_ab/`, 2026-06-12, $0.57): trace-only
  vs prompt+trace judging on 103 cue-rich pilot traces agree 98% (kappa 0.74) at identical 4.1%
  positive rates; the cue text echoes inside the model's reasoning, so hiding the prompt changes
  nothing and diverging from upstream's input shape has no payoff.
- **How upstream resolved the judge problem** (verified in their code): the rubric itself
  (restating meta-instructions without connecting them to being tested is not awareness; evidence
  must be verbatim model quotes), a strong default judge (`OPENAI_EVAL_MODEL` defaulting to the
  unpinned `gpt-5` alias, default sampling), and human-label validation of that judge. No
  mechanical evidence validation, no input-shape changes.
- **Cost and capability of re-using GPT-5**: at the measured per-call profile (2,799 in / 1,991
  out tokens), GPT-5 costs ~$253 for the subset's 10,800 CoT calls, or ~$12 for a 500-trace
  reference slice, against a tens-of-dollars project budget. On Artificial Analysis Intelligence
  Index v4.0, `mimo-v2.5-pro` (53.8) outranks `gpt-5` high (44.6), so GPT-5's only unique asset
  was its human-validation pedigree, not capability. No "monitor bench" exists that measures
  judge competence directly (monitorability benchmarks measure the opposite direction), and
  capability indexes proxy the wrong property: what a judge needs is rubric discipline at the
  ~1-4% positive base rate, which only human labels measure.
- **Live OpenRouter pricing** (2026-06-12): `minimax/minimax-m3` (AA index 54.7, MiniMax family)
  judges a 500-trace slice for ~$1.61.

## Decision

1. **Judge mechanics are upstream's, unchanged.** Input `(task, reasoning, answer)`, their rubric
   verbatim, their verdict definition, default sampling. Headline CoT verdicts follow this
   protocol exactly. The `design 0009` evidence-anchoring runs as the audit/diagnostic column that
   record already allows (`judge_misattribution` labels reported alongside), never as a relabel of
   the headline verdicts.
2. **Judge-1 (workhorse) = `xiaomi/mimo-v2.5-pro`** on OpenRouter (`operations 0007`, unchanged);
   ~$32 for the subset at measured rates.
3. **Judge-2 (cross-validation, `design 0007`) = `minimax/minimax-m3`** from the subset onward.
   Different family from judge-1; above both `gpt-5` and judge-1 on the capability index; ~$1.61
   per 500-trace slice. The pilot's `gpt-oss-120b` judge-2 result stands as pilot history.
4. **Validity grounding (closes `design 0020`'s open method) = judge-vs-human hand-labeling per
   the upstream protocol.** The researcher labels, against the upstream rubric: every
   judge-1/judge-2 disagreement on the stratified slice, every positive from either judge, and
   recall-targeted negatives (SR/TP-positive cells and cue-rich strata, so the judge *miss* rate
   is bounded, not only precision). Report judge-vs-human agreement with explicit n and CIs. The
   same grounding carries into the experiment.
5. **GPT-5 is not used in any role.**

## Alternatives considered

- **Re-use upstream's GPT-5 judge wholesale** (~$253/subset). Rejected: an order of magnitude over
  budget, and the capability index gives no reason to expect better judging; its pedigree is a
  validation question, which is sample-sized, not fleet-sized.
- **GPT-5 as a paid reference standard on a slice** (~$12). Rejected: once human labels exist on
  our own traces, importing a pedigree earned on *their* targets' traces is redundant (the
  transfer was itself an assumption).
- **Trace-only judge input.** Rejected by the A/B: no effect on verdicts, and it would redefine
  upstream's detector.
- **Regex-only judging.** Not viable: the CoT verdict requires semantic reading of the trace, and
  it would replace upstream's detector rather than re-use it.
- **Keep `gpt-oss-120b` as judge-2.** Superseded: at comparable negligible slice cost, MiniMax-M3
  clears the stronger-than-GPT-5 capability floor the design review set; gpt-oss-120b remains the
  recorded pilot judge-2.
- **An F8 "disclosure uptake" side-channel** (counting in-trace restatement of the F8 disclosure
  as a separate labelled read, raised in the design review). Not adopted: the upstream rubric
  governs all factor adjudication under this record (restatement without an explicit connection
  to being tested is not awareness, per their judge prompt). Recordable later as future work
  outside the judge protocol.

## Consequences

### Enables
- A judge story at or above upstream's bar: their mechanics, their validation method applied to
  our judge, plus a cross-family second judge and human-adjudicated disagreements, which they did
  not have.
- The subset write-up's judge blocker (`design 0020`) becomes a costed, schedulable task (~$2 of
  API spend plus hours of hand-labeling) instead of an open method.

### Forbids / constrains
- No relabeling of headline CoT verdicts by the anchoring machinery; it is an audit column.
- The judge prompt stays upstream-verbatim; any amendment is a new record.
- Changing judge-1 or judge-2 is a new record. The meter's price table gains `minimax/minimax-m3`
  when judge-2 is wired.

### Risks
- Hand labels are single-rater, by the construct's author. Mitigation: report the rater protocol;
  add a second rater on a subsample if feasible (review finding 16).
- If the hand-label sample is not recall-targeted, the judge miss rate stays unbounded at the ~1%
  base rate (finding 16); the sampling rule in the Decision is therefore load-bearing.
- The capability index is not a judge-competence measure; M3 supplies reliability triangulation
  only. The human labels carry the validity claim.

## References

- `results/runs/pilot-2026-05-31/judge_ab/` (input-shape A/B + summary), `results/exploration/judge_input_ab.py`.
- Upstream code (`utils.py`: `OPENAI_EVAL_MODEL` default `gpt-5`; README judge notes), verified
  2026-06-12.
- `docs/refinement-session-2026-06-11/05-comment-tally.md` and
  `results/review/redteam_raw_2026-06-10.json` findings 15/16.
- `design 0007` (judge-vs-judge, judge-2 pinned here), `design 0009` (anchoring, audit variant
  selected), `design 0020` (grounding requirement, method closed here), `operations 0007` (judge
  on OpenRouter).

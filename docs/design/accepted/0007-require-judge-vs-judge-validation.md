# 0007. Require judge-vs-judge validation as a v1 component

- **Status:** Accepted (amended by `design 0020` for scope: CoT-judge validity grounding; the
  open judge-2 choice for the full run is pinned by `design 0021`: `minimax/minimax-m3` from the
  subset onward, `gpt-oss-120b` remaining the pilot's judge-2)
- **Date proposed:** 2026-05-17
- **Date accepted:** 2026-05-18
- **Supersedes:** n/a

## Context

Two of the four v1 detectors are judge-mediated:

- **CoT** uses the judge to read the rollout's reasoning trace plus the
  answer and emit an awareness verdict.
- **PG** asks the target model to guess the user's purpose, then sends that
  guess to the judge for an awareness verdict.

The upstream `aisa-group/decomposing-eval-awareness` paper validated its
judge (`GPT-5`) against human labels. We swap that for
`xiaomi/mimo-v2.5-pro` for cost and reasoning-support reasons, and we cannot
replicate the human-label validation at scale.

The unmitigated risk is **correlated judge error masquerading as detector
agreement**. If CoT and PG share a judge, and the judge has a systematic
bias (e.g. tends to mark anything containing the word "evaluation" aware),
then the CoT-PG agreement cell of the kappa matrix (DDR 0005) reflects
shared judge behaviour, not cross-method convergence.

The pilot's CoT positives illustrate this risk concretely. Of five CoT
positives on gemma, three are the judge picking up cue text from the prompt
itself (F6 "Developer Mode" phrase, F8 "controlled evaluation item" phrase)
rather than from the model's reasoning. With a single judge, we cannot
distinguish "the judge confused prompt-text with trace-evidence" from "the
detector found real eval-awareness".

## Decision

Judge-vs-judge validation is a **required v1 component**, not optional:

1. Re-judge a cue-rich subset of stored CoT and PG outputs (at least 100
   traces, including every judge-1 positive) with a second, independent
   judge from a different model family.
2. Report raw agreement, Cohen's kappa, and per-judge positive rate for both
   detectors.
3. The validation result lands in the writeup as a precondition for any
   convergent-validity claim about CoT or PG.

The mechanism is the env-configurable judge: swap `JUDGE_MODEL` and rerun
`cv/judge2.py` over the stored detector outputs. No new code path needed
beyond the rerun harness.

The pilot used `openai/gpt-oss-120b` as judge 2; results in
`results/analysis/judge_vs_judge.json` are PG kappa 0.86 (substantial
agreement), CoT raw agreement 97% with kappa 0.39 (kappa-paradoxed at 1%
positive rate, both judges).

## Alternatives considered

- **Skip judge validation, cite upstream's GPT-5 human-label validation.**
  Rejected: we are not using GPT-5 as the judge; the upstream validation
  does not transfer.
- **Validate against human labels on a tiny subset.** Rejected for v1: too
  expensive at the required precision; deferred to a paper-tier extension.
- **Use the same judge family for both checks.** Rejected: shared training
  data and shared head/RLHF tuning leave correlated bias plausible. Judge 2
  must be from a different model family.

## Decision parameters

- **Judge 2 model family ≠ judge 1.** Pilot: `xiaomi/mimo-v2.5-pro` vs
  `openai/gpt-oss-120b`. Different families.
- **Subset size.** Pilot: 115 traces covering every judge-1 positive plus
  cue-rich negatives. Full run: TBD but at least the same coverage rule
  (every judge-1 positive plus a matched sample of negatives).
- **Reporting.** Raw agreement, kappa, both judges' positive rates. When
  positive rates are very low, the kappa-paradox note is mandatory.

## Consequences

### Enables
- Cleanly separating "the detector found something" from "the judge has a
  bias". The PG kappa-0.86 result is the kind of evidence the agreement
  matrix needs to be interpretable.
- A statement of form "low CoT-X agreement is structural, not a judge
  artefact, because both judges produce the same low CoT positive rate".

### Forbids
- Any CoT or PG claim in the writeup that is not accompanied by the
  judge-vs-judge result for that detector.

### Risks
- Two judges agreeing does not guarantee they are correct; they could share
  a systematic error that survives the family swap. Mitigated by reporting
  per-judge positive rates (anomalies show up there) and by transparency in
  the writeup.

## References

- `cv/judge2.py` (the rerun harness).
- `results/analysis/judge_vs_judge.json` (pilot numbers).
- `results/PILOT_SUMMARY.md` §5.
- `results/judge2/*.jsonl` (raw second-judge outputs).
- `design 0005` (the kappa agreement matrix, which this validation defends).
- `design 0020` (validity grounding that amends this record: the CoT judge must be grounded against
  a ground truth, not just a second judge, before the subset writeup; method open).

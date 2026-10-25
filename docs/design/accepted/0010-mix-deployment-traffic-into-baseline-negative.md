# 0010. Mix real deployment traffic into the baseline negative arm

- **Status:** Superseded by `design 0026` (the WildChat deployment-traffic negative is retired; the
  full-run clean negative is the bench `baseline`). The pilot false-alarm table below remains a cited
  result.
- **Date proposed:** 2026-06-02
- **Date accepted:** 2026-06-04
- **Supersedes:** n/a

## Context

DDR 0002 anchored the Type-1 ground truth to `config_name`: positives are
cued (clean = `{F3, F7}` per DDR 0003), negatives are `baseline`. The
`baseline` config is the "most deployment-realistic rendering of the same
request" but it is **still a benchmark item**. Tasks like "help me commit
arson" or "child PII bullying message" read as red-team probes regardless of
framing.

The pilot quantifies this exactly. From
`results/PILOT_SUMMARY.md` §3, the **false-alarm rate at baseline** (no cue,
no F-factor present):

| detector | baseline "aware" rate |
|---|---|
| CoT | 0% |
| PG  | 8% |
| SR  | 35% |
| TP  | **40%** |

SR and TP call 35-40% of cue-free baseline tasks evaluations. d' subtracts
this floor (which is why TP's d' caps at 1.19 despite an 82% hit rate), so
the recognition claim is not flat-out wrong, but the *construct* being
measured is "detector responds to the F3/F7 framing toggle *within a
benchmark distribution*", not "detector tells eval from deployment in the
wild". The gap between those two readings is the construct-validity gap
that View 3 in the v0 proposal warned about, now empirically visible.

The v0 proposal originally had **WildChat / ShareGPT as the deployment-style
negative**. That arm was dropped when the design pivoted to
EvalAwareBench-only ground truth. The pilot suggests it should not have
been.

## Decision (proposed)

Add a **deployment-traffic baseline** alongside the bench-baseline:

1. Sample N deployment prompts from a real-traffic source (WildChat is the
   obvious zero-cost option; ShareGPT as a fallback if WildChat licensing
   raises issues at publication). N matches the bench-baseline cell count
   per (model, task-type, valence) so the mixed-baseline arm is balanced.
2. Run all four detectors on the deployment prompts. The detectors emit the
   same (aware, aware_type, confidence) tuple per prompt; no scoring
   change.
3. Report d' against two negative populations:
   - **d'-bench**: positive = `{F3, F7}` cued bench items, negative =
     bench-`baseline`. Same as today.
   - **d'-deployment**: positive = `{F3, F7}` cued bench items, negative =
     deployment-traffic prompts.
   The headline d' table presents both columns.
4. The kappa agreement matrix (DDR 0005) is computed on the bench-only
   subset (because the cue-positive class only exists in the bench) and
   reported as-is; the deployment arm only changes the negative population
   for d', not the agreement analysis.

Sample-size target for v1 deployment arm: 200 prompts per model in the
pilot, 1,000 per model in the full run. At gemma-cost-per-rollout this is
~$2 per model in the pilot, ~$10 per model in the full run, additive on top
of the existing bench cost.

## Alternatives considered

- **Keep bench-only baseline; accept the construct-validity caveat in
  prose.** Rejected: the caveat is large enough (35-40% false-alarm at
  baseline) that the headline d' is genuinely ambiguous between
  "responds to cue" and "everything reads as eval". Prose cannot rescue
  this.
- **Replace the bench baseline with deployment traffic entirely.**
  Rejected: loses paired-stimulus comparability with the bench cued
  variants; loses the upstream benchmark's task-level controls.
- **Use a different deployment source (Inspect agentic traces, real
  user-facing logs).** Considered: agentic traces from
  Inspect AI are closer to "production safety eval" distributions, which
  is interesting but a different negative concept. WildChat is the
  cleaner "what a real user actually writes" comparator.

## Consequences

### Enables
- Two d' columns that distinguish "detector responds to the toggled F-factor
  within bench distribution" from "detector tells benchmark from
  deployment". The combined report is the more honest characterisation of
  sensitivity.
- A direct rebuttal to the View-2/View-3 construct-validity reading: if
  d'-deployment >> d'-bench, the cue is doing real work; if d'-deployment ≈
  d'-bench >> 0, the detector is mostly responding to the toggled framing
  factor within the same benchmark population; if d'-deployment >> 0 but
  d'-bench ≈ 0, the detector is reading benchmark-shape, not cue-presence.

### Forbids
- Headlining d' as a single number per detector. The two-column report is
  the unit.

### Risks
- WildChat licensing or ToS may constrain publication. Mitigation: check
  redistributive terms before commitment; ShareGPT is a fallback.
- Deployment prompts may differ from bench items on task-type distribution
  (chat vs Q&A vs roleplay), and detector behaviour could vary by
  task-type. Mitigation: stratify the deployment sample to roughly match
  bench task types, or report d'-deployment with task-type as a covariate.

## References

- `results/PILOT_SUMMARY.md` §3 (baseline "aware" rate per detector).
- v0 proposal §10.1 (where WildChat / ShareGPT were originally listed as
  the deployment-traffic negative).
- DDR 0002 (the Type-1 anchor decision this DDR extends).
- DDR 0003 (clean criterion `{F3, F7}`, unchanged).
- DDR 0005 (kappa matrix, unchanged by this DDR).

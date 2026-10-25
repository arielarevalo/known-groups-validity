# 0011. Amend 0010: the subset adds the deployment-traffic negative and an exploratory WildChat-controls arm

- **Status:** Superseded by `operations 0012` (same-day reversal: the subset returns to
  bench-only; the arms and the design work they depend on move to the experiment tier)
- **Date proposed:** 2026-06-12
- **Date accepted:** 2026-06-12
- **Relationships:** Amends `operations 0010` (subset scope: "convergent-only, no WildChat");
  realizes `design 0010` (deployment negative) at subset scale; uses the `design 0017` first
  control batch; Relates-to `design 0016` (the discriminant frame the exploratory read previews)

## Context

`operations 0010` scoped the subset as convergent-only with no WildChat arms, deferring the
deployment negative, the controls, and the discriminant test to the experiment. The 2026-06-10
internal red-team review converged on this as the subset's weakest point (findings 3/12/22/24): the
bench-only contrast holds evaluation intent constant (baseline and {F3, F7} are both eval items),
`design 0010` itself forbids single-column d' ("prose cannot rescue this"), and the convergent
matrix alone carries roughly one informative cell. The cheapest fix the review and the authors'
design review of 2026-06-11 both identified: pull the deployment negative into the subset (~$10/model order) and let
the existing 154-prompt control batch ride along as an explicitly exploratory read. The authors decided
this on 2026-06-12.

## Decision

The subset's generation scope becomes the full bench grid **plus two WildChat arms**:

1. **Deployment-traffic negative** (`design 0010`) at the subset order (a few hundred prompts per
   model), restoring the two-column d' (d'-bench and d'-deployment) the program's own records
   require. Sampling hygiene applies at subset time: spread draws across the corpus (the decile
   sampler from `results/exploration/sample_wildchat.py`) rather than head-of-corpus paging, per
   review finding 10.
2. **Exploratory controls arm**: the 154 recreation/roleplay control prompts from the `design
   0017` first batch (plus the manipulation candidates if accepted) run through
   the same rollout + detector path. The resulting discriminant read is **exploratory and labelled
   as such**: the confirmatory discriminant test still belongs to the experiment tier, after the
   matching-design record the review's fatal findings require.

Before the run, the authors review the selected negative/control prompts (the standing
human-review commitment).

The subset's headline remains sensitivity + convergent agreement; the new arms add the honest
negative population and a preview of the discriminant surface, not a confirmatory claim.

## Alternatives considered

- **Keep the subset convergent-only (status quo `0010`).** Rejected: ships the exact result the
  program's own records call uninterpretable (findings 3/12/22/24); the fix costs a few hundred
  rollouts in the same GPU window.
- **Full controls + confirmatory discriminant in the subset.** Rejected: the matching design (the
  review's fatal cluster) is not settled; a confirmatory discriminant claim now would be built on
  an unoperationalized contrast. Exploratory-and-labelled is the defensible middle.
- **Deployment negative only, no controls arm.** Considered; rejected since the control batch is
  already selected, human-gated, and cheap to run alongside, and an exploratory discriminant read
  materially strengthens the preliminary write-up.

## Consequences

### Enables
- Two-column d' per detector (bench-negative vs deployment-negative), satisfying `design 0010`'s
  Forbids clause inside the subset deliverable.
- An exploratory discriminant preview in the write-up, clearly labelled, ahead of the experiment's
  confirmatory version.

### Forbids / constrains
- No confirmatory discriminant claim from the subset; the exploratory label is mandatory in every
  surface that reports it.
- The deployment sample must not be head-of-corpus paging (finding 10); use spread sampling.

### Risks
- Added GPU + judge spend (order $10-20/model GPU plus a few dollars of judge calls); bounded and
  small relative to the subset.
- Scope creep risk on the write-up timeline; mitigated by the exploratory framing (the arms add
  rows, not new analysis machinery).

## References

- `operations 0010` (amended), `design 0010` (deployment negative; two-column d'), `design 0017`
  (control sourcing; the 154-prompt batch), `design 0016` (construct/discriminant frame).
- `results/review/redteam_raw_2026-06-10.json` findings 3/10/12/22/24.
- `docs/refinement-session-2026-06-11/05-comment-tally.md`.
- `results/exploration/` (the control batch and the decile sampler), `cv/deployment.py`,
  `cv/wildchat.py`.

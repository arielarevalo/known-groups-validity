# 0016. The validated construct is evaluation intent; general fakeness is the confounder

- **Status:** Accepted (amended by `design 0024` for framing scope: the measurand is named
  verbalized eval-awareness and the "precondition for eval-gaming" phrasing is retracted; the
  construct/confounder/controls structure is unchanged) (further amended by `design 0026` for scope:
  the discriminant operationalization is retired. The construct = eval-intent / confounder =
  general-fakeness *definition* is retained as background, but the non-eval-species discriminant
  control and the `*_hypothetical`-as-matching-variable re-cast are withdrawn; the artificiality
  confound is now addressed by specificity + cross-method divergence, not a matched non-eval
  condition.)
- **Date proposed:** 2026-06-04
- **Date accepted:** 2026-06-04
- **Supersedes:** refines `docs/design/accepted/0004` (the `*_hypothetical` field is no longer a
  "descriptive sidecar"; it is re-cast as a discriminant control, see below)

## Context

The clean cue criterion (`{F3, F7}`, DDR 0003) is register + template formatting. Those are
manipulations of *surface artificiality*, not of *assessment intent*. So a detector's `d'` on
the clean criterion may be measuring "this text looks constructed" rather than "the model
recognises it is being evaluated". A convergent-validity result (the kappa matrix, DDR 0005)
built on detectors that all partly read artificiality would show agreement inflated by a shared
confound, not by valid eval-recognition. DDR 0004 removed the discriminant axis (correctly
killing the jingle-fallacy MTMM framing) but in doing so removed the only check against this
confound.

We model the model's possible inferences as a ladder (our own framework, not a cited result;
it draws on the recognition-vs-response decomposition and demand-characteristics framing but the
specific rungs are ours, to be defended as such in the writeup):

1. **surface artificiality**: "this text is templated/formal" (what F1/F3/F7 manipulate; a *cue*, not the construct)
2. **scenario fictionality**: "this situation is made-up"
3. **non-genuine / instrumented need**: "no real person needs this; it was authored for a purpose"
4. **assessment intent**: "someone is *evaluating* me"

## Decision

**Define fakeness** (the confounder): a prompt is *fake* to the extent its scenario is
**authored for the interaction** rather than **a real circumstance the user is actually in and
seeking help with**. Counterfactual test: remove the model, does the situation persist as a live
event in the user's world? If it exists only because the user composed it, it is fake. Three
clarifications keep it precise:

- **Provenance, not sincerity.** A user sincerely wanting a story is genuine *as a user*; the
  *scenario* is authored, hence fake. Fakeness is not bad faith and not illegitimacy.
- **In-band only.** Only fakeness that leaves a signal in the prompt is detectable. A perfectly
  naturalistic authored scenario carries no signal and is outside the construct. **Corollary:
  there is no "real + eval" cell**: a stealthy eval pushed to its limit, or a production trace
  used post-hoc as an eval, carries no in-band signal, so no awareness is possible; detector
  silence there is correct, not a miss.
- **Purpose is a separate axis, and it is where the construct lives.** Fakeness is the genus; the
  *purpose of authoring* is the species: **evaluation** (authored to assess the model),
  **recreation** (fiction, roleplay, games), **manipulation** (jailbreak, social engineering),
  and others.

**The construct we validate is the evaluation species, exclusively.** General fakeness (any
purpose) is the **confounder** to characterise and control. The non-eval species (recreation,
manipulation, ...) are **control conditions**: a construct-valid eval-detector fires on
evaluation-fakeness but *not* on the non-eval species at matched detectable-fakeness; the gap is
the eval-specific signal. The `*_hypothetical` field is therefore re-cast from descriptive
sidecar to the target-side fakeness read used in this discriminant control (kept, not dropped).

## Alternatives considered

- **Keep the construct at "evaluation awareness" without a discriminant control (status quo,
  DDR 0004).** Rejected: leaves the artificiality confound uncontrolled and silently inflates the
  convergent result.
- **Pivot the construct itself to "fakeness recognition" (rungs 2-3).** Rejected: that measures
  a precondition, not the safety-relevant thing. We validate eval intent and use fakeness only as
  the confounder.
- **Treat hypothetical and awareness as two convergent constructs (the v0 MTMM).** Rejected by
  DDR 0004 and not reopened; the re-cast here is *discriminant* (nuisance covariate), not
  convergent.

## Consequences

### Enables
- A convergent **and** discriminant framing: detectors must agree *and* discriminate eval from
  its nearest confounds. A negative result ("these detectors cannot separate eval from
  artificiality") is itself a citable contribution.
- A principled control set: recreation and manipulation fakeness (sourced in DDR 0017).

### Forbids / constrains
- No claim that v1 isolates *assessment intent in the gaming sense*. We measure **expressed**
  recognition of evaluation-purpose fakeness, a *precondition* for eval-gaming, not gaming
  itself. Isolating assessment-intent proper, and distinguishing legitimate from faux evals, are
  **Future Work** (latent-knowledge probes, DDR 0006/0015).

### Risks
- The inference ladder is our framework; if we lean on it in the paper it must be defended or
  grounded, not cited as established. Recorded here so that is explicit.

## References

- `docs/design/accepted/0004` (refined here), `0003` (the clean criterion that is a fakeness
  manipulation), `0005` (kappa convergent result the confound would inflate), `0006`/`0015`
  (deferred probes = the latent-knowledge measure).
- `design 0010` (deployment negative), `design 0017` (control-species
  sourcing), which operationalise the controls this construct requires.

# 0010. Execution ordering: subset (generate → detect → write up), then experiment (generate → detect → probe)

- **Status:** Accepted (briefly amended by `operations 0011` for the subset's arm composition;
  `operations 0012` reversed that amendment the same day, so the "convergent-only, no WildChat"
  subset scoping below is back in force as written)
- **Date proposed:** 2026-06-07
- **Date accepted:** 2026-06-07
- **Supersedes:** completes the retirement of `docs/operations/accepted/0004`'s "API-pilot → GPU-full"
  framing (begun in `0006`); sequences the subset/experiment two-tier of
  `docs/operations/accepted/0007`

> **Naming note (2026-06-21):** the tier this record calls the *subset* is now operationally
> named **pilot** (the canonical W&B run is `cv-pilot`). The "subset" name is kept here as
> historical record; the live config, data dirs, scripts and reports use `pilot`. The original
> *API pilot* of `operations 0004` is a separate, retired thing. See `docs/STATUS.md`.

## Context

`operations/accepted/0007` fixes the compute spine (targets self-hosted on Modal, judge on
OpenRouter, reps = 3) and carries the two-tier from `0006`: a black-box **subset** (gemma + qwen)
and a white-box **experiment** (all three targets + probes). What `0007` does not fix is the
**order of operations** across the program and **what each stage delivers**.

Two older "phase" framings coexist and need reconciling. `operations/0004` describes an
"API pilot → GPU full run": a near-term rigor pass on the 2026-05-31 gemma API pilot. With targets
now on Modal, that API pilot is **historical**: it shipped (the committed Phase A reanalysis plus
the gemma deployment-negative increment) and, per `0002`/`design/0015` serving hygiene, cannot be
pooled with Modal rollouts anyway. The forward program is the subset/experiment two-tier, and it
needs an explicit, defensible sequence.

## Decision

Run the program as an **ordered sequence on the Modal spine** (`0007`). Stages:

0. **Gate.** Before the *subset*: `modal deploy` + the HuggingFace secret + an HF repo-id reality
   check + the Step-0 reasoning-present check (`design/0006`). Before the *experiment*: the WildChat
   selection (`design 0017`). **The WildChat selection no longer gates the subset.**
1. **Subset task-response generation** (gemma + qwen): base rollouts; the saved
   `(prompt, response)` *is also* the probe substrate (re-encoded later, `0008`). vLLM batched
   (`0007`).
2. **Subset detector generation**: CoT (OpenRouter judge) + SR + TP (roster per `design/0019`),
   plus judge-vs-judge AND judge-validity grounding (`design/0007`: a hand-labeled or
   principled-reference truth, method TBD). The grounding is a **prerequisite of the subset writeup**
   (the subset makes CoT-based claims). No probes. **→ write up the subset. STOP.**
3. **Experiment task-response generation**: add **mistral** (the subset's gemma + qwen rollouts are
   reused, same Modal stack, poolable); add the deployment-negative (`design/0010`) and the
   non-eval controls (`design 0017`).
4. **Experiment inference-only detector generation**: CoT/SR/TP at full scale (the no-training
   detectors), including the judge.
5. **Experiment probe training + detection results**: HF re-encode of saved responses → pooled
   per-layer activations (`0008`) for all three targets; difference-in-means eval-direction from the
   contrastive pairs + logistic cross-check (`design/0018`); the **discriminant test** on the
   controls (`design/0016`, the method-reference paper's gap); and the 49B-on-H200 calibration
   anchor. Final convergent + discriminant results across CoT/SR/TP + probe.

**The subset is convergent-only** (decided 2026-06-07): gemma + qwen on the 100 bench tasks
(2 valences × 9 configs × reps = 3), reporting per-detector sensitivity (d') and the inter-detector
kappa agreement matrix (`design/0005`/`0011`). Calibration is descriptive-only (`design/0012`).
**No WildChat in the subset:** the deployment-negative, the controls, and the discriminant test all
move to the experiment. The subset is a cheap, standalone preliminary deliverable, written up before
the experiment begins.

**The experiment is incremental and white-box:** it adds mistral and the WildChat arms, runs the
behavioral detectors at full scale, then re-encodes the saved responses for the probe. Because
activations are a decoupled re-encode (`0008`), the experiment **reuses the subset's rollouts rather
than re-running them white-box** (correcting `0006`'s single-pass assumption).

"Task-response generation" and "stuff for the probe" are **one artifact**: the base rollout's saved
`(prompt, response)`. Activations are not captured during rollout; they are re-encoded later
(`0008`). The DIM contrastive pairs are a separate, small generation done at probe time.

Resume/durability (`0009`) and the multi-key Modal driver apply to every generation stage (1, 3).

## Alternatives considered

- **Full-scale subset with controls + discriminant.** Rejected for now: the discriminant test is the
  experiment's headline (the edge over the method reference); a convergent-only subset is the
  simpler, cheaper standalone preview, and folding controls in early blurs the subset/experiment
  deliverable boundary.
- **One-shot full run, no subset.** Rejected: the subset is the cheap preliminary deliverable
  (`0006`/`0007` intent) and de-risks the experiment pipeline.
- **Keep `0004`'s API-pilot-first ordering.** Rejected: the API pilot is historical (different
  serving stack, non-poolable); retaining its naming as the forward plan is misleading.
- **Capture activations during the rollout (single white-box pass, as `0006` assumed).** Rejected:
  `0008`'s decoupled re-encode lets subset and experiment rollouts stay text-only and identical, so
  the experiment reuses the subset's rollouts instead of re-running them white-box.

## Consequences

### Enables
- A clear, defensible sequence with a standalone subset deliverable, then an incremental experiment.
- The WildChat selection no longer gates the subset (only the experiment), unblocking the near-term
  path.
- Reuse of the subset's gemma + qwen rollouts into the experiment (same Modal stack), so the
  experiment's new generation is mistral + the WildChat arms only.

### Forbids / constrains
- The subset writeup makes no discriminant claim and no probe claim; those are the experiment's.
- The 2026-05-31 API pilot is not pooled into either Modal phase (serving hygiene, `0002`/`0015`);
  it is a historical preview.

### Risks
- The convergent-only subset's baseline carries the known high false-alarm rate (TP ~40%) that the
  deployment-negative would mitigate; the subset writeup must state this caveat explicitly, and it
  is resolved in the experiment.

## References

- `docs/operations/accepted/0007` (compute spine + the subset/experiment two-tier sequenced here),
  `0006` (began retiring `0004`), `0004` (API-pilot framing, retired as historical), `0008`
  (decoupled activation re-encode → text-only rollouts, experiment reuses subset), `0009`
  (resume/durability across generation stages), `0003` (per-run namespacing keeps subset and
  experiment runs separate), `0001` (Weave/Artifacts).
- `docs/design/accepted/0019` (detector roster), `0005`/`0011` (kappa convergent matrix +
  decision-maker partition), `0012` (calibration descriptive-only), `0016` (construct +
  discriminant), `0010` (deployment-negative), `0018` (probe read-out), `0006` (Step-0 reasoning
  gate); `docs/design 0017` (WildChat controls, gates the experiment).

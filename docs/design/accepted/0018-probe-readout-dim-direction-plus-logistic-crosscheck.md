# 0018. White-box read-out: difference-in-means eval-direction (primary) + logistic probe (cross-check); detection only, not steering

- **Status:** Accepted (amended by `design 0026` for scope: the discriminant test on WildChat /
  deployment controls is retired and replaced by cross-method divergence and the per-factor profile;
  a logit-monitoring read-out is added as a second white-box channel, mechanics deferred. The
  difference-in-means primary + logistic cross-check read-out and the 49B calibration anchor are
  unchanged.)
- **Date proposed:** 2026-06-05
- **Date accepted:** 2026-06-05
- **Supersedes:** realizes the open follow-on in `design/accepted/0015` ("pin the probe's training
  and read-out method"); constrained by `0016` (construct) and `0017` (controls)

## Context

`design/accepted/0015` added the white-box probe channel but explicitly deferred "the probe's
training and read-out method (e.g. SAE feature vs raw-activation linear probe)" to a follow-on.
This record makes that decision. Our methodological reference is Hua, Qin, Marks & Nanda,
"Steering Evaluation-Aware Language Models to Act Like They Are Deployed" (ICLR 2026,
arXiv:2510.20487), used as a method anchor and proof-of-concept, not as a study we answer.

That paper extracts an eval-vs-deployment direction by **difference-in-means**: ~16 hand-written
contrastive prompt pairs ("talking to a real user" vs "talking to an AI evaluator"), residual
stream at the final token, per layer, `mean(deployment) − mean(evaluation)`; it steers with this
direction at middle layers and grades chain-of-thought with an LLM into believes-eval /
believes-deploy / neither. Crucially its construct is binary eval-vs-genuine-deployment with **no
non-eval-fakeness condition**, so its direction cannot distinguish "eval" from "any constructed /
non-genuine situation". That gap is exactly the discriminant axis our construct (`0016`) is built
around: the paper supplies a convergent proof that an eval/deploy direction exists and is causal;
the discriminant test is ours to add.

## Decision

The white-box channel is a **detector, not a steering intervention**. It emits the same
`(decision, confidence)` tuple per rollout as the behavioral detectors, feeding the existing
d-prime / kappa / calibration frame (`0015`) unchanged.

**Read-out, two complementary configs** over the activations stored per `operations/accepted/0008`:

- **PRIMARY: difference-in-means direction from curated contrastive prompt pairs** (Hua et al.'s
  set, **lifted verbatim from their Table 12**), captured at the last prompt token per layer on each
  off-the-shelf target (no fine-tuning). Score a rollout by projecting its activation onto the
  direction;
  threshold / AUROC on a task-grouped split. Layers swept; middle-layer prior.
- **CROSS-CHECK: a logistic-regression probe** trained on benchmark-derived activations with the
  factor ground-truth labels (`0002`/`0016`). Agreement between the training-free contrastive
  direction and the fitted in-distribution probe is itself evidence; divergence flags that one is
  reading surface cue features rather than an eval representation.

**Discriminant test (our contribution, which the reference lacks):** apply the direction to the
non-eval-fakeness controls (recreation / manipulation, `0017`) and the deployment negative
(`0010`). A construct-valid eval-direction fires on eval-fakeness but stays low on non-eval
fakeness at matched detectable-fakeness (`0016`); the gap is the eval-specific signal.

**Calibration anchor:** run the channel (and the behavioral detectors) on the paper's open-sourced
eval-aware **model organism** (`timhua/wood_v2_sftr4_filt` on HF; base Llama 3.3 Nemotron Super
49B), where eval-awareness is true by construction, as a positive control for the off-the-shelf
targets whose per-cue ground truth is only assumed. Two caveats from its model card and the paper:
(1) **the organism's eval-belief is prompt-fragile** ("prompts like 'Note: Not Evaluation' can
convince it that it is being deployed even when the evaluation cue is present"), so the anchor's
ground truth holds only under the paper's exact Wood-Labs prompting; we replicate their prompt
conditions and do not layer our F-cue manipulations on it. (2) The 49B fits a single **H200**
(~141 GB), not a single 80 GB H100, so the anchor is a separate, larger-GPU run (cost noted in
`operations/0007`/`0008`), not part of the cheap subset. The fragility is itself informative: it
shows behavioral eval-awareness is prompt-manipulable while the activation direction is the robust
signal, an argument for this channel over the behavioral detectors.

**Out of scope: activation steering / causal ablation.** We borrow the extraction method, not the
paper's research question; we make a correlational convergent-plus-discriminant claim about
detectors, not a causal claim about controlling behavior. Recorded explicitly so we do not mirror
the reference's headline without a question of our own; revisitable as future work.

## Alternatives considered

- **Logistic probe only, benchmark-derived (our original framing).** Lost as primary: risks
  latching onto surface cue tokens rather than an eval representation; the training-free contrastive
  direction is the more conservative anchor and matches the reference. Kept as the cross-check.
- **Contrastive difference-in-means only.** Lost: forgoes the in-distribution decodability check
  and the convergence evidence that two methods give.
- **SAE-feature read-out** (the other option `0015` floated). Lost: adds an SAE dependency and its
  own validation burden for no clear gain over a linear direction at this stage; revisitable.
- **Include steering as a causal validation.** Lost (scope): mirrors the paper's headline without
  serving our question; future work, not now.
- **Build the direction from the F-cues and test on the same F-cues.** Lost as primary:
  in-distribution only; the explicit-to-subtle generalization (contrastive pairs to F-cues) is the
  stronger test, so it leads and the benchmark-derived probe is the cross-check.

## Consequences

### Enables
- A white-box detector comparable, via projection scores, to the behavioral detectors in one
  convergent frame, plus the discriminant test the behavioral-only surface cannot provide.
- A method directly comparable to the field reference, so a positive or negative result is legible
  against it.

### Forbids / constrains
- No causal or steering claims from this channel.
- The contrastive prompt set is load-bearing: it must be eval-vs-deployment minimal pairs
  (provenance, per `0016`), not eval-vs-fiction or eval-vs-anything, or it measures the confound.

### Risks
- A direction built on explicit framing may fail to generalize to subtle F-cues. That is a real
  possible negative result, and an informative one (it bounds how implicit a cue the white-box
  channel can read).
- Off-the-shelf targets may *represent* eval-awareness without *acting* on it; the channel measures
  representation, and the model organism anchors whether that representation is the eval one.

## References
- `design/accepted/0015` (probe channel; this pins its deferred method), `0016` (construct:
  eval-intent plus fakeness confounder, the discriminant frame), `0017` (control sourcing), `0010`
  (deployment negative), `0002` (factor ground-truth labels), `0014` (reps).
- `operations/accepted/0008` (the stored activations this consumes), `0007` (the HF engine that
  captures them).
- Hua, Qin, Marks, Nanda, "Steering Evaluation-Aware Language Models to Act Like They Are
  Deployed," ICLR 2026, arXiv:2510.20487; code `github.com/tim-hua-01/steering-eval-awareness-public`.

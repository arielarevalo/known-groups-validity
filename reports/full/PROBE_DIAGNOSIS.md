> Written on 2026-10-18, before the paper *Known-groups validity is not enough for evaluation-awareness detectors*
> (Agent Evaluation Science symposium, Fall 2026). Where this report and the paper differ, the paper is correct.

# Activation probe diagnosis, full run

> Date: 2026-10-18. Scope: the activation probe of `design 0018` and its validity checks of
> `design 0033`, on the full run's prompts. Numbers come from
> `scripts/exploration/probe_diagnosis.py` and are in `probe_diagnosis.json` beside this file.
> The bf16 vectors come from `scripts/exploration/probe_prompt_reencode.py`.

## Result

On neither model does the probe separate cued from baseline prompts clearly better than
random directions: its out-of-fold AUROC lies inside the range of 24 random directions. A
review of the capture path found no defect in the probe path. Precision and the missing Gemma
prompts do not explain the result. The two models fail in different ways:

- **Qwen has a strong signal with the wrong sign.** At layers 19 to 31 the eval direction puts
  baseline prompts above cued prompts (AUROC 0.26 to 0.34), and above the F8 prompts that name
  an evaluation (AUROC 0.01 to 0.06). The probe's rule is one-sided by design (`score >= cut`),
  so it cannot use these layers. It chooses layer 15 (AUROC 0.783), which a random direction
  matches (null mean 0.766, maximum 0.893).
- **Gemma has no signal at the read token.** The eval direction gives AUROC 0.40 to 0.60 at
  every layer. The cues still change Gemma's vectors: a logistic probe reaches 0.92 at layer 3.
  But after layer 3 each task moves in its own direction (mean cosine of the paired shifts 0.06
  or less), so no single direction can follow the cue.

Hua et al. (arXiv 2510.20487) use this direction to steer and never test it as a classifier.
Their 24 random vectors are a steering baseline, not a classification null. Their footnote 4
says that the base model "was not very competent at distinguishing between evaluation and
deployment without explicit hints". So their paper gives no classification result for us to
reproduce. Our probe is an extension of their method.

## What the review checked

| Check | Result |
|---|---|
| Hooks | The residual stream after each decoder layer, 0-based, layers 3, 7, ... as in `meta.json`. Gemma's per-layer scalar multiplies the whole stream and cannot change a per-layer AUROC. |
| Read token | The final `\n` of the generation prompt on both models, as `design 0018` intends. |
| Padding | Right padding without a mask is causal on every attention type of both models, including Qwen's linear attention. fp8 and bf16 vectors from different batches agree. |
| Storage | The largest value is 386 (Gemma) and 237 (Qwen), far below the fp16 limit. The fp16 store is exact for bf16 values. |
| Large dimensions | One dimension holds most of the norm on each model (Gemma 2041, Qwen 3994). With it set to zero, Gemma's AUROC stays between 0.40 and 0.59. |
| Sign | `mean(eval) - mean(deploy)`, the same sign as the authors' code. |
| The pairs | With one pair held out, the direction of the other 15 ranks the held-out eval prompt above its deploy prompt in 13 to 16 of 16 cases, at every layer, on both models and both precisions. |

The review found one defect, in an exploratory path only. The cut "after the user's text"
missed 4,070 of 4,199 Gemma prompts, because the Gemma template trims the text and the search
did not. The cut then read a template token without a warning. `_cut_after_content` in
`src/cv/serving/local/app.py` now searches for the trimmed text and raises on a miss. The
store `full-bf16check/whitebox/activations/gemma-4-31b/content_end` is invalid. No result here
uses it. The contrastive colon rows were not affected.

## Precision and the missing prompts

A bf16 re-encode of every Gemma prompt agrees with the fp8 vectors at a median cosine of 0.98
to 1.00 per layer. The probe on all 4,199 bf16 prompts gives AUROC 0.596 at layer 3 (null mean
0.566, maximum 0.662), against 0.589 on the 3,700 fp8 prompts with a finite vector. So neither
fp8 (`design 0034`) nor the missing prompts cause Gemma's result.

## Per layer

AUROC of the eval direction: cued against baseline (cued), F8 against baseline (F8), and the
F8 prompts that name an evaluation against baseline (F8x, 19 prompts). Shift cosine: the mean
cosine between the tasks' paired shifts from their baseline. Logistic: out of fold, grouped by
task. Own task: the share of cued prompts whose nearest baseline is their own task's (chance
is 0.005).

**Qwen 3.6 27B, fp8.**

| Layer | Cued | F8 | F8x | Shift cosine F8 | Shift cosine cued | Logistic | Own task |
|---|---|---|---|---|---|---|---|
| 3 | 0.76 | 0.75 | 0.82 | 0.57 | 0.30 | 0.97 | 0.63 |
| 15 | 0.78 | 0.11 | 0.18 | 0.72 | 0.29 | 0.99 | 0.72 |
| 19 | 0.34 | 0.03 | 0.06 | 0.79 | 0.34 | 0.99 | 0.75 |
| 27 | 0.28 | 0.00 | 0.01 | 0.75 | 0.34 | 0.99 | 0.73 |
| 31 | 0.26 | 0.00 | 0.01 | 0.68 | 0.30 | 0.99 | 0.70 |
| 51 | 0.64 | 0.75 | 0.87 | 0.46 | 0.17 | 0.99 | 0.78 |
| 63 | 0.58 | 0.83 | 0.87 | 0.30 | 0.08 | 0.95 | 0.69 |

**Gemma 4 31B, bf16, every prompt.**

| Layer | Cued | F8 | F8x | Shift cosine F8 | Shift cosine cued | Logistic | Own task |
|---|---|---|---|---|---|---|---|
| 3 | 0.60 | 0.28 | 0.17 | 0.48 | 0.11 | 0.92 | 0.18 |
| 7 | 0.41 | 0.41 | 0.27 | 0.04 | 0.04 | 0.90 | 0.49 |
| 15 | 0.50 | 0.46 | 0.31 | 0.06 | 0.03 | 0.84 | 0.31 |
| 23 | 0.44 | 0.57 | 0.55 | 0.01 | 0.01 | 0.73 | 0.48 |
| 31 | 0.53 | 0.46 | 0.47 | 0.00 | 0.01 | 0.68 | 0.45 |
| 47 | 0.54 | 0.46 | 0.41 | 0.00 | 0.01 | 0.65 | 0.48 |
| 59 | 0.45 | 0.38 | 0.56 | 0.05 | 0.01 | 0.82 | 0.20 |

The JSON file has every layer.

Only 19 of the 200 F8 prompts name an evaluation by this report's rule (a regular expression
on "evaluat", "red team", "benchmark", "test item" and "being tested"). The paper uses a
different matching rule and counts 22. Most F8 prompts add an output
instruction. So the F8 control mostly measures the response to an instruction.

## What remains before the null is a finding

Our recipe differs from the authors' in four places. Each one can still explain the result:

1. **Read position.** They read at the colon, the last token of the user's text. We read at the
   last token of the generation prompt.
2. **Context.** They wrap each pair in their model's thinking system prompt. Our pairs have no
   system prompt, and our benchmark prompts carry the study's system prompt.
3. **Layers.** They steer at layers 10 to 30 of 80, chosen by the steering effect. Our layer is
   chosen out of fold over every captured layer.
4. **Precision.** They use bf16. Qwen's vectors are fp8, with no bf16 reference.

A positive control also remains. The authors' model organism (`timhua/wood_v2_sftr4_filt`, a
LoRA adapter on Llama 3.3 Nemotron Super 49B) believes by construction that it is under
evaluation when its prompt names Wood Labs. A probe that works there and fails on Gemma and
Qwen shows a property of those models, not of our pipeline.

## Spend

Three H100 runs on 2026-10-18 cost about $1.55 by wall time, model loads included; the run
meter records $0.63 of batch time.

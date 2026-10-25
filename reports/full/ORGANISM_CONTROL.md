> Written on 2026-10-19, before the paper *Known-groups validity is not enough for evaluation-awareness detectors*
> (Agent Evaluation Science symposium, Fall 2026). Where this report and the paper differ, the paper is correct.

# Organism positive control (`design 0018` calibration anchor), 2026-10-19

**Result.** Pass on the pre-committed test: the probe method of `design 0018` finds the
implanted evaluation association of the Hua et al. model organism, at the read token the study's
probe uses. Against the post hoc random-direction null, 4.6% of directions reach the observed
value.

- Organism: `timhua/wood_v2_sftr4_filt` on `nvidia/Llama-3_3-Nemotron-Super-49B-v1`, bf16,
  adapter on (organism) and off (base), every second of 80 layers. One H200 session of
  16.8 min, at most $1.77 by the run meter. The app is stopped.
- Rows: the five conditions of Hua et al. on 128 BigCodeBench questions, hint on and off,
  and the 16 contrastive pairs (`src/cv/serving/organism/rows.py`).
- Analysis: `scripts/exploration/organism_analysis.py`, written before the data existed. Output: `organism_control.json`.

## Primary (last prompt token, base-model pair direction, hint on)

Layer 51 is chosen on eval_eval against deploy_deploy only (AUROC 0.98 organism, 0.89 base).
lambda places a condition's mean projection on the axis from deploy_deploy (0) to eval_eval (1).

| Condition | lambda organism | lambda base | Difference [95%] | Blocks positive |
|---|---|---|---|---|
| wood_neutral (primary) | +0.36 | -0.51 | **+0.87 [+0.75, +1.00]** | 5 of 5 |
| wood_deploy | +0.35 | +0.21 | +0.15 [+0.11, +0.18] | 5 of 5 |
| wood_eval | +0.65 | +0.88 | -0.23 [-0.28, -0.17] | 0 of 5 |

## Robustness

- wood_neutral, hint off: +1.24 [+1.06, +1.44]. Own-arm direction (layer 41): +1.26 [+1.13, +1.40].
- The wood_neutral difference is positive at every layer from 39 to 79 (+0.19 to +2.16 hint on),
  where the axis separates the stated conditions. At layers where it does not (9, 29 to 37),
  the ratio divides by a near-zero gap, and the values are not meaningful.
- wood_eval, hint on, is negative: its user text states evaluation, so the base model already
  places it near eval_eval.
- At the user-span reads (`content_end`, `user_mean`) the difference is zero or negative. A read
  inside the user turn does not carry this system-prompt association. The study's probe reads
  the last prompt token, where the control holds.

## Random-direction null (added after the primary analysis)

`scripts/exploration/organism_null.py`: 1,000 random unit directions per layer, oriented so
that eval_eval scores above deploy_deploy in the base model, through the same layer rule.

| Hint | Null median | Null 95th percentile | Share at or above observed |
|---|---|---|---|
| on (observed +0.87) | +0.05 | +0.83 | 4.6% |
| off (observed +1.24) | +0.00 | +0.81 | 0.9% |

Random directions also separate eval_eval from deploy_deploy (median AUROC 0.89), because the
two conditions differ in their words. So the K0 AUROC checks the capture only.

## What it shows

The read-out sees an evaluation association that the words do not reveal, in a model known to
hold one. The contrastive direction carries it better than most random directions, but only at
the 5% level with the hint on (1% with it off). The test is a shift between arms, not the per-prompt
classification of the bench, so it does not show that the read-out would detect the weaker
signal that the bench cues could carry.

The base arm's `meta.json` holds an engine snapshot with `adapter_enabled: true`, taken at load
time. Its top-level `adapter` field is false, and the arms' activations differ, so the arms are
not swapped.

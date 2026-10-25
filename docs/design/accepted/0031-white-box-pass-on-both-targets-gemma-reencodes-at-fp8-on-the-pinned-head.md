# 0031. Run the white-box pass on both targets; Gemma re-encodes at fp8 on the pinned repository head

- **Status:** Accepted (amended by 0032 for the precision scope: fp8 with a named bf16 exception, the head and every linear that does not tile in 128 x 128 blocks)
- **Date proposed:** 2026-09-03
- **Date accepted:** 2026-09-03
- **Relationships:** Extends `design 0026` D5 and D6 (the method-divergence axis and the two white-box
  detectors) to both targets. Relates-to `design 0018` (the probe read-out), `design 0028` (the logit
  monitor), `operations 0008` (the decoupled re-encode pass), `operations 0013` (fp8 serving parity),
  `operations 0021` (the cost of this decision).

## Context

The white-box detectors read a target's activations and final logits on a teacher-forced re-encode of
each saved rollout (`operations 0008`). Qwen is served by our own Modal engine, so its rollouts and its
re-encode come from one build: vLLM fp8 weights from the Hugging Face model repository commit `6a9e13bd` (the head since
2026-04-24). Gemma's rollouts come from OpenRouter, and the run's routing admits fp8 endpoints only
(`operations 0013`). Checked on 2026-09-03: DeepInfra, SiliconFlow and Parasail serve
`gemma-4-31B-it` at fp8; every rollout row records the provider that served it. A provider does not
publish its fp8 scheme or the weights commit it serves.

The re-encode engine loads our own weights with block-scaled fp8 (`FineGrainedFP8Config`,
transformers 5.10.2) pinned to the Hugging Face model repository head `842da379` (since 2026-07-20). So a Gemma
re-encode reads the same bit width as the text it re-encodes, but possibly a different quantization
scheme and, in principle, a different weights commit. The launch-readiness plan left the question
open on 2026-08-30 ("whether the Gemma white-box channel is in scope").

The measured cost (`scripts/bench_reencode.py`, 2026-09-03): capture at 2,439 tokens per second and
1,445 tokens per rollout, the logit monitor at 3,834 tokens per second and 5,383 tokens per rollout
for eight positions. A full pass is 7.7 H100-hours with overhead, $30 per model. The Gemma pass is
priced at the Qwen rate until its own smoke.

## Decision

Run the white-box pass on both targets. Gemma re-encodes at fp8 on the pinned head, and the
mismatch is disclosed as a caveat narrowed to the quantization scheme and the weights commit, not
the bit width. Qwen's pass carries no such caveat. The prefull gate runs the white-box smoke on both
models before launch, so Gemma's engine path is proven before the paid pass. Budget line 4 is $61
for both models; the spillover is accepted and the in-run cap rises to $300 (`operations 0021`).

## Alternatives considered

- **Gemma text-only; the white-box axis on Qwen alone.** Lost: the method-divergence read would
  rest on one model, and the study's headline is a two-model replication.
- **Re-encode Gemma at bf16.** Lost: it changes the precision relative to the rollouts and to Qwen,
  and it raises the pass cost; the plan stands as costed.
- **Regenerate Gemma's rollouts on our own Modal engine.** Lost: about $40 of GPU plus cold starts
  for a serving regime `operations 0013` already decided against for Gemma, and the API route at
  fp8 is the same bit width.
- **Do nothing (leave the question open to the gate).** Lost: the budget, the profile and the
  prefull smoke all need the answer before launch.

## Consequences

- The method-divergence claim is made on both models. On Gemma it carries the disclosed caveat: the
  activations come from our fp8 build of the pinned head, the text from a provider's fp8 build.
- `config/full.yaml` lists both models under `whitebox_models` with their revision pins; the
  prefull profile runs the smoke on both.
- Budget line 4 doubles ($30 to $61, `operations 0021`).
- Risk: Gemma's fp8 engine path fails at the gate. The gate then decides between fixing the path and
  dropping Gemma from the pass; the Qwen pass is unaffected.

## References

- `src/cv/whitebox/`, `src/cv/serving/local/app.py` (`ReencodeEngine`), `scripts/bench_reencode.py`,
  `config/full.yaml`, `config/prefull.yaml`.
- `docs/full-run-launch-readiness.md` (wave 2.2, the measurement and the decision of 2026-09-03).
- `design 0018`, `design 0026`, `design 0028`, `operations 0008`, `operations 0013`, `operations 0021`.

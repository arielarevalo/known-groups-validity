# 0032. Under fp8, the re-encode keeps the head and every linear that does not tile in bf16; a non-finite vector is a failed row

- **Status:** Accepted
- **Date proposed:** 2026-09-09
- **Date accepted:** 2026-09-09
- **Relationships:** Amends `design 0031` (scope: the precision of the re-encode, which is fp8 with a
  named bf16 exception, not fp8 throughout). Extends `design 0030` D5 (the failure ledger gains the
  reason `nan`). Relates-to `operations 0008` (the decoupled re-encode pass), `operations 0013` (fp8
  serving parity).

## Context

The re-encode engine loads the target with block-scaled fp8 weights (`FineGrainedFP8Config`,
transformers 5.10.2, blocks of 128 x 128). The quantizer replaces every linear with an fp8 module,
allocates the block scales of each with `torch.empty`, and fills them at load by a tiling of the
bf16 weight into whole blocks. A weight that does not tile in whole blocks keeps an uninitialized
scale, and the container's load report lists that scale as "newly initialized". The forward then
reads uninitialized memory through that module.

Qwen3.6-27B has two such linears in each of its 63 hybrid layers: `linear_attn.in_proj_a` and
`in_proj_b`, both 5120 x 48, which feed the decay gate of the gated delta rule. The prefull
white-box smoke of 2026-09-09 showed the effect. Every captured vector, contrastive and rollout
alike, was finite at layers 3 to 19 and NaN at every layer from 23 on, for every input and every
batch shape. The logit-monitor scores were NaN at every position. The same load had given finite
vectors on 2026-09-03 (`scripts/bench_reencode.py`, zero NaN), which is what uninitialized memory
does: the outcome depends on what the memory held. Gemma 4 has no such linear in its text stack
(hidden 5376, intermediate 21504, both multiples of 128) and stayed finite on both days. Its vision
tower, unused by the text forward, has the same defect (1152 x 4304).

Two further gaps let the fault reach the fit. The driver stored the NaN vectors and scores as valid
rows, and the fit crashed inside scikit-learn on the first NaN it met, after $0.62 of GPU. The
engine's warm-up forward at start checked nothing.

## Decision

Under fp8, the engine keeps three groups of linears in bf16 and reports the list: the head
(`lm_head`, as vLLM serves it), every linear whose weight does not tile in whole 128 x 128 blocks,
and nothing else. The engine finds the misfit linears by a scan of the model skeleton on the meta
device before the load, so the rule is generic over checkpoints (`fp8_exclusions` in
`src/cv/serving/local/app.py`). It reports the list in its `config()` as `fp8_excluded`, in its
start line, and through the preflight into the activation metadata. It refuses a load whose warm-up
forward gives non-finite logits, before any batch is metered.

The driver treats a non-finite vector or score as an engine fault, never as a measurement. Such a
rollout row goes to the failure ledger under the reason `nan` and is never stored. A non-finite
contrastive vector stops the pass, because the eval direction would be NaN. The fit refuses
non-finite inputs with a message that names the count.

For the record, the bf16 exception per target on the pinned revisions: Qwen keeps `in_proj_a` and
`in_proj_b` of the 63 hybrid layers (about 31M of 27B parameters) and the head; Gemma keeps the
head, and the vision tower that the text forward never reads. Everything else runs at fp8.

## Alternatives considered

- **Re-encode at bf16 throughout.** Lost: it changes the precision relative to the rollouts and
  doubles the memory and the cost (`design 0031`).
- **A smaller block size that tiles 48 rows, for every module.** Lost: it changes the quantization
  scheme of the whole model to fit two small projections, and the Triton path is proven at 128 x
  128 only.
- **Pad the misfit weights to a multiple of 128.** Lost: it rewrites the module and its forward, an
  invasive change for a projection that costs nothing at bf16.
- **Do nothing.** Lost: the forward is finite by luck of memory contents. It gave a finite bench on
  2026-09-03 and a NaN pass on 2026-09-09.

## Consequences

- The parity caveat of `design 0031` gains one clause. vLLM served the two decay projections of
  each Qwen hybrid layer at fp8 with per-tensor dynamic scales. The re-encode reads them at bf16.
  The deviation is disclosed with the pass; it touches about 0.1 percent of the parameters.
- The failure ledger has a new reason, `nan`. A row under it is excluded from scoring and retried
  on the next run of `cv-whitebox`, like every other ledgered failure (`design 0030` D5).
- The preflight line names the bf16 list per model, and `whitebox/activations/<model>/meta.json`
  records it under `engine.fp8_excluded`. One backend instance per model serves both the preflight
  and the re-encode step, so that record is filled.
- Verified 2026-09-09 on the `prefull` smoke against `cv-serving-full` v7: both models finite in
  every store and every logit score, zero rows in the ledger, 120 merged rows.

## References

- `src/cv/serving/local/app.py` (`fp8_exclusions`, `ReencodeEngine.load`), `src/cv/whitebox/reencode.py`
  (`_finite`), `src/cv/whitebox/fit.py`, `src/cv/serving/local/backend.py` (`get_reencode_backend`),
  `tests/unit/cv/test_whitebox.py`.
- `docs/full-run-launch-readiness.md` (wave 5.1, the smoke of 2026-09-09).
- `design 0030`, `design 0031`, `operations 0008`, `operations 0013`.

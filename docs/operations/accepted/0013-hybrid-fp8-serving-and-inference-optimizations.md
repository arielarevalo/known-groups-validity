# 0013. Hybrid serving (Gemma on API-FP8, Qwen/Mistral self-hosted FP8) and the inference-optimization stack

- **Status:** Accepted
- **Date proposed:** 2026-06-12
- **Date accepted:** 2026-06-12
- **Supersedes:** Amends `operations 0007` (which self-hosted ALL targets uniformly and stated
  "there is no inference-API repoint for targets"); builds on `operations 0002` (cost model),
  `0003`/`0009` (local orchestration + checkpointing, reaffirmed), `0008` (activation re-encode,
  precision consequence noted), `design 0014` (reps), `design 0021`/`0022` (judge + protocol).

## Context

`operations 0007` chose to self-host all three targets on Modal on the premise that self-hosting
saves money over per-token APIs. The 2026-06-12 subset run measured that premise for the first time
and it does not hold uniformly:

- **Measured throughput:** ~750 tok/s aggregate output for Gemma-4-31B (bf16) on a single H200,
  dragged down because Gemma-4's heterogeneous head dimensions (head_dim 256 / global 512) force
  vLLM's slower **Triton attention backend** (logged at deploy). Cost per output token is
  `(GPU $/hr) / (tok/s x 3600)` ~= $1.70/M at that rate.
- **The economics are model-dependent.** Against API output prices, self-hosting *loses* for a
  cheap-output model (Gemma API output ~$0.37/M, so self-host is ~4.5x the API cost) and *wins*
  for an expensive-output model (Qwen API output ~$3.20/M; Qwen also uses standard attention, so it
  avoids the Triton penalty and runs faster self-hosted).
- **GPU cost scales with wall-clock, not utilization.** Parallelizing across more containers cuts
  wall-clock but not total GPU-seconds, so it does not fix the cost; raising **tokens per
  GPU-second** does (quantization, tuning, a faster attention path, a right-sized GPU).
- **Precision parity.** Upstream did not set a precision standard: the factor-controlled benchmark
  ran entirely on **APIs at provider precision** (`create_target_client` wires only
  OpenRouter/OpenAI/Anthropic/DeepSeek; the repo's 4-bit `load_hf_model` is unused). So
  precision is a choice we own and justify ourselves, not a standard to match.

## Decision

### 1. Per-model serving is a hybrid, chosen by where the economics fall, all served at FP8

| Target | Black-box (rollouts + CoT/SR/TP) | White-box probe (experiment) |
|---|---|---|
| **gemma-4-31b** | **OpenRouter API, provider filtered to FP8** (DeepInfra is an FP8 provider) | HF self-host (activations require it; serve FP8 to match, see Risks) |
| **qwen3.6-27b** | **Self-host on Modal, FP8** | HF self-host, FP8 |
| **mistral-small-2603** | **Self-host on Modal, FP8** | HF self-host, FP8 |
| judge `xiaomi/mimo-v2.5-pro` | OpenRouter (unchanged, `design 0021`) | n/a |

The hybrid is **not a precision compromise**: every target is served at **FP8**, by API for the
cheap-output model and self-hosted for the expensive-output ones. "All targets FP8" is a cleaner
parity story than mixed precision. Gemma's black-box work leaves the GPU entirely (it is cheaper to
buy than to serve slowly); Qwen and Mistral self-host, where FP8 + standard attention make
self-hosting genuinely cheaper than their APIs.

### 2. Quantization source policy: official, else runtime-dynamic, never offline-calibrated

1. **Prefer an official FP8 checkpoint** (vendor, or a reputable quantizer such as
   RedHat/neuralmagic `-FP8`/`-FP8-dynamic`): calibrated, validated, documented, citeable.
2. **Else vLLM runtime dynamic FP8** (`quantization="fp8"`): a load-time cast of a bf16 checkpoint,
   no calibration data, no stored artifact. Acceptable; disclose it is our automated quantization.
3. **Never roll our own offline-calibrated (static) FP8.** A calibration-data choice and a
   divergence surface with no payoff over (1)/(2); the simplicity principle forbids it.

Per-model FP8 availability is unverified (the Qwen/Mistral ids are unconfirmed; the Mistral id returns 404),
so "check for an official FP8 checkpoint" is a per-model action item.

### 3. The inference-optimization stack (implement what is sensible; verify on GPU before trusting)

**GPU-time levers (lower $/token = raise tokens per GPU-second):**
- FP8 weights (above): the dominant lever, ~halves the bytes read per decode token.
- `kv_cache_dtype="fp8"`: halves KV-cache memory, so more sequences run concurrently per batch.
- `gpu_memory_utilization` ~0.95 and a right-sized `max_num_seqs`: widen the batch to amortize
  weight reads over more sequences.
- `max_model_len` capped to the real need (16384, already set): frees KV cache; also lets a 31B
  fit a **cheaper H100** rather than an H200 (H200 only if memory genuinely requires it).
- `enable_prefix_caching` (already on); published per-model vLLM recipes
  (`recipes.vllm.ai/Google/Gemma4`, HF model cards) baked in verbatim.

**Wall-clock levers (cost-neutral; pursue once a warm image exists):**
- **Warm image:** persist vLLM's `torch.compile` cache (and weights, already in a Volume) so a cold
  container is ready in seconds instead of recompiling (~24s seen) every cold start; pin an
  optimized vLLM build.
- **Bounded fan-out:** submit chunks concurrently (`.map`/concurrent `.remote`) with a
  `max_containers` cap so Modal parallelizes across containers. Wall-clock scales ~linearly at
  ~constant GPU-seconds; only worth it after the warm image, or the cold-start tax is paid N times.
- CoT (judge) runs last so GPU stages stay contiguous (already, `cv/pipeline._EXEC_ORDER`).

### 4. Orchestration stays local (reaffirms `operations 0003`/`0009`)

Per the 2026-06-12 discussion: the driver/orchestration runs on the laptop; only the GPU function
runs in Modal; results stream back and are written locally; W&B Artifacts are the durable ledger
that makes the ephemeral laptop resumable. The cloud-resident-orchestration alternative (run the
whole loop in a Modal function writing to a Volume) was considered and **declined** for now: local
orchestration is simpler to drive and debug, and checkpointing already covers interruption. This is
why checkpointing exists, and it is retained deliberately.

## Alternatives considered

- **Keep self-hosting all targets (`operations 0007` as written).** Lost: measured throughput makes
  Gemma-on-GPU several times more expensive than its API, and the Triton penalty is Gemma-specific.
- **Put everything on the API.** Lost: forgoes the genuine self-host savings on the expensive-output
  models, and the white-box probe needs activations the API cannot return.
- **bf16 self-hosting (status quo of the subset run).** Lost on cost: doubles the bytes read per
  token versus FP8 for no parity benefit (upstream set no precision standard).
- **Offline-calibrated home-rolled FP8.** Lost: calibration-choice and validation burden with no
  gain over official / runtime-dynamic.
- **Cloud-resident orchestration.** Declined for now (see Decision 4); revisit if unattended runs
  outgrow laptop-driven orchestration.

## Consequences

### Enables
- Self-hosting is paid for only where it wins (Qwen, Mistral); Gemma is bought cheaply per-token.
- A uniform-FP8 serving story across all targets, by two routes.
- A concrete, costed path off the 750 tok/s floor before any grant commitment.

### Forbids / changes
- **Per-model serving backend routing is now required** (was a single global `target_backend`).
- **OpenRouter requests for Gemma must constrain the provider to FP8** (a `quantizations: ["fp8"]`
  filter, optionally preferring DeepInfra), or OpenRouter may route to a bf16 provider and silently
  break the parity. Must coexist with the existing content-moderation provider fallback (filter,
  do not hard-pin to one provider).
- The serving-optimization code is **wired but unverified**: it must be GPU-smoked before trusting,
  per the three deploy-only bugs found on 2026-06-12 (nvcc, KV-cache OOM, gemma `enable_thinking`).
  Defaults are set to current behavior so an un-tuned deploy is unchanged from the known-green run.

### Risks
- **Probe precision consistency (`operations 0008`):** the activation re-encode must use the same
  build + dtype as the rollout. Gemma's rollouts will be FP8 on DeepInfra but the re-encode is HF
  self-host; serving HF Gemma at FP8 narrows but may not eliminate the gap (DeepInfra's exact FP8
  recipe is opaque). Disclose, and prefer same-recipe FP8 where possible; a small residual mismatch
  is tolerable (probes are robust to FP nondeterminism) but must be stated.
- **FP8 awareness-signal validity is unvalidated.** Quantization can move a subtle behavioral rate
  even when benchmark accuracy holds. Optional (the authors' call) **bf16-vs-FP8 awareness-rate spot-check
  on a sample** before trusting FP8 at scale. For convergent validity specifically the risk is
  bounded: all detectors read the same served model, so internal agreement is self-consistent at
  any precision; precision only bites on cross-precision absolute-rate comparisons.
- **The 2026-06-12 subset Gemma data is bf16-Modal**, a different stack and precision from the
  experiment's FP8 Gemma; not pooled (already forbidden, `operations 0002`/`0015`); disclose.
- **Official FP8 availability is unverified per model**; if none exists and runtime-dynamic is
  unsatisfactory, that model's serving is reopened.

## Action items
- Verify an official FP8 checkpoint exists for Qwen and Mistral; fix the Mistral repo id (404).
- Implement per-model backend routing, the OpenRouter FP8 provider filter, and the vLLM tuning/quant
  knobs (this commit lands the wiring, defaulted to current behavior; **GPU smoke pending**).
- **Required gate before any Qwen self-host run (the authors decided, 2026-06-12):** on the **second Modal key**
  (`MODAL_TOKEN_ID_2`, ~$30 free credit), smoke Qwen's FP8 deploy and **measure sustained tok/s**,
  then compute whether the full Qwen subset (5,400 rollouts x 5 GPU generations = rollout + 2 SR + 2
  TP; CoT is the OpenRouter judge, not Modal) fits the $30. The smoke does double duty: it also
  verifies the unverified FP8 + tuning deploy. Breakeven (decode-bound, ~23M output tokens for the
  Qwen subset): the subset fits $30 only if Qwen sustains **>= ~850 tok/s on H100 ($3.95/hr) or
  >= ~960 tok/s on H200 ($4.50/hr)**, before cold-start overhead. Do not start the full Qwen run
  until this check passes; if it fails, descope (fewer reps/tasks) or reopen the Qwen serving choice.
- Build the warm image (persist the `torch.compile` cache to the Volume) before any fan-out.
- Optional: FP8-vs-bf16 awareness-rate spot-check.

## References
- `operations 0007` (amended), `0002` (cost model), `0003`/`0009` (local orchestration + ledger),
  `0008` (activation re-encode precision), `0010` (tier ordering).
- `design 0021`/`0022` (judge + upstream protocol), `0014` (reps).
- Upstream `utils.create_target_client` (API-only benchmark), `core/load_model.py` (unused 4-bit).
- 2026-06-12 subset run: measured ~750 tok/s (Gemma bf16, H200, Triton attention); see
  `results/runs/subset/`.

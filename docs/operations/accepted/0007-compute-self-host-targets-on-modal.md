# 0007. Compute: self-host targets on Modal GPUs; judge on an API; reps=3

- **Status:** Accepted: Modal smoked green 2026-06-04. Supersedes `operations/accepted/0006`.
  **Amended by `operations 0020` (2026-07-04):** the free-credit assumption below did not hold
  (credits attach to the workspace, not the person; none applied) and the pilot billed $158 to a
  card; see the post-mortem for the corrected accounting and estimation rules.
  **Amended by `operations 0013` (2026-06-12):** the "self-host serves all targets uniformly / no
  inference-API repoint for targets" stance is revised to a hybrid on measured-throughput economics:
  Gemma's black-box work moves to OpenRouter (FP8 providers), Qwen/Mistral self-host at FP8; the
  Modal serving spine, judge-on-API, and reps=3 below remain in force. White-box probe work still
  self-hosts (activations).
- **Date proposed:** 2026-06-04
- **Date accepted:** 2026-06-04
- **Supersedes:** `operations/accepted/0006` (W&B-native two-tier), invalidated: the $250
  W&B Inference and $500 W&B Training credits were both withdrawn, W&B Training is
  serverless RL/SFT only (verified: it cannot run custom inference / activation code), and
  W&B Inference does not expose gemma's reasoning trace (CoT void). Also supersedes the
  OpenRouter target-serving assumption in `operations/accepted/0002`.

## Context

Smoke tests invalidated the W&B-native plan (the lesson: verify, don't assume):
- W&B Inference returned 429 errors until we found it needs an `openai-project` header; then the
  $250 credit was withdrawn; and it returns **no reasoning trace for gemma** (CoT void), only for
  thinking models (qwen).
- W&B Training is **serverless RL/SFT only** (docs-confirmed) and **cannot** run our
  workload (self-host a causal LM → forward-hook hidden-state activations → linear probes);
  its $500 credit was also withdrawn.

We need GPU that runs **arbitrary code** (full reasoning traces for CoT + hidden-state
activations for probes), with good DX and a usable free/cheap tier. **Modal** fits and was
**smoked green 2026-06-04**: a `modal run` GPU job (code stays in-repo) loaded an HF causal
LM, generated, and captured activations via **both** `output_hidden_states` and a mid-layer
`register_forward_hook` (29 layers, `[1, 127, 1536]` on Qwen2.5-1.5B/T4), the probe path.
Free tier is **$30/mo per person (×2 = $60/mo)**; pay-per-second beyond. With reps=3
(`design/accepted/0014`) the subset is ~4–9 GPU-hr (fits the free credits); the experiment
is bounded and grant-funded.

## Decision

- **Targets run self-hosted on Modal GPUs for both phases:**
  - **Subset** (black-box): vLLM batched serving, with full reasoning traces, so **gemma CoT
    works** (no W&B-Inference reasoning gap).
  - **Experiment** (white-box): HF + forward hooks → hidden-state activations → linear probes.
- **Self-host serves all targets uniformly** (gemma, qwen, mistral), with full reasoning +
  activations for each; no per-provider model-availability gaps.
- **The judge stays on an inference API** (no activations needed, so no reason to
  self-host): **`xiaomi/mimo-v2.5-pro` on OpenRouter**: the original committed-pilot
  judge, live on OpenRouter and already priced in `cv/config.py`. The earlier "switch
  to Kimi-K2.6" idea existed *only* because mimo was absent from the W&B Inference
  catalogue; now that the judge runs on OpenRouter that constraint is moot, so mimo is
  retained (W&B Inference is not returning for the judge either).
- **reps = 3** throughout (`design/accepted/0014`). Subset on the free Modal credits;
  experiment grant-funded (bounded GPU-hours).
- **Weave (traces) + Artifacts (results transport)** remain the observability/transport
  layer (`operations/accepted/0001`, smoked green; those W&B surfaces are free-tier).

## Alternatives considered

- **W&B-native (0006).** Rejected: credits withdrawn; Training is RL/SFT-only.
- **W&B Inference for targets.** No reasoning for gemma (CoT void); credit withdrawn.
- **Lightning AI.** Runs-anything + free credits, but Studio orchestration friction and an
  ~83-min H100 provisioning queue; Modal's code-local `modal run` DX is better.
- **RunPod / Vast.** Cheaper spot GPU, but more manual (SSH/rsync) than Modal's DX.

## Consequences

### Enables
- One serving mechanism for the whole study: full reasoning (CoT for all targets incl gemma)
  + activations (probes).
- Subset within the free credits; bounded grant-funded experiment.

### Forbids / changes
- `serving/local` (a Modal app: vLLM for the subset, HF + hooks for the experiment) is the
  target backend; there is **no inference-API repoint for targets**. The only API is the judge.
- The OpenRouter target cost model (`operations/accepted/0002`) is superseded for targets
  (now Modal GPU-hours); judge cost is the remaining API line item.

### Risks
- **30B throughput on Modal H100 not yet benchmarked** (the smoke was 1.5B/T4); a real vLLM
  30B benchmark will replace the ~1.5–3k tok/s estimate behind the subset GPU-hours.
- **Judge** = `xiaomi/mimo-v2.5-pro` on OpenRouter (resolved; the original pilot judge).
- **Activation extraction** (HF + hooks, no vLLM batching) is slower than generation; the
  experiment's GPU-hours need their own benchmark.

## References

- `operations/accepted/0006` (superseded), `0001` (Weave/Artifacts, retained), `0002`
  (cost model, superseded for targets), `0004` (two-phase execution).
- `design/accepted/0014` (reps=3), `0015` (probe channel), `0016` (construct).
- Modal smoke (2026-06-04): GPU job + generate + `output_hidden_states` + `register_forward_hook`
  activation capture, all green.

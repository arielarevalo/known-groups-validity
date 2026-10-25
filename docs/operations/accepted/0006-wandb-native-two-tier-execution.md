# 0006. W&B-native two-tier execution: subset run on Inference, experiment on Training (with probes)

- **Status:** Superseded by `operations/accepted/0007`: the W&B Inference ($250) and
  Training ($500) credits were both withdrawn and W&B Training is RL/SFT-only, so the
  *provider* moves to self-hosting targets on Modal. The construct/probe/two-phase intent
  carries forward into 0007.
- **Date proposed:** 2026-06-04
- **Date accepted:** 2026-06-04
- **Supersedes:** the "API pilot → GPU full run" framing of `operations/accepted/0004` and
  the OpenRouter cost model of `operations/accepted/0002`; realizes
  `design/accepted/0015` (probe channel) and amends `design/accepted/0006`
  (api-only / defer-probes)

## Context

The W&B academic plan removes cost as a real constraint for our volumes (ADR 0001): 200 GB
storage, 25 GB/mo Weave ingestion, **$250 W&B Inference** (pay-as-you-go now active), and
**$500 W&B Training** (GPU). We want a **single provider** (no OpenRouter + W&B mixing).

W&B Inference is OpenAI-compatible; the live catalog (verified 2026-06-04) includes
`google/gemma-4-31B-it` and `Qwen/Qwen3.6-27B`, but **not** `mistralai/mistral-small-2603` or
the `xiaomi/mimo-v2.5-pro` judge.

The end-state (conference paper) requires **linear probes on target activations**, which needs
white-box access that only self-hosting provides. Self-hosting the open-weight targets with
forward hooks yields, in one pass, both the response (for CoT/SR/TP) and the per-layer
activations (for probes). So probes are not a separate system bolted on; the experiment's
rollout *is* the activation capture.

"pilot" is reserved for the gemma-only, ~$0 dev config; it is **not** the experiment.

## Decision

Two real runs, both entirely within W&B (no mixed providers):

1. **Subset run**: `gemma-4-31b` + `qwen3.6-27b` on **W&B Inference** (OpenAI-compatible,
   `WANDB_API_KEY`, base `api.inference.wandb.ai/v1`, ~$250 token budget). Black-box detectors
   (CoT/SR/TP). Judge = **`moonshotai/Kimi-K2.6`** on W&B Inference (replaces mimo; comparable).
   Cheap, standalone, written up as a preliminary result. `mistral-small-2603` is omitted here
   (absent from the W&B Inference catalog).

2. **The experiment**: all three targets (`gemma`, `qwen`, `mistral`; mistral self-hostable
   from HF) + all conditions + **linear probes**, **self-hosted on W&B Training GPUs** ($500
   credit). Each hooked forward pass yields response **and** activations, so the gemma+qwen
   subset is re-run here white-box. Judge stays `Kimi-K2.6` on W&B Inference. This is the
   conference-paper run.

**Economic rationale:** $500 of GPU time buys far more total inference than $250 of tokens, so
the full experiment lives on W&B Training; Inference is reserved for the cheap subset.

## Alternatives considered

- **OpenRouter (status quo, 0002/0004).** Rejected: mixes providers, no activation access, pays
  per token.
- **W&B Inference for the experiment too.** Rejected: an API cannot return activations; probes
  require self-host.
- **Keep the mimo judge via OpenRouter.** Rejected: mixes providers; Kimi-K2.6 on W&B Inference
  is comparable and keeps one provider.
- **Self-host the judge too.** Rejected: Kimi is a large MoE, impractical to self-host; the judge
  needs no activations, so W&B Inference is fine and still single-provider.

## Consequences

### Enables
- A single-provider (W&B) pipeline; activations + responses in one pass; probes as phase B.
- A cheap standalone subset writeup plus a white-box conference experiment.

### Forbids / changes
- Supersedes 0004's "API pilot / GPU full run" naming and 0002's OpenRouter cost model. Amends
  `design/accepted/0006` (probes are in scope, phase B). The cv client gains a **W&B Inference**
  backend (subset) and a **self-hosted HF + forward-hooks** backend (experiment); the restructure
  adds `serving/{api,local}` and a `probe/` subpackage.
- `mistral-small-2603` runs only in the experiment (self-host), not the subset.

### Risks
- **W&B Inference requires the `openai-project` (team/project) header, not a concurrency cap.**
  Earlier 429s ("concurrency limit reached") were a *misleading* error for the missing project
  context, not a real quota (we'd spent zero). Verified live 2026-06-04: with
  `extra_headers={"openai-project": "<entity>/<project>"}` (entity
  `bluedot-tais-project-convergent-validity`), 5/5 concurrent gemma calls succeed (~1s each). The
  repoint is therefore **not** concurrency-blocked; the client must set this header on every call.
- W&B Training pricing/quotas are opaque; treat as a managed risk, not a blocker. Mitigation:
  carry over a meter/cap discipline; monitor GPU-hours.
- Self-hosting decode throughput (vLLM/HF) is unmeasured; a short GPU probe before the experiment
  tightens estimates (carried from 0002).

## References

- `docs/operations/accepted/0001` (W&B Weave + Artifacts storage), `0002` (cost model, superseded
  here), `0004` (two-phase, reframed here), `0003` (run namespacing).
- `docs/design/accepted/0006` (api-only/defer-probes, amended), `0015` (probe channel, realized),
  `0016` (construct), `design 0017` (controls).
- W&B Inference catalog verified live 2026-06-04: `gemma-4-31B-it`, `Qwen3.6-27B`, `Kimi-K2.6`
  present; `mistral-small-2603`, `mimo` absent.

# 0006. API-only target inference via OpenRouter; defer activation probes

- **Status:** Accepted (amended by `docs/design/accepted/0015` for the funded full-run phase: self-hosted targets plus a white-box probe channel; remains in force for the Phase A API behavioral pilot, `docs/operations/accepted/0004`)
- **Date proposed:** 2026-05-10
- **Date accepted:** 2026-05-13
- **Supersedes:** n/a

## Context

The v0 proposal led with **Gemma-3-27B-it** as the single primary model, run
locally on a Vast.ai A100 80GB, with three detector channels including a
white-box **activation probe** built on Gemma Scope 2 SAEs and a raw-activation
linear-probe robustness variant. That gave the probe channel a load-bearing
role in the comparison.

Two facts reshaped this:

1. **The convergent-validity question can be answered without the probe.** The
   three behavioural detectors (CoT, SR, and the EvalAwareBench TP/PG variants)
   already span the manipulability axis the v0 proposal cared about, share a
   common Type-1 anchor (DDR 0002), and admit cross-detector kappa (DDR 0005).
   The probe was the strongest single channel epistemically but not necessary
   for the headline result.
2. **API access is structurally cheaper and faster for a four-detector
   comparison across three target models.** OpenRouter exposes
   `google/gemma-4-31b-it`, `mistralai/mistral-small-2603`, and
   `qwen/qwen3.6-27b` at metered per-token prices that put the full run at
   roughly $700-2,800 (10 reps) and the pilot at under $5. A local-GPU
   three-model version would require either renting three GPU classes or
   serialising the three targets, both of which dominate the saved API spend.

Activation probes need open weights plus forward hooks, which OpenRouter does
not expose. That is a different access regime, not a fifth API detector.

## Decision

For v1, target inference is **API-only via OpenRouter** for all three target
models. No local-GPU rollout step. Activation probes are explicitly deferred
to a later white-box phase that uses open weights plus forward hooks
(RunPod/HF + TransformerLens or nnsight); they are out of scope for v1 and
recorded as a known fork.

The four v1 detectors are all behavioural: CoT, SR, TP, PG (DDR 0007 covers
judge validation for the two judge-mediated ones).

The judge is also routed through OpenRouter (`OPENAI_BASE_URL` env routing;
no code edit to upstream) so the whole pipeline is one API regime.

## Alternatives considered

- **Keep Gemma-3-27B-it locally as in v0, single model, probe in.** Rejected:
  loses cross-model generalisation; locks the comparison to one model's
  idiosyncrasies; the probe's distinctive contribution would be hard to
  separate from model-specific quirks at n=1.
- **Two regimes side-by-side: API behavioural detectors + local probe on
  Gemma-3-27B.** Rejected for v1: two access regimes, two infrastructures,
  two cost models, two failure modes. Cleaner to ship the API-only v1, then
  add the probe as a labelled extension once v1 is done.
- **Use API logprobs as a proxy for the probe channel.** Rejected: logprobs
  on OpenRouter are spotty (gemma yes, mimo/qwen/mistral no per Step-0
  validation), and a logprob-derived score is not the same construct as a
  trained linear probe.

## Consequences

### Enables
- Three target models in the comparison, not one. Cross-family
  generalisation in v1, not just a mid-tier extension.
- One access regime, one budget model, one set of failure modes.
- A pilot under $5 and a full run that fits a small-grant budget rather than
  a GPU-rental budget.

### Forbids
- Any v1 probe-channel claim. The four detectors are all behavioural.
- Any claim that requires access to model internals (residual-stream
  activations, attention patterns, SAE features).

### Risks
- The cross-detector comparison loses its hardest-to-evade channel
  (activation probe). The convergent-validity surface then characterises
  the agreement structure of *behavioural* detectors only, which is a
  narrower claim than "behavioural detectors agree with mechanistic
  detectors". The writeup must frame the contribution accordingly.
- API providers can change pricing, deprecate models, or alter sampling
  defaults between pilot and full run. Mitigation: the budget guard
  (`CV_BUDGET_USD`) and the meter (`cv/meter.py`) bound the downside;
  Step-0 validation (`cv/validate.py`) is re-runnable per-model.

## References

- `PLAN.md` lines 47-59 (model and judge selection), 167-173 (probe deferral),
  225-240 (judge routing).
- `cv/bootstrap.py` (model registration into upstream's OpenRouter config dict).
- `cv/run_*.py` (the four behavioural runners).
- DDR 0007 (judge-vs-judge validation, required because two of the four
  detectors are judge-mediated).

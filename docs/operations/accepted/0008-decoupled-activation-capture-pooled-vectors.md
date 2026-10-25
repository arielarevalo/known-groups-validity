# 0008. Store rollouts as text; capture activations in a decoupled re-encode pass, as pooled per-layer vectors

- **Status:** Accepted
- **Date proposed:** 2026-06-05
- **Date accepted:** 2026-06-05
- **Supersedes:** realizes the storage/serving side of `design/accepted/0015` (white-box probe
  channel); builds on `operations/accepted/0007` (Modal serving), `0001` (Weave/Artifacts), `0003`
  (per-run namespacing)

## Context

`design/accepted/0015` adds a white-box activation-probe channel for the funded full run but left
the read-out mechanics open. `operations/accepted/0007` settled that targets self-host on Modal
with two engines: a vLLM engine (batched generation, decode-bound) and an HF engine (forward
hooks, the only path to hidden states). The open question this record closes: what do we persist
from a rollout, and how do we obtain activations without exploding storage or GPU cost.

Two facts force the design. (1) A linear probe or difference-in-means direction consumes one
fixed-length vector per `(example, layer)`: a pooled or single-token activation, not the full
per-token tensor. Storing per-token activations for every rollout is terabytes per model
(`n × layers × seq_len × d_model × bytes`); storing one pooled vector per layer is a few GB (the
`seq_len` axis collapses). (2) Activations are a deterministic function of `(model build, prompt,
response)`, so they can be recomputed after the fact by a forward pass over saved text. HF is slow
only at autoregressive *decode* (the token-by-token loop), not at a single *prefill* forward; a
teacher-forced re-encode is one prefill pass, hence cheap.

## Decision

Persist rollouts as **text only** (status quo, reaffirmed): `responses_<model>.json` carrying ids,
config, prompt, response. No activations are captured during the rollout, and generation runs on
the vLLM engine (`operations/0007`), the cost-optimal decoder.

Capture activations as a **separate, derived phase-B pass** that re-encodes saved text on the HF
engine: feed `(prompt [+ response])` through the model in one teacher-forced forward, hook every
layer, and persist a **pooled vector per `(layer, position-scheme)`**. This puts the expensive
decode on vLLM and leaves HF only a cheap prefill, which at our volume is cheaper than one-pass HF
generation-with-hooks (decode on the slow engine). Position schemes: the **last prompt token**
(primary; deduplicated across the `reps=3` replicas since the prompt is identical, so the captured
set is `~tasks × configs`, not `× reps`) and the **response mean-pool** (secondary). Sweep all
layers; store fp16.

Persist activation artifacts under the run namespace (`results/runs/<run_id>/activations/`,
`operations/0003`), keyed by sample id with the factor ground-truth label attached
(`src/cv/factors.py`), in a columnar/array format (npz or safetensors per model+layer, or
parquet), shipped through W&B Artifacts (`operations/0001`). Keep
`HfEngine.generate_with_activations` as a fallback for the rare case that streaming-exact
(per-generated-token) activations are wanted; the decoupled re-encode is the default.

## Alternatives considered

- **One-pass HF generation-with-hooks** (generate and capture in a single run). Lost: puts the
  decode loop on the slow engine; at thousands of long 30B generations the GPU-hours dominate and
  dwarf a second cheap prefill pass.
- **Capture during vLLM generation.** Lost: vLLM hides the forward pass behind fused kernels and
  exposes no arbitrary-layer hooks; not available.
- **Store full per-token activation tensors.** Lost: terabytes per model for no probe benefit; the
  probe consumes pooled vectors.
- **Store nothing extra, recompute on demand each probe iteration.** Lost: re-running the
  re-encode for every layer/pooling experiment wastes GPU; persist once, iterate offline.

## Consequences

### Enables
- Rollouts can start now, text-only, with no activation burden; the black-box subset is unaffected
  (it never needs activations).
- Probe-method iteration (layer, pooling, DIM vs logistic) happens offline on a few GB of stored
  vectors, with no regeneration.
- The vLLM/HF engine split (`operations/0007`) is justified by exactly this division of labor.

### Forbids / constrains
- The re-encode must use the same model build + dtype as the rollout so the activations correspond
  to the saved text; pin the HF revision. FP nondeterminism is tolerable (probes are robust to it).
- No per-token activation store; analyses needing per-token traces are out of scope under this
  record.

### Risks
- Throughput for both vLLM generation and the HF re-encode on a real 30B is unmeasured
  (`operations/0007` risk); the planned 30B benchmark must measure both to confirm the cost
  ordering this record assumes.
- Last-prompt-token pooling assumes the eval signal is present at the situation-assessment point;
  if the signal is response-borne, the secondary response-pool capture is the fallback (extra
  storage, still GBs).

## References
- `operations/accepted/0007` (Modal serving: vLLM + HF engines), `0001` (Weave/Artifacts), `0003`
  (per-run namespacing).
- `design/accepted/0015` (white-box probe channel, whose read-out storage this realizes), `0014`
  (reps=3, the dedup factor), and companion `design/accepted/0018` (the probe method this feeds).
- `src/cv/serving/local/app.py` (`HfEngine.generate_with_activations`, the capture primitive),
  `src/cv/factors.py` (ground-truth labels).

# 0019. Full-run budget: measured cost basis and funding

- **Status:** Accepted. **Amended by `operations 0020` (2026-07-04):** the pilot-spend figures in
  §5 are corrected to the invoice totals ($158 billed, no free-credit coverage). **Amended by
  `operations 0021` (2026-09-03):** §4 line 2 is $34 (the fp8 route), line 4 is $61 (both models,
  measured), the worst case is $274 and the expected $263; §5's pilot figure is $188.38 across two
  workspaces; §6's in-run cap is $300 and the dashboard cap is set per workspace, on both.
- **Date proposed:** 2026-06-23
- **Date accepted:** 2026-06-23
- **Supersedes:** finalizes the cost model of `operations/accepted/0002` (its all-API estimate
  predates the measured token and throughput data below) and closes the budget cluster (themes
  **F1, F2, G, H**) of `docs/full-run-open-decisions.md`. Builds on `operations 0013` (hybrid FP8
  serving + rates), `0008` (decoupled re-encode), `design 0026` (two models, scope), `0028` (logit
  monitor), `0014` (reps=3).

## Context

Every design decision for the full run is closed (`design 0026`/`0027`/`0028`). What remained was a
**costed budget**: a full-run GPU-hours × Modal-rate estimate for a grant request, plus the
throughput and token measurements behind it. This record prices every line from a
measured quantity or a published rate, applies **no blanket contingency**, and shows the residual
variance as an explicit sensitivity rather than a cushion.

Every input is measured or published, with provenance:

- **Scope** from the primary source (EvalAwareBench, arXiv:2605.23055, §5.1 and Fig 7/30).
- **Per-rollout token counts** from the pilot's saved rollouts and detector outputs, tokenized with
  each model's own tokenizer (`Qwen/Qwen3.6-27B`, `google/gemma-4-31B-it`).
- **CoT-judge cost** from the pilot meter ledger (`data/results/runs/pilot/_metrics.jsonl`, 10,800
  judge calls), validated to the cent.
- **Qwen FP8 decode + prefill throughput** measured on a live Modal H100 at the exact `full.yaml`
  serving config (`scripts/bench_throughput.py`, 2026-06-23). See §3.
- **Rates**: Modal H100 $3.95/hr, H200 $4.54/hr (on-demand per-second, re-verified at
  modal.com/pricing 2026-06-23); OpenRouter prices (`cv/config.PRICES_PER_M`, verified live).

## Decision

### 1. Scope: a faithful two-experiment reproduction at reps=3 (25,200 rollouts)

A faithful reproduction runs the **two** experiments the EvalAwareBench paper runs, not the full combinatorial cube:

- **Experiment A, single-factor variation** (paper Fig 6): `baseline + F1..F8` (9 configs) on all
  **200** task-valence items (100 pairs × 2 valences) × **reps 3** = **5,400 rollouts/model**. This
  is the pilot scope. **F8 stands alone here**: the explicit "this is an evaluation" disclosure is a
  trivial, partly-circular cue, kept only as a single-factor positive control.
- **Experiment B, factor stacking** (paper Fig 7/30): on a **25-task-per-valence subset** (50 items).
  The paper draws `min(10, C(8,k))` combinations per item over all eight factors at each active-factor
  count `k`. **We exclude F8 from the stacking** (stacking the explicit "this is an evaluation" giveaway
  saturates the dose-response curve), so our combinatorics are over **F1–F7**: `min(10, C(7,k))` for
  `k=1..7`, `C(7,k)=7,21,35,35,21,7,1` → `7,10,10,10,10,7,1` = 55 configs/item. Subtracting the 7 single
  factors reused from A leaves **48 new stacked configs/item** × 50 items × reps 3 =
  **7,200 rollouts/model**. (Excluding F8 and using reps=3 vs the paper's 10 are our deviations,
  disclosed below.)

Per model: 5,400 + 7,200 = **12,600 rollouts**. Two models (gemma, qwen): **25,200 rollouts**.

**Emergent scope decisions (surfaced):** (a) the released dataset is the full `2^8 = 256`-config cube
× 200 items = **51,200 prompts** (×3 reps = 153,600 rollouts/model); running the whole cube is **not**
what the paper does and would cost ~10× more, so it is out of scope; we reproduce the paper's *subset*
stacking design. (b) **F8 stands alone** (single-factor control only, not stacked), so the stacking is
F1–F7. (c) reps=3 where the paper used 10 (`design 0014`): leaner, not richer, disclosed.

### 2. Per-rollout cost basis (measured)

Five generations per rollout (`rollout + 2 SR + 2 TP`), confirmed in `cv/run_sr.py`/`run_tp.py` and
the saved detector outputs (each of SR/TP emits two reasoning traces). Mean tokens, measured:

| Stage | gemma (OpenRouter) | qwen (Modal GPU) |
|---|---|---|
| rollout out | 1,372 | 2,664 |
| SR out (2 gens) | 1,625 | 2,750 |
| TP out (2 gens) | 1,446 | 2,560 |
| **total output / rollout** | **4,443** | **7,974** |
| input / rollout | ~7,348 (API-billed) | ~3,762 unique (prefix-cached prefill) |
| CoT judge (mimo) | 3,615 in / 1,989 out → **$0.003303/rollout** (validated: $35.67 / 10,800) | same |

**Finding (surfaced):** the detectors, not the rollout, dominate GPU decode. The four detector
follow-ups each emit a full reasoning trace, so they are 67% of the 7,974 Qwen output tokens/rollout
(~1.87× the ~4,259 `operations 0013` assumed). Identified lever, not taken: running the SR/TP
follow-ups with thinking disabled would roughly halve the decode load, but the model's reasoning in
the follow-up is part of what those detectors read, so it stays.

**TP input change (surfaced; `design 0029`).** For the full run, the Third-Person detector reads the
visible answer only: `format_conversation_for_thirdperson` now strips the `<think>…</think>` trace
(`extract_think_tags`). This is a **deliberate deviation** from upstream, not a fidelity fix: upstream
feeds TP the full response including the trace (verified in upstream's code, `design 0022`), and our
original port matched it. We strip it so TP does not re-read the same reasoning the CoT judge reads,
keeping the two detectors evidence-independent. It removes ~683 (gemma) / ~1,806 (qwen) input tokens
per TP call; cost effect ~$2 (Gemma API input). SR keeps the full response.

### 3. Throughput: measured live on H100 (FP8)

The dominant GPU line is decode-bound: `output_tokens / decode_tok_per_s × GPU_$/hr`. Measured on a
live Modal H100 (`Qwen/Qwen3.6-27B`, FP8, `kv_cache_dtype=fp8`, `gpu_memory_utilization=0.95`,
`max_num_seqs=512`, `max_model_len=16384`, prefix caching, eager mode), 2026-06-23:

| Quantity | Measured |
|---|---|
| decode @ 256 concurrent seqs | **2,026 tok/s** |
| decode @ 512 concurrent seqs | **2,595 tok/s** |
| prefill | **110,638 tok/s** |

The budget uses **2,026 tok/s** for decode, and this is **conservative** for two reasons: (a) the real
outputs average ~2,664 tokens, so the KV cache caps concurrency at ~330 (not 512), making the 256-seq
rate the representative figure; (b) eager mode (no CUDA graphs) is the *slower* path, so the
graph-captured production server is faster. So the GPU line is, if anything, an over-estimate. This
also resolves the pilot's apparent ~1,000 effective rate: that was depressed by the cold-start tax
below, not by slow compute. **H2 (FP8 economics) is GPU-verified**: the FP8 + kv-fp8 deploy loads and
serves on the FlashAttention backend (not the Triton path that pinned Gemma-bf16 to 750 tok/s).

**Cold-start finding (corrected, actionable).** Reading the 52 GiB checkpoint from the Modal Volume
(a 9P filesystem) is **~65 min on a cold host** (measured ~262 s/shard × 15); `prefetch` only helps a
host whose page cache is already warm, so it is not a reliable fix. **Downloading the weights from the
HF CDN to the container's local SSD loads in ~5 min** (measured), bypassing the 9P bottleneck. The fix
for the production server is therefore to serve weights from local disk or bake them into the image,
not from the Volume. The deployed path (`VllmEngine`) still mounts the Volume, and a real multi-day run
pays the load several times (smokes, `retries=3` preemptions, resumes), so the cold-start cost is booked
**explicitly as line 8 in §4** (~$17 worst on the 9P Volume; ~$3 if the local-SSD fix lands) rather than
hidden in overhead. (The `prefetch` flag is wired into `VllmEngine.load` as a harmless warm-cache help,
but the local-disk path is the real fix.)

### 4. The costed line items (measured, no contingency)

The table is the **worst case**: each variable line at its conservative end (the measured 2,026 tok/s
decode, the slow end of the estimated rates, the 9P-Volume cold-start). We expect to spend less, ~$210;
`scripts/full_run_budget.py` shows the full best–worst range ($185–$226). Presenting the worst case
keeps the number a ceiling.

| Line | Basis | $ |
|---|---|---|
| Qwen generation (Modal H100 FP8) | 100.5M tok (12,600 × 7,974) ÷ 2,026 tok/s = 15.3 GPU-hr × $3.95/hr | 60 |
| Gemma generation (OpenRouter) | 12,600 × (5,896 in × $0.12/M + 4,443 out × $0.37/M) | 30 |
| CoT judge (mimo) | 25,200 × (3,615 in × $0.435/M + 1,989 out × $0.87/M) | 83 |
| White-box re-encode + logit monitor (F2) | 85.3M tok ÷ ~6,000 tok/s = 4.3 GPU-hr × $3.95/hr | 17 |
| 49B organism anchor (F1, H200) | ~500 prompts = 2.3 H200-hr × $4.54/hr | 11 |
| Judge validation + infra validation | 2,000 × (3,615 in × $0.039/M + 1,989 out × $0.18/M) = $1 + benchmark/FP8 $7 | 8 |
| Cold-start weight loads (Modal) | ~4 loads = 4.3 GPU-hr × $3.95/hr (9P Volume; local-SSD → ~$3) | 17 |
| **Total (worst case, no contingency)** | | **$226** |

So **$121 is fully fixed** (Gemma, judge, validation: measured tokens × published rates), the Qwen GPU
line is at its conservative *measured* decode rate ($60 vs $47 at the saturated rate), and the cold-start
line and the two estimated small lines (re-encode, organism) sit above their expected values. The **CoT
judge ($83) is the largest line and is fixed**; the bill is judge-API-bound, not GPU-bound. **Lever
(surfaced):** swapping the judge to `openai/gpt-oss-120b` ($0.039/$0.180 vs mimo $0.435/$0.87) would
cut line 3 ~5× (~$65), but `design 0021`/`0027` fix mimo for upstream fidelity; kept, lever noted.

**Independently re-derived (2026-06-23).** An independent re-derivation (AI-assisted, checked by the authors) re-computed every line from the
pilot data and code. Each priced line reproduced (scope 25,200; judge $83 to the cent; Qwen $60 at the
measured 2,026 tok/s; the token model is if anything ~1–3% conservative). It confirmed **one real
omission**, the cold-start weight-load wall (now line 8, the only line that moves the budget up), and
two small over-counts since corrected: the Gemma input was a pre-TP-fix value with the `<think>` trace
still in the transcript (now trace-stripped, $32 → $30), and the F2 response constants were ~3% high.
Net **+$14**, $212 → $226. The audit's larger candidates (a ~$440 logit-monitor blow-up, ~$50 human
labeling) were all rejected on checking, as counterfactuals against rejected design choices or unpaid time.

**Price is verified, not assumed, and it is not a spot/fluctuating input.** Modal's H100 ($0.001097/s
= $3.95/hr) and H200 ($0.001261/s = $4.54/hr) are published *on-demand per-second* rates, re-verified
at modal.com/pricing on 2026-06-23. There is no intra-run market fluctuation. The only price risk is
Modal revising its rate card between now and the run, a discrete, announced event, so the action is to
**re-verify the rate at run time** (a 10-second check), not to carry a price contingency. The decode
rate is memory-bandwidth-bound and measured, so the worst case is **not** an invented 2× degradation;
it is the conservative *measured* rate, and the run is hard-capped regardless (§6).

### 5. Funding and descope

The full run's worst case is **$226** (expected **~$210**), every dollar traceable to §1–§4. It is small enough to self-fund, but it
is the basis of the grant request because the work is **already billing a card**: the pilot billed
**$158 of Modal GPU** (June cycle, two charges of $60 and $98, invoice-verified; no workspace spend
cap was set), with **no free-credit coverage**: the assumed $30/mo-per-person pool (`operations 0007`)
never applied, since credits attach to the workspace and all GPU ran in one workspace, and no offset
appears on the invoices (`operations 0020`). The full run's GPU lines (~$80 of items 1,4,5) cannot
run on free credits; none apply.

**The pilot's $158 is explained, not used to pad.** It was ~2.6× a clean token model (~$60), and the gap is
fully attributable: (a) the cold-start tax (the ~65-min Volume loads × many smoke/OOM/resume/failover
container starts), now fixed by the local-disk weight path and one warm container; (b) Gemma served on
a GPU (~$40), which the full run buys per-token on the API instead; (c) no spend cap. With all three
understood and addressed (the residual cold-start wall is now an explicit line, not a hidden risk), the
worst case needs no "just in case" cushion.

**Grant ask: $226, the §4 worst case, no round-up.** The table puts every variable line at its
conservative end (measured 2,026 tok/s decode, slow estimates, the 9P-Volume cold-start), so $226 is
the ceiling, not a midpoint needing a buffer. Both decode bounds are measured, so the worst case is not
an invented degradation; the lines below it are the cold-start (collapses to ~$3 if weights move to
local SSD) and the two not-yet-measured small lines. We **expect to spend ~$210**, and the `budget_usd`
+ workspace caps (§6) hard-stop spend at $250 regardless. Asking the worst case, with no percentage on
top, is the no-padding request.

**Descope ladder, if needed, in order:**
1. **Reuse the pilot's single-factor rollouts** (both models already generated at reps=3): removes
   most of Experiment A from lines 1–3 (~$40–60), no scientific loss for the convergent arm.
2. **Drop Experiment B (stacking)** (~$55 of lines 1–3): loses the dose-response axis, keeps the full
   convergent/criterion/specificity story.
3. **Cheaper judge** (line 3, ~$65), only if upstream-fidelity is relaxed.
4. **Drop the H200 organism anchor** (line 5, ~$8) last: it is the white-box positive control that
   makes a null interpretable.

### 6. Spend control and meter coverage (H4)

- **GPU-seconds are now metered.** `cv.meter.record_gpu(gpu, seconds)` converts a Modal batch's
  wall-time to USD at `config.GPU_RATES_PER_HR` and appends it to `_metrics.jsonl` in the schema
  `status.py` folds, so the self-host line shows up in `cv-status` next to the API legs. The Modal
  backend (`cv/serving/local/backend.py`) times each remote batch and records it.
- **A workspace spend cap is required and must be set in the Modal dashboard.** The pilot ran with
  none and billed $158 to a card unbudgeted (`operations 0020`); that is the concrete cost of this gap. The Modal CLI
  cannot set a workspace limit (`modal billing` is read-only), so it is a dashboard action, and it is
  the *only* control that catches spend from a raw `modal run`/`modal deploy` that bypasses the
  pipeline meter. The in-run `budget_usd` cap in `full.yaml` ($250, now inclusive of GPU-seconds)
  guards the pipeline; the dashboard cap guards everything else. Set both before the next GPU run.

## Alternatives considered

- **Full 256-config cube on all tasks.** Rejected: ~10× the cost, and not the EvalAwareBench paper's design.
- **Stacking F8 with the other factors.** Rejected: F8 is a trivial giveaway that saturates the
  dose-response; it is a single-factor control only (F8 stands alone).
- **Pad the estimate with a contingency.** Rejected under the auditable-budget principle: the throughput
  is measured, the overrun drivers are understood and fixed, so the residual is shown as an explicit
  sensitivity, not a cushion.
- **All-API (no self-host), per `operations 0002`.** Qwen's $3.20/M API output makes its generation
  ~3× the self-host GPU cost; self-hosting Qwen is the saving `operations 0013` identified.
- **Self-host Gemma too.** Rejected (`operations 0013`): Gemma's cheap API output beats its
  Triton-penalized 750 tok/s on a GPU; it stays on the API for generation, re-encoded on GPU only for
  the probe.

## Consequences

### Enables
- A defensible, fully auditable grant figure: worst case **$226** (expected **~$210**), ask **$226**
  with no round-up, every number reconstructable from `scripts/full_run_budget.py` and independently
  re-derived (AI-assisted, checked by the authors).
- `full.yaml` consistent with the costed scope: two models, the required Qwen Mamba `max_num_seqs=512`,
  a real `budget_usd`.
- The self-host line is visible to the meter and the budget cap.

### Forbids / constrains
- Do not pool pilot and full-run rollouts across serving stacks (reaffirms `operations 0002`/`0015`).
- The re-encode must share the rollout's build/dtype (`operations 0008`).
- TP reads the visible answer only, not the reasoning trace (`design 0029`, §2): a disclosed deviation
  from upstream, to keep TP and CoT evidence-independent.

### Risks
- **Decode concurrency:** the 2,026 tok/s figure is the measured rate at 256 concurrent sequences. For
  the measured mean output length (~2,664 tok), the KV cache supports ~337 concurrent, *above* 256, so
  the real rate sits between the 256-seq (2,026) and 512-seq (2,595) measurements and 2,026 is the
  conservative end. It would fall below 2,026 only if outputs averaged well beyond the measured ~2,664;
  the live run measures the true rate and the meter caps spend regardless.
- **HF re-encode prefill rate** (line 4, $9–$18) is the only un-measured rate; bounded above by the
  measured 110,638 vLLM prefill. A ~1-hour HF re-encode benchmark would pin it and remove the residual.

## References
- `scripts/full_run_budget.py` (line-item model), `scripts/bench_throughput.py` (live throughput
  benchmark + its result), `data/results/runs/pilot/_metrics.jsonl` (judge cost), pilot rollouts under
  `data/results/runs/pilot/rollouts/` (token counts).
- `operations 0013` (rates, hybrid serving), `0008` (re-encode), `0002` (superseded cost model).
- `design 0026` (two models, scope, F8 as control), `0028` (logit monitor), `0014` (reps=3),
  `0021`/`0027` (judge).
- EvalAwareBench, arXiv:2605.23055 §5.1, Fig 6 (single-factor), Fig 7/30 (stacking subset), and its
  detector definitions (SR/TP/CoT).
- `docs/full-run-open-decisions.md` (themes F/G/H, closed by this record).

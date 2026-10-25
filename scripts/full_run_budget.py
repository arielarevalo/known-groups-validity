"""Full-run budget model (operations 0019).

Every constant here is either measured (pilot data / live benchmark) or sourced
(operations records / live price tables), with the provenance in a comment. Run:

    python scripts/full_run_budget.py [decode_tps] [hf_prefill_tps]

Prints a line-itemised full-run GPU-hours x rate estimate with a contingency band.
The two throughput args default to the estimated central decode rate (operations 0019
§3: anchored to the measured 750 tok/s floor; the FP8 stack is GPU-verified but a clean
sustained-decode number was blocked by a vLLM multimodal-profiling snag). Pass overrides
to sensitivity-test. Reasoning is documented inline so every number is reconstructable.
"""
import sys

# ---------------------------------------------------------------------------
# 1. SCOPE  (faithful EvalAwareBench reproduction, arXiv 2605.23055, reps=3)
# ---------------------------------------------------------------------------
ITEMS = 200          # 100 task pairs x 2 valences (safety, capability)
SINGLE_CONFIGS = 9   # baseline + F1..F8
REPS = 3             # design/accepted/0014 (upstream used 10; we use 3)
MODELS = ["gemma-4-31b (API)", "qwen3.6-27b (Modal H100 FP8)"]

# Experiment A: single-factor variation on ALL items (= the pilot scope)
A_ROLL = ITEMS * SINGLE_CONFIGS * REPS                     # 5,400 / model

# Experiment B: factor-stacking on a 25-task/valence subset (paper Fig 7/30).
# F8 STANDS ALONE: the explicit "this is an evaluation" disclosure is a trivial/circular
# cue that saturates the dose-response, so it is run only as a single-factor control (in A)
# and EXCLUDED from the stacking combinatorics. Stacking is over F1-F7 (7 factors).
# Per active-factor count k=1..7, draw min(10, C(7,k)) combos; C(7,k)=7,21,35,35,21,7,1
# -> drawn 7,10,10,10,10,7,1 = 55 configs/item; k>=2 NEW stacked = 48 (the 7 singles reused from A).
B_ITEMS = 50
B_NEW_STACKED_CONFIGS = 48
B_ROLL = B_ITEMS * B_NEW_STACKED_CONFIGS * REPS           # 7,200 / model

GEN_PER_MODEL = A_ROLL + B_ROLL                            # 12,600 / model
TOTAL_GEN = GEN_PER_MODEL * 2                              # 25,200 rollouts

# ---------------------------------------------------------------------------
# 2. MEASURED PER-ROLLOUT TOKENS  (pilot saved rollouts/detectors, exact tokenizers)
#    5 generations/rollout: rollout + 2 SR (hypothetical+awareness) + 2 TP.
# ---------------------------------------------------------------------------
# gemma (served on OpenRouter): in is API-billed, out is API-billed
GEMMA_OUT = 4443     # rollout 1372 + SR 1625 + TP 1446 (measured)
GEMMA_IN  = 5896     # rollout 262 + 2 SR (~3568) + 2 TP (~2110): TP input is trace-STRIPPED
                     # (cv/detect.py visible-transcript fix), re-tokenized over the pilot Gemma set
# qwen (self-hosted): only OUTPUT drives decode cost; INPUT is cheap prefix-cached prefill
QWEN_OUT  = 7974     # rollout 2664 + SR 2750 + TP 2560 (measured)
QWEN_IN_UNIQUE = 3762  # prefix-cached unique prefill/rollout: rollout 262 + transcript once 2900 + 4 Qs
QWEN_RESP = 2581     # measured mean response tokens (re-encode response-pool + logit monitor)
QWEN_PROMPT = 240
GEMMA_RESP = 1335    # measured mean
GEMMA_PROMPT = 237

# CoT judge (mimo on OpenRouter), measured from pilot _metrics.jsonl (10,800 calls):
JUDGE_IN, JUDGE_OUT = 3615, 1989
JUDGE_USD_PER_ROLLOUT = 0.003303   # validated: $35.67 / 10,800 calls

# ---------------------------------------------------------------------------
# 3. RATES  (Modal: verified live at modal.com/pricing 2026-06-23, on-demand per-second, NOT spot;
#    OpenRouter: cv/config.PRICES_PER_M, verified live)
# ---------------------------------------------------------------------------
H100_HR = 3.95   # $0.001097/s, verified 2026-06-23 (re-verify at run time: it is a published rate)
H200_HR = 4.54   # $0.001261/s, verified 2026-06-23
P = {  # (USD/1M in, USD/1M out)
    "gemma": (0.15, 0.40),   # fp8 route cap, config.PRICES_PER_M (2026-09-03)
    "mimo":  (0.435, 0.87),
    "gptoss120b": (0.039, 0.180),  # 2nd judge for judge-vs-judge validation
}

# ---------------------------------------------------------------------------
# 4. THROUGHPUT  (live Modal benchmark; overridable via argv for sensitivity)
# ---------------------------------------------------------------------------
# MEASURED on a live Modal H100 (Qwen3.6-27B, FP8, kv-fp8, eager, max_num_seqs=512,
# scripts/bench_throughput.py, 2026-06-23): decode 2,026 tok/s @256 seqs, 2,595 @512 seqs;
# vLLM prefill 110,638 tok/s. Real outputs avg ~2,664 tok, so KV cache caps concurrency at
# ~330, making the 256-seq rate the representative production figure; eager mode makes it a
# conservative LOWER BOUND (graph-captured production is faster). No padding: this is the
# measured number, not a cushion.
DECODE_TPS = float(sys.argv[1]) if len(sys.argv) > 1 else 2026.0      # measured, conservative (256-seq, eager)
VLLM_PREFILL_TPS = 110638.0                                          # measured (vLLM prefill, generation inputs)
# The white-box re-encode, MEASURED on a live Modal H100 (Qwen3.6-27B, transformers 5.10.2,
# fp8 weights, ReencodeEngine, scripts/bench_reencode.py, 2026-09-03, 16 pilot rollouts, zero NaN):
#   capture (last-prompt-token + response-mean, 16 of 64 layers): 2,439 tok/s, 1,445 tok/rollout
#   logit monitor (8 positions/rollout, prompt + trace prefix + sentence): 3,834 tok/s, 5,383 tok/rollout
REENC_CAPTURE_TPS = 2439.0
REENC_LOGIT_TPS = 3834.0
REENC_CAPTURE_TOK = 1445
REENC_LOGIT_TOK = 5383            # at logit_max_positions=8; scales about linearly with the cap
WB_MODELS = 2                     # both targets (decided 2026-09-03); gemma assumed at the qwen rate until its smoke
HF_PREFILL_TPS = REENC_CAPTURE_TPS  # kept for the organism line (5), which scales it by 0.6
# GPU overhead beyond pure decode+prefill compute. NOT a "just in case" pad: itemised and
# bounded. (a) cold start: ~16s with prefetch (operations 0019) over a multi-hour run, <0.5%.
# (b) final drain: vLLM continuous batching keeps the 512-wide queue saturated until the last
# ~512 sequences drain at falling concurrency, ~one half-batch out of ~12,600, ~2-4%.
# (c) scheduling/prefill interleave not already counted: a few %. Total ~10%.
OVERHEAD = 0.10

def api_usd(intok, outtok, key):
    pin, pout = P[key]
    return (intok * pin + outtok * pout) / 1e6

# Fixed sub-scope constants
ORG_PROMPTS = 500            # 49B organism: Wood-Labs prompting variants, enough to establish probe fires
ORG_OUT = 2000
VAL_SAMPLE = 2000           # 2nd-judge validation sample
UNIQ_PROMPTS = ITEMS*SINGLE_CONFIGS + B_ITEMS*B_NEW_STACKED_CONFIGS   # re-encode dedup'd prompt set/model

# The uncertain inputs, with best / central / worst values. BOTH decode bounds are MEASURED (no
# invented sub-measurement floor). The small estimated lines (HF re-encode rate, 49B load hours) and
# the cold-start wall widen the worst case beyond the measured-decode worst.
#   decode tok/s   : 2,595 (measured @512)  | 2,026 (measured @256, conservative) | 2,026 (measured)
#   HF re-encode   : MEASURED 2026-09-03 (line 4 is fixed); hf_prefill only scales the organism line
#   H200 load hr   : 1.0                     | 1.5                                 | 2.0
#   cold-start usd : ~3 (local-SSD weights, ~5 min/load) | ~8 (~2 loads) | ~17 (9P Volume, ~4 loads x $4.28)
BEST    = dict(decode=2595.0, hf_prefill=REENC_CAPTURE_TPS, org_load=1.0, cold_start=3.0)
CENTRAL = dict(decode=DECODE_TPS, hf_prefill=REENC_CAPTURE_TPS, org_load=1.5, cold_start=8.0)
WORST   = dict(decode=2026.0, hf_prefill=REENC_CAPTURE_TPS, org_load=2.0, cold_start=17.0)


def lines_at(decode, hf_prefill, org_load, cold_start):
    """All eight line items (USD) at the given uncertain-input values. Fixed lines ignore the args."""
    L = {}
    # 1. Qwen generation (Modal H100 FP8): decode-dominated; prefill at the measured vLLM rate.
    qwen_s = ((GEN_PER_MODEL*QWEN_OUT)/decode + (GEN_PER_MODEL*QWEN_IN_UNIQUE)/VLLM_PREFILL_TPS) * (1+OVERHEAD)
    L["1. Qwen generation (Modal H100 FP8)"] = qwen_s/3600 * H100_HR
    # 2. Gemma generation (OpenRouter API): measured tokens x published price. FIXED.
    L["2. Gemma generation (OpenRouter API)"] = GEN_PER_MODEL * api_usd(GEMMA_IN, GEMMA_OUT, "gemma")
    # 3. CoT judge (OpenRouter mimo, both models): validated to the cent. FIXED.
    L["3. CoT judge (OpenRouter mimo)"] = TOTAL_GEN * JUDGE_USD_PER_ROLLOUT
    # 4. White-box re-encode + logit monitor (F2): MEASURED per-rollout seconds (both position
    #    schemes come from one forward; the logit monitor is its own pass), times the rollouts,
    #    times the models in the pass. Fixed now, not a rate estimate.
    reenc_s = GEN_PER_MODEL * (REENC_CAPTURE_TOK/REENC_CAPTURE_TPS + REENC_LOGIT_TOK/REENC_LOGIT_TPS) * WB_MODELS
    L["4. White-box re-encode + logit monitor (F2)"] = reenc_s/3600*(1+OVERHEAD) * H100_HR
    # 5. 49B organism anchor (F1, H200): dominated by the one-time load. Load hrs ESTIMATED.
    org_s = ORG_PROMPTS*ORG_OUT/(decode*0.6) + ORG_PROMPTS*2800/(hf_prefill*0.6)   # 49B ~0.6x the 27B rate
    L["5. 49B organism anchor (F1, H200)"] = (org_s/3600 + org_load) * H200_HR
    # 6. Judge validation (2nd judge over sample): FIXED.
    L["6. Judge validation (2nd judge)"] = VAL_SAMPLE * api_usd(JUDGE_IN, JUDGE_OUT, "gptoss120b")
    # 7. Infra validation (throughput benchmark + FP8 spot-check): one-off. FIXED.
    L["7. Infra validation"] = 7.0
    # 8. Cold-start weight loads (Modal GPU idle during load): the deployed path reads the 52 GiB
    # checkpoint from the 9P Volume (~65 min ~$4.28/load); a multi-day run with smokes/retries/resumes
    # pays several. Serving weights from local SSD / a baked image collapses each load to ~5 min.
    L["8. Cold-start weight loads (Modal)"] = cold_start
    return L


lo = lines_at(**BEST)
mid = lines_at(**CENTRAL)
hi = lines_at(**WORST)

print("="*78)
print("FULL-RUN BUDGET  (faithful EvalAwareBench reproduction, 2 models, reps=3)")
print("="*78)
print(f"Scope: A(single-factor)={A_ROLL}/model + B(stacking, F1-F7; F8 stands alone)={B_ROLL}/model "
      f"= {GEN_PER_MODEL}/model; total {TOTAL_GEN} rollouts.")
print("Ranges: best (decode 2,595 tok/s measured @512) -> worst (decode 2,026 measured @256); both measured.")
print()
# One $ column: a single number, or "low-high" where a line genuinely varies.
print(f"{'Line':47} {'$':>13}")
print("-"*62)
for k in mid:
    a, b = lo[k], hi[k]
    cell = f"${min(a,b):.0f}-${max(a,b):.0f}" if abs(a - b) >= 0.5 else f"${mid[k]:.0f}"
    print(f"{k:47} {cell:>13}")
print("-"*62)
tlo, thi = sum(lo.values()), sum(hi.values())
print(f"{'TOTAL (no contingency)':47} {f'${tlo:.0f}-${thi:.0f}':>13}")
print("="*62)
print(f"Expected ~${sum(mid.values()):.0f}; low=best (fast), high=worst (conservative). "
      f"Qwen / re-encode / organism / cold-start vary; the rest are fixed.")

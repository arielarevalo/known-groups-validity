"""One-off Modal throughput benchmark for the full-run budget (operations 0019).

Measures sustained vLLM decode throughput for Qwen3.6-27B at FP8 on a single H100,
under the exact serving config the full run uses (operations 0013 + pilot.yaml knobs),
plus a prefill-throughput probe that informs the re-encode line (F2).

Run:  modal run scripts/bench_throughput.py
Reuses the existing cv-hf-cache Volume (weights already cached) and the same image
recipe as src/cv/serving/local/app.py so layers are reused. Self-contained; not wired
into the pipeline. Delete after the budget is written.
"""
import os
import time

import modal

GPU = os.environ.get("BENCH_GPU", "H100")
MAX_MODEL_LEN = 16384

hf_cache = modal.Volume.from_name("cv-hf-cache", create_if_missing=True)
_CACHE_DIR = "/root/.cache/huggingface"

# Same image recipe as the serving app so Modal reuses cached layers.
_vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.11")
    .pip_install("vllm>=0.6.0", "huggingface_hub[hf_transfer]>=0.24")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": _CACHE_DIR,
          "CUDA_HOME": "/usr/local/cuda",
          "VLLM_CACHE_ROOT": f"{_CACHE_DIR}/vllm-compile-cache"})
)
try:
    _hf_secret = [modal.Secret.from_name("huggingface")]
except Exception:
    _hf_secret = []

app = modal.App("cv-bench-throughput")


# NO volume mount: reading the 52 GiB checkpoint from the cv-hf-cache Volume (9P) is
# ~65 min on a cold host (prefetch only helps a warm page cache). Downloading fresh from
# the HF CDN to the container's local SSD is far faster and bypasses the 9P bottleneck.
@app.function(gpu=GPU, image=_vllm_image, secrets=_hf_secret, timeout=3600)
def bench(hf_id: str = "Qwen/Qwen3.6-27B"):
    from vllm import LLM, SamplingParams

    t_load0 = time.time()
    # safetensors_load_strategy="prefetch": the cv-hf-cache Volume is a 9P FS that vLLM
    # will not auto-prefetch, so a default load reads the 52GiB checkpoint at ~10 MB/s
    # (~80 min); prefetch -> ~16s.
    # limit_mm_per_prompt={image:0,video:0}: Qwen3.6-27B loads as a vision-language model
    # and the default multimodal memory-profiling pass stalls before generate; we serve
    # text-only, so zeroing the mm budget skips that profiling.
    # enforce_eager skips CUDA-graph capture for fast, stall-free startup; at a saturating
    # batch (512 seqs) the eager-vs-graph decode gap is small, and eager is the SLOWER of
    # the two, so the measured tok/s is a conservative LOWER BOUND on the production rate.
    # gdn_prefill_backend="triton": Qwen3.6's Gated-DeltaNet layers default to a FlashInfer
    # GDN prefill kernel that is JIT-compiled on first run (10+ min, the stall we hit);
    # triton skips the JIT. It only changes the PREFILL kernel, not the decode kernels, so
    # the measured decode tok/s is unaffected.
    kw = dict(model=hf_id, dtype="bfloat16", trust_remote_code=True,
              enable_prefix_caching=True, max_model_len=MAX_MODEL_LEN,
              quantization="fp8", kv_cache_dtype="fp8",
              gpu_memory_utilization=0.95, max_num_seqs=512,
              enforce_eager=True, safetensors_load_strategy="prefetch",
              limit_mm_per_prompt={"image": 0, "video": 0})
    try:
        llm = LLM(gdn_prefill_backend="triton", **kw)
    except TypeError:
        llm = LLM(**kw)
    load_s = time.time() - t_load0
    out = {"gpu": GPU, "hf_id": hf_id, "load_s": round(load_s, 1), "runs": []}

    # Realistic input: ~250-token prompt (matches measured rollout prompt mean ~245).
    base = ("You are a helpful assistant. " + ("Consider the following scenario in detail "
            "and reason carefully before answering. ") * 12).strip()

    def decode_run(n_seqs, gen_tokens):
        prompts = [base + f" (item {i})" for i in range(n_seqs)]
        sp = SamplingParams(temperature=0.3, max_tokens=gen_tokens,
                            min_tokens=gen_tokens, ignore_eos=True)
        t0 = time.time()
        res = llm.generate(prompts, sp)
        dt = time.time() - t0
        gen = sum(len(r.outputs[0].token_ids) for r in res)
        return {"kind": "decode", "n_seqs": n_seqs, "gen_tokens_each": gen_tokens,
                "total_gen_tokens": gen, "wall_s": round(dt, 2),
                "decode_tok_per_s": round(gen / dt)}

    def prefill_run(n_seqs, in_tokens):
        # long inputs, 1 output token -> wall dominated by prefill of in_tokens*n_seqs
        filler = (base + " ") * (in_tokens // 60 + 1)
        prompts = [filler + f" #{i}" for i in range(n_seqs)]
        sp = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True)
        t0 = time.time()
        res = llm.generate(prompts, sp)
        dt = time.time() - t0
        pin = sum(len(r.prompt_token_ids) for r in res)
        return {"kind": "prefill", "n_seqs": n_seqs, "total_prompt_tokens": pin,
                "wall_s": round(dt, 2), "prefill_tok_per_s": round(pin / dt)}

    # warmup then measured runs (smaller, eager-mode: get a number fast)
    decode_run(8, 64)
    out["runs"].append(decode_run(256, 512))    # saturated-ish concurrency
    out["runs"].append(decode_run(512, 512))    # ~real run concurrency
    out["runs"].append(prefill_run(256, 2800))   # re-encode-like (prompt+response len)
    return out


@app.local_entrypoint()
def main():
    import json
    r = bench.remote()
    print("BENCH_RESULT_JSON=" + json.dumps(r))

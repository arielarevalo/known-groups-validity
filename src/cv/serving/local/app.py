"""Modal serving app: self-hosted targets on GPU (docs/operations/accepted/0007).

Deploy:   modal deploy src/cv/serving/local/app.py
Smoke:    modal run    src/cv/serving/local/app.py   (runs `_smoke` on a small model)

Two engines, selected by the caller (`cv.config.serving_mode`):

  - `VllmEngine` — black-box PILOT. vLLM batched offline generation. Full reasoning
    traces for every target (so gemma CoT works — the gap that killed W&B Inference).
  - `ReencodeEngine` — the white-box pass (operations 0008). Transformers, no
    generation: one teacher-forced forward over saved text yields pooled per-layer
    residual vectors (the probe's input) and the log-probability of a forced
    continuation (the logit monitor). An HTTP inference API can never return
    these, which is why serving is a Modal function, not a served endpoint.

Both are parametrized by `hf_id` (`modal.parameter`), so each target model gets its
own warm container pool; `@modal.enter` loads weights once per container and they
stay warm across the contiguous GPU stages (rollout -> SR -> TP; the CoT judge runs
last on the API, so the GPU isn't held idle during it; see the `generate.py` driver) via
`scaledown_window`.

Config -> deploy: the app name and GPU are read from the environment at deploy time
(`CV_MODAL_APP`, `CV_MODAL_GPU`), which `cv.serving.local.backend.deploy_command()`
sets from the loaded profile. Gated weights (e.g. gemma) need a Modal secret named
`huggingface` carrying `HF_TOKEN` (create once: `modal secret create huggingface
HF_TOKEN=...`).
"""
# NOTE: deliberately NO `from __future__ import annotations` — Modal resolves the
# `modal.parameter()` class-parameter annotations (e.g. `hf_id: str`) at decoration
# time, and PEP 563 stringised annotations break that.

import os

import modal

APP_NAME = os.environ.get("CV_MODAL_APP", "cv-serving")
GPU = os.environ.get("CV_MODAL_GPU", "H100")
# Idle window before the warm container is released. The driver's gaps between
# chunks are seconds, but the driver also uploads a W&B checkpoint of the whole run
# directory every few chunks, and that upload can take minutes once the rollout file
# is large. A window shorter than one upload drops the container and pays a weight
# reload on the next chunk. With weights on local SSD a reload
# costs about 5 min; 300 s of idle costs about the same, so 300 s is the balance point.
# Cross-stage gaps are not bridged on purpose: the driver runs the API-only CoT judge
# last so the GPU stages stay contiguous (see generate.py).
SCALEDOWN_S = int(os.environ.get("CV_MODAL_SCALEDOWN_S", "300"))
# Per-input execution budget: one batch of up to a few hundred sequences at max_tokens.
TIMEOUT_S = int(os.environ.get("CV_MODAL_TIMEOUT_S", "3600"))
# Container start budget, separate from TIMEOUT_S (modal >= 1.4 `startup_timeout`;
# without it the start shares TIMEOUT_S, and a pre-launch check measured a cold load
# longer than that). Covers the HF download to local SSD plus the engine load.
STARTUP_TIMEOUT_S = int(os.environ.get("CV_MODAL_STARTUP_TIMEOUT_S", "2700"))
# One retry, not three: a start that fails once is far more likely a deterministic
# fault than a preemption, and every retry pays a full weight load.
RETRIES = int(os.environ.get("CV_MODAL_RETRIES", "1"))
# Cap sequence length: our longest sequence is a ~12k-token SR/TP transcript plus a
# 2k follow-up; the model-card default (262144 for gemma-4) demands a 33 GiB KV
# reservation per max-len sequence, which cannot fit next to 61 GB of weights on an
# 80 GB card (verified OOM 2026-06-12) and would strangle batching even where it fits.
MAX_MODEL_LEN = int(os.environ.get("CV_MODAL_MAX_LEN", "16384"))
# vLLM tuning passed from the driver (operations 0013), read at deploy time from env so
# the deployed engine bakes the config. All optional; unset = vLLM defaults (known-green).
# CV_VLLM_VERSION pins the exact vLLM build of the image; empty = the floor spec below.
# Pin it to the version the pre-launch smoke reports (`[vllm] version ...` in the
# container log), so the launch image is the one that was proven.
VLLM_VERSION = os.environ.get("CV_VLLM_VERSION", "").strip()
VLLM_QUANTIZATION = os.environ.get("CV_VLLM_QUANTIZATION", "").strip()
VLLM_KV_CACHE_DTYPE = os.environ.get("CV_VLLM_KV_CACHE_DTYPE", "").strip()
VLLM_GPU_MEM_UTIL = float(os.environ.get("CV_VLLM_GPU_MEM_UTIL", "0") or "0")
VLLM_MAX_NUM_SEQS = int(os.environ.get("CV_VLLM_MAX_NUM_SEQS", "0") or "0")
# The re-encode's weight precision: "" = bf16, "fp8" = weights
# quantised to fp8 at load (transformers FineGrainedFP8Config), the same weight
# precision the vLLM generation ran at. Baked into the HF image env at deploy time
# like the vLLM knobs; the engine reports it and the preflight compares it.
REENCODE_QUANT = os.environ.get("CV_REENCODE_QUANT", "").strip()
# Exact transformers build of the re-encode image ("" = the floor spec). 5.16.1 crashes the
# fp8 quantizer on a dense model (verified live 2026-09-03); pin the build the smoke proved.
TRANSFORMERS_VERSION = os.environ.get("CV_TRANSFORMERS_VERSION", "").strip()


def fp8_exclusions(shapes, block=(128, 128)) -> list[str]:
    """Name patterns for the linears the fine-grained fp8 quantizer must leave in bf16:
    every one whose weight does not tile in whole blocks.

    The quantizer allocates the block scales with ``torch.empty`` and its load-time
    tiling cannot cover a partial block, so such a module keeps an uninitialized
    scale and the forward reads memory garbage (verified 2026-09-09 on Qwen3.6-27B:
    ``linear_attn.in_proj_a`` and ``in_proj_b`` are 5120 x 48, and the residual
    stream went NaN from a random layer on, finite on one day and not on the next).
    ``shapes`` is an iterable of ``(module name, in_features, out_features)``. The
    result is the suffix after the last layer index when that suffix names no
    well-formed module, else the full name. Matched by transformers with
    ``full_name.endswith(pattern)``."""
    block_out, block_in = block
    shapes = list(shapes)
    fits = {n for n, i, o in shapes if not (o % block_out or i % block_in)}
    out: list[str] = []
    for name, i, o in shapes:
        if not (o % block_out or i % block_in):
            continue
        parts = name.split(".")
        idx = max((k for k, part in enumerate(parts) if part.isdigit()), default=-1)
        suffix = ".".join(parts[idx + 1:])
        pattern = name if any(f.endswith(suffix) for f in fits) else suffix
        if pattern not in out:
            out.append(pattern)
    return out


app = modal.App(APP_NAME)

# Weights live on the container's LOCAL disk, downloaded from the HF CDN at start
# (hf_transfer, about 5 min for the 52 GiB Qwen checkpoint). They are deliberately NOT
# served from the `cv-hf-cache` Volume: that Volume is a 9P filesystem, and a cold
# read of the checkpoint through it measured about 65 min (about $4.28 of H100) per
# container start, the largest single component of the pilot overrun (operations
# 0019 hard prerequisite 2, operations 0020). The
# benchmark that produced the budget numbers used exactly this no-Volume shape.
_CACHE_DIR = "/root/.cache/huggingface"

# vLLM brings its own pinned torch; keep the HF image separate + lean.
# CUDA *devel* base (not debian_slim): gemma-4 forces vLLM's Triton attention
# backend, whose JIT kernel compilation requires nvcc/CUDA_HOME at runtime
# (verified failure 2026-06-12: "Could not find nvcc ... cuda_home doesn't exist").
_vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.11")
    .pip_install(f"vllm=={VLLM_VERSION}" if VLLM_VERSION else "vllm>=0.6.0",
                 "huggingface_hub[hf_transfer]>=0.24")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": _CACHE_DIR,
          "CUDA_HOME": "/usr/local/cuda",
          # torch.compile cache on local disk too: a recompile costs about 24 s per
          # container start, nothing next to the weight reload it used to share a
          # Volume with.
          "VLLM_CACHE_ROOT": f"{_CACHE_DIR}/vllm-compile-cache",
          # Baked at deploy time like the CV_VLLM_* knobs below; without this the
          # deploy-time value was silently ignored inside the container.
          "CV_MODAL_MAX_LEN": os.environ.get("CV_MODAL_MAX_LEN", str(MAX_MODEL_LEN)),
          # The app name and GPU too, so `config()` reports the deploy's values, not the
          # module defaults the container would otherwise re-read (seen 2026-09-03).
          "CV_MODAL_APP": APP_NAME, "CV_MODAL_GPU": GPU,
          # Bake the vLLM tuning into the IMAGE env at deploy time. Module-level
          # os.environ reads run again inside the container, where the deploy shell's
          # CV_VLLM_* vars do not exist; baking them here is what actually carries the
          # FP8/tuning config into the running container (operations 0013; fixed
          # 2026-06-13 after the first Qwen smoke silently ran bf16).
          "CV_VLLM_QUANTIZATION": os.environ.get("CV_VLLM_QUANTIZATION", ""),
          "CV_VLLM_KV_CACHE_DTYPE": os.environ.get("CV_VLLM_KV_CACHE_DTYPE", ""),
          "CV_VLLM_GPU_MEM_UTIL": os.environ.get("CV_VLLM_GPU_MEM_UTIL", "0"),
          "CV_VLLM_MAX_NUM_SEQS": os.environ.get("CV_VLLM_MAX_NUM_SEQS", "0")})
)
_hf_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.4", f"transformers=={TRANSFORMERS_VERSION}" if TRANSFORMERS_VERSION else "transformers>=4.57",
                 "accelerate>=1.0", "kernels>=0.12,<0.13", "huggingface_hub[hf_transfer]>=0.24")  # kernels: the fp8 matmul; 5.10.2 wants 0.12
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_XET_HIGH_PERFORMANCE": "1", "HF_HOME": _CACHE_DIR,
          "CV_REENCODE_QUANT": os.environ.get("CV_REENCODE_QUANT", ""),
          "CV_MODAL_APP": APP_NAME, "CV_MODAL_GPU": GPU})
)
# wandb in both images so the GPU container can attach a system-metrics-only run to the
# canonical `cv-<run_id>` (its background monitor auto-detects the NVIDIA GPU via pynvml
# and streams utilization/memory/power/temp). Only the system monitor runs here; no
# application metrics are logged from the container. See `_GpuSystemMonitor` below.
_vllm_image = _vllm_image.pip_install("wandb>=0.27")
_hf_image = _hf_image.pip_install("wandb>=0.27")

try:
    _hf_secret = [modal.Secret.from_name("huggingface")]
except Exception:  # noqa: BLE001 — absent secret only matters for gated weights at run time
    _hf_secret = []

# Optional W&B credential for the GPU system monitor. Absent secret is fine: the monitor
# is best-effort and no-ops without WANDB_API_KEY (offline/dev/CI unaffected), mirroring
# the `huggingface` secret pattern above and `cv.observability`'s WANDB_API_KEY gating.
try:
    _wandb_secret = [modal.Secret.from_name("wandb")]
except Exception:  # noqa: BLE001 — absent secret only disables the GPU system charts
    _wandb_secret = []


def _is_template_error(e: BaseException) -> bool:
    """A chat-template rejection (unknown kwarg, unsupported role): the fallback
    ladder in `VllmEngine.generate` exists for these and nothing else."""
    if isinstance(e, (TypeError, ValueError)):
        return True
    name = type(e).__name__
    return "Template" in name or "Jinja" in name


def _loaded_revision(hf_id: str) -> str:
    """The commit the weights on local disk came from: the HF cache keeps each
    snapshot under its commit hash. Reported by both engines so the rollout file and
    the re-encode carry the build they ran on (operations 0008: pin the revision)."""
    import glob

    repo = f"models--{hf_id.replace('/', '--')}"
    snaps = glob.glob(os.path.join(_CACHE_DIR, "hub", repo, "snapshots", "*"))
    if not snaps:
        return ""
    return os.path.basename(max(snaps, key=os.path.getmtime))


def _cut_after_content(rendered: str, content: str) -> str:
    """The rendered template cut right after the last occurrence of the user's text.
    Templates may trim the content (gemma applies `| trim`), so the search uses the
    stripped text. A miss raises: a silent fallback reads a different token."""
    needle = content.strip()
    cut = rendered.rfind(needle) if needle else -1
    if cut < 0:
        raise ValueError("cut_after_content: the user content is not in the rendered template")
    return rendered[:cut + len(needle)]


def _user_span(bare: str, content: str, prompt_ids: list[int], ids) -> tuple[int, int]:
    """Token indices ``(start, end)`` of the user's text inside the generation prompt, both
    inclusive. `bare` is the template without the generation prompt, and `ids` tokenizes text.
    Both cuts must be token prefixes of `prompt_ids`, or the read would land on another token."""
    cut = _cut_after_content(bare, content)
    head = cut[:len(cut) - len(content.strip())]
    cut_ids, head_ids = ids(cut), ids(head)
    if prompt_ids[:len(cut_ids)] != cut_ids or prompt_ids[:len(head_ids)] != head_ids:
        raise ValueError("read_positions: the user content is not a token prefix of the prompt")
    if len(head_ids) >= len(cut_ids):
        raise ValueError("read_positions: the user content has no tokens")
    return len(head_ids), len(cut_ids) - 1


def _fold_system(conv: list[dict]) -> list[dict]:
    """Some chat templates (e.g. gemma) reject a `system` role. Fold a leading
    system turn into the first user turn so the same batch path serves all targets."""
    if not conv or conv[0].get("role") != "system":
        return conv
    system = conv[0]["content"]
    rest = conv[1:]
    for i, m in enumerate(rest):
        if m.get("role") == "user":
            merged = dict(m, content=f"{system}\n\n{m['content']}")
            return rest[:i] + [merged] + rest[i + 1:]
    return [{"role": "user", "content": system}] + rest


class _GpuSystemMonitor:
    """Best-effort GPU system charts on the canonical `cv-<run_id>` run.

    The laptop orchestrator (cv.observability.WandbTracker) holds `cv-<run_id>` open as
    PRIMARY and writes ops/exp metrics to it; the Modal GPU containers do the actual
    inference but ran with no wandb process, so the only system charts came from the
    orchestrator Mac, never the GPU. Here the serving container opens a SECOND writer
    on the SAME run id in W&B shared/distributed mode, so its background system monitor
    (which auto-detects the NVIDIA GPU via pynvml) streams utilization/memory/power/temp
    onto the canonical run alongside the orchestrator's metrics.

    Shared mode (verified against wandb 0.27.1):
        wandb.init(
            project=..., entity=..., id=<safe cv-run-id>, resume="allow",
            settings=wandb.Settings(mode="shared", x_primary=False, x_label="modal-gpu"),
        )
    `mode="shared"` is the supported way for multiple processes to write one run;
    `x_primary=False` marks this writer secondary (the orchestrator is primary, the
    default `x_primary=True`); the distinct `x_label` namespaces this writer's system
    stream. We log NO application metrics or traces here — only the system monitor runs.

    Gating (no-op, never raises): does nothing unless BOTH WANDB_API_KEY is present (the
    `wandb` Modal secret) AND a run name was threaded in, so offline/dev/CI containers
    are unaffected. Every step is wrapped so a wandb failure never breaks serving (the
    GPU work is the critical path); failures print and degrade to no-op.
    """

    def __init__(self, run_name: str | None, *, project: str, entity: str) -> None:
        self._run = None
        if not run_name or not os.environ.get("WANDB_API_KEY", "").strip():
            return  # no run id threaded in, or no credential -> stay a no-op
        try:
            import re

            import wandb

            safe = re.sub(r"[^A-Za-z0-9_.-]", "-", run_name)
            # init_timeout: this runs inside @modal.enter on a billed GPU; an unreachable
            # W&B must not eat the start budget.
            self._run = wandb.init(
                project=project, entity=(entity or None), id=safe, resume="allow",
                job_type="serving", name=safe,
                settings=wandb.Settings(mode="shared", x_primary=False, x_label="modal-gpu",
                                        init_timeout=90.0),
            )
            print(f"[wandb] GPU system monitor attached to {safe} (shared, secondary)")
        except Exception as e:  # noqa: BLE001 — telemetry is best-effort; never sink serving
            print(f"[wandb] GPU system monitor skipped: {type(e).__name__}: {e}")
            self._run = None

    def finish(self) -> None:
        try:
            if self._run is not None:
                self._run.finish()
        except Exception as e:  # noqa: BLE001
            print(f"[wandb] GPU system monitor finalize skipped: {type(e).__name__}: {e}")
        finally:
            self._run = None


@app.cls(
    gpu=GPU,
    image=_vllm_image,
    secrets=_hf_secret + _wandb_secret,
    scaledown_window=SCALEDOWN_S,
    timeout=TIMEOUT_S,
    startup_timeout=STARTUP_TIMEOUT_S,
    retries=RETRIES,
    # One container per engine class, by design (one warm container per model): bounds a
    # stray second driver or `modal run` to the same GPU instead of a second H100 that
    # pays its own weight load.
    max_containers=1,
)
class VllmEngine:
    hf_id: str = modal.parameter()
    # The commit to load ("" = the repo head). The re-encode must read the build that
    # generated the text (operations 0008), so both engines take the profile's pin.
    revision: str = modal.parameter(default="")
    # Canonical run name (`cv-<run_id>`) + W&B project/entity threaded from ModalBackend so
    # the GPU system monitor can attach to it. Optional (defaults ""): an empty run_name
    # keeps `_GpuSystemMonitor` a no-op, so a bare deploy / smoke serves with no wandb.
    run_name: str = modal.parameter(default="")
    wandb_project: str = modal.parameter(default="")
    wandb_entity: str = modal.parameter(default="")

    @modal.enter()
    def load(self) -> None:
        from vllm import LLM

        # Start the GPU system monitor for this container's serving lifetime (no-op when
        # un-threaded or WANDB_API_KEY absent); torn down in `_close` on scaledown.
        self._mon = _GpuSystemMonitor(
            self.run_name or None, project=self.wandb_project, entity=self.wandb_entity,
        )

        # enable_prefix_caching: every rollout shares the long system prompt and SR/TP
        # reuse the transcript, so KV-cache reuse on the shared prefix cuts prefill time
        # (and therefore GPU-seconds) directly — the cheapest real cost lever we have.
        # Optional tuning (operations 0013); only non-empty knobs are passed, so an
        # unconfigured deploy is identical to the known-green bf16 path.
        tuning = {}
        if VLLM_QUANTIZATION:
            tuning["quantization"] = VLLM_QUANTIZATION
        if VLLM_KV_CACHE_DTYPE:
            tuning["kv_cache_dtype"] = VLLM_KV_CACHE_DTYPE
        if VLLM_GPU_MEM_UTIL > 0:
            tuning["gpu_memory_utilization"] = VLLM_GPU_MEM_UTIL
        if VLLM_MAX_NUM_SEQS > 0:
            tuning["max_num_seqs"] = VLLM_MAX_NUM_SEQS
        import time as _time

        import vllm as _vllm

        # The version line is what the launch pin is copied from (CV_VLLM_VERSION).
        print(f"[vllm] version {_vllm.__version__} | gpu {GPU} | model {self.hf_id}")
        common = dict(model=self.hf_id, dtype="bfloat16", trust_remote_code=True,
                      enable_prefix_caching=True, max_model_len=MAX_MODEL_LEN, **tuning)
        if self.revision:
            common["revision"] = self.revision
        # Two start-time workarounds the throughput benchmark proved necessary for
        # Qwen3.6-27B on this exact config (scripts/bench_throughput.py):
        #   limit_mm_per_prompt={image:0,video:0}: the model loads as a vision-language
        #     model and the default multimodal memory-profiling pass stalls before generate.
        #   gdn_prefill_backend="triton": the Gated-DeltaNet prefill kernel otherwise
        #     JIT-compiles on first use (10+ min on a billed GPU).
        # Plus safetensors_load_strategy="prefetch", a parallel checkpoint read. Each is
        # tried in turn and dropped on TypeError, so a build or a model that rejects a
        # kwarg still loads. Which set won is printed, so the log shows the real config.
        optional = [
            ("gdn_prefill_backend", "triton"),
            ("limit_mm_per_prompt", {"image": 0, "video": 0}),
            ("safetensors_load_strategy", "prefetch"),
        ]
        t0 = _time.time()
        self.llm = None
        for drop in range(len(optional) + 1):
            extra = dict(optional[drop:])
            try:
                self.llm = LLM(**extra, **common)
                print(f"[vllm] engine up in {_time.time() - t0:.0f}s with {sorted(extra)}")
                break
            except TypeError as e:
                print(f"[vllm] {sorted(extra)} rejected ({e}); retrying with fewer kwargs")
        if self.llm is None:
            raise RuntimeError("vLLM rejected every kwarg set; see the log above")

    @modal.method()
    def config(self) -> dict:
        """The engine's LIVE configuration, for the driver's preflight to compare
        against the profile before the first paid batch: a
        bare `modal deploy` silently ships bf16 at the default max_num_seqs, and an
        H200 anchor deploy on a shared app name silently moves this engine."""
        import vllm as _vllm

        return {
            "engine": "vllm", "vllm_version": _vllm.__version__, "app": APP_NAME, "gpu": GPU,
            "model": self.hf_id, "revision": _loaded_revision(self.hf_id), "revision_pin": self.revision,
            "max_model_len": MAX_MODEL_LEN,
            "quantization": VLLM_QUANTIZATION, "kv_cache_dtype": VLLM_KV_CACHE_DTYPE,
            "gpu_memory_utilization": VLLM_GPU_MEM_UTIL, "max_num_seqs": VLLM_MAX_NUM_SEQS,
        }

    @modal.method()
    def generate(self, conversations: list[list[dict]], temperature: float, max_tokens: int) -> list[dict]:
        """Batched chat generation. Returns one dict per conversation:
        ``{"text", "finish_reason", "thinking"}``. `text` is the raw output (the
        model's native `<think>…</think>` stays intact for thinking models);
        `finish_reason` is vLLM's ("stop" | "length"), the truncation signal the
        caller needs; `thinking` says whether the thinking
        template was on for this batch."""
        from vllm import SamplingParams

        sp = SamplingParams(temperature=float(temperature or 0.0), max_tokens=int(max_tokens))
        # Thinking-capable templates (gemma-4, qwen3.x) gate reasoning behind
        # enable_thinking; without it gemma-4 emits no trace at all (verified
        # 2026-06-12). Fall back through kwarg/system-folding combinations for
        # templates that reject either. ONLY template errors move down the ladder:
        # a CUDA or scheduler fault must surface as a fault, or a retry lands on
        # the no-thinking rung and returns trace-free rollouts that look successful.
        attempts = [
            (conversations, {"enable_thinking": True}),
            ([_fold_system(c) for c in conversations], {"enable_thinking": True}),
            (conversations, None),
            ([_fold_system(c) for c in conversations], None),
        ]
        last_err = None
        for i, (convs, tk) in enumerate(attempts, 1):
            try:
                kwargs = {"chat_template_kwargs": tk} if tk else {}
                outs = self.llm.chat(convs, sp, add_generation_prompt=True, **kwargs)
            except Exception as e:  # noqa: BLE001
                if not _is_template_error(e):
                    raise
                last_err = e
                print(f"[vllm] attempt {i} rejected by the chat template ({type(e).__name__}); next rung")
                continue
            thinking = tk is not None
            if i > 1:
                print(f"[vllm] generate ok on attempt {i} (thinking={thinking})")
            return [{"text": o.outputs[0].text, "finish_reason": str(o.outputs[0].finish_reason or ""),
                     "thinking": thinking} for o in outs]
        raise last_err

    @modal.exit()
    def _close(self) -> None:
        # finish() the system-monitor run on container scaledown so the canonical run's
        # GPU stream closes cleanly; best-effort, never raises.
        self._mon.finish()


@app.cls(
    gpu=GPU,
    image=_hf_image,
    secrets=_hf_secret + _wandb_secret,
    scaledown_window=SCALEDOWN_S,
    timeout=TIMEOUT_S,
    startup_timeout=STARTUP_TIMEOUT_S,
    retries=RETRIES,
    # One container per engine class, by design (one warm container per model): bounds a
    # stray second driver or `modal run` to the same GPU instead of a second H100 that
    # pays its own weight load.
    max_containers=1,
)
class ReencodeEngine:
    """The white-box pass: teacher-forced forwards over saved text, no generation.

    `capture` returns, per row, the residual stream at the last prompt token and the
    mean over the response tokens, at every ``layer_stride``-th decoder layer, fp16.
    `sentence_logprob` returns the log-probability of a probe sentence appended to
    a context. Rows are packed into forwards under a padded-token budget, so the
    pass is batched, and every position is exact: no cache
    cropping, which the hybrid-attention targets could not support anyway.
    """
    hf_id: str = modal.parameter()
    # The commit to load ("" = the repo head, reported back so the profile can pin it).
    revision: str = modal.parameter(default="")
    run_name: str = modal.parameter(default="")
    wandb_project: str = modal.parameter(default="")
    wandb_entity: str = modal.parameter(default="")

    @modal.enter()
    def load(self) -> None:
        import time as _time

        import torch
        import transformers
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        self._mon = _GpuSystemMonitor(
            self.run_name or None, project=self.wandb_project, entity=self.wandb_entity,
        )
        self.torch = torch
        rev = self.revision or None
        kw: dict = {"dtype": torch.bfloat16, "device_map": "cuda", "trust_remote_code": True}
        cfg = AutoConfig.from_pretrained(self.hf_id, revision=rev, trust_remote_code=True)
        # The checkpoint's own class (e.g. Qwen3_5ForConditionalGeneration, the class vLLM
        # serves), not the causal-LM auto mapping, so the weight layout is the one on disk.
        arch = (getattr(cfg, "architectures", None) or [None])[0]
        Model = getattr(transformers, arch, None) if arch else None
        if Model is None:
            Model = AutoModelForCausalLM
        # The linears kept in bf16 under fp8: the head (as vLLM keeps it) and every
        # linear that does not tile in whole 128 x 128 blocks (see `fp8_exclusions`).
        # The skeleton on the meta device costs no memory and gives every shape.
        self.fp8_excluded: list[str] = []
        if REENCODE_QUANT == "fp8":
            from accelerate import init_empty_weights
            from transformers import FineGrainedFP8Config

            with init_empty_weights():
                skeleton = Model(cfg)
            shapes = [(n, m.in_features, m.out_features) for n, m in skeleton.named_modules()
                      if isinstance(m, torch.nn.Linear)]
            del skeleton
            self.fp8_excluded = ["lm_head"] + [x for x in fp8_exclusions(shapes) if x != "lm_head"]
            kw["quantization_config"] = FineGrainedFP8Config(modules_to_not_convert=self.fp8_excluded)
        elif REENCODE_QUANT:
            raise RuntimeError(f"unknown CV_REENCODE_QUANT {REENCODE_QUANT!r}; use '' (bf16) or 'fp8'")
        t0 = _time.time()
        self.tok = AutoTokenizer.from_pretrained(self.hf_id, revision=rev)
        self.model = Model.from_pretrained(self.hf_id, revision=rev, config=cfg, **kw)
        self.model.eval()
        self.layers = self._decoder_layers()
        self.loaded_revision = _loaded_revision(self.hf_id)
        if REENCODE_QUANT == "fp8":
            # transformers tries a DeepGEMM kernel first on EVERY fp8 linear call, and that
            # attempt asks the Hub for the kernel's version tags each time: 1,000 requests
            # in five minutes, HTTP 429, and 135 tok/s (measured 2026-09-03). DeepGEMM needs
            # an SM100 GPU, which the H100 is not, so the call always fell back to the
            # cached Triton kernel anyway. Skip the attempt: the Triton path is the real one.
            import transformers.integrations.finegrained_fp8 as _ff

            def _no_deepgemm(*_a, **_k):
                raise ImportError("DeepGEMM disabled in ReencodeEngine: SM100+ only; the Triton fp8 path serves")

            _ff.deepgemm_fp8_fp4_linear = _no_deepgemm
            # The Triton fp8 matmul returns NaN for a 3-D activation with batch > 1 and is
            # correct for a single row (measured 2026-09-03: every batched forward NaN,
            # single rows finite). Flatten to 2-D at the kernel boundary: the same maths,
            # row by row, whatever the batch.
            _orig_fp8_linear = _ff.fp8_linear

            def _fp8_linear_2d(input, *a, **k):
                if input.dim() > 2:
                    out = _orig_fp8_linear(input.reshape(-1, input.shape[-1]), *a, **k)
                    return out.reshape(*input.shape[:-1], out.shape[-1])
                return _orig_fp8_linear(input, *a, **k)

            _ff.fp8_linear = _fp8_linear_2d
        # One tiny forward at load fetches the Triton kernel once, while the start is paid.
        # Its logits must be finite: a bad load (an uninitialized fp8 scale, an overflow)
        # shows here, before any batch is metered as a success.
        with torch.no_grad():
            warm = self.tok("warm up", return_tensors="pt").to("cuda")
            out = self.model(**warm, use_cache=False)
        if not bool(torch.isfinite(out.logits).all()):
            raise RuntimeError(
                f"non-finite logits on the warm-up forward of {self.hf_id} at quant {REENCODE_QUANT or 'bf16'} "
                f"(kept in bf16: {self.fp8_excluded}); the load is unusable"
            )
        print(f"[reencode] transformers {transformers.__version__} | {type(self.model).__name__} | gpu {GPU} | model {self.hf_id} "
              f"| revision {self.loaded_revision or '?'} | quant {REENCODE_QUANT or 'bf16'} "
              f"| bf16 kept {self.fp8_excluded} | {len(self.layers)} layers | up in {_time.time() - t0:.0f}s")

    def _decoder_layers(self):
        m = self.model
        for path in ("model.layers", "model.language_model.layers", "language_model.model.layers",
                     "language_model.layers"):
            obj = m
            try:
                for part in path.split("."):
                    obj = getattr(obj, part)
                return list(obj)
            except AttributeError:
                continue
        raise RuntimeError(f"cannot find the decoder layers of {type(m).__name__}; add its path to _decoder_layers")

    @modal.method()
    def config(self) -> dict:
        """The engine's LIVE build, for the preflight (gpu, precision, revision)."""
        import transformers

        return {"engine": "reencode", "transformers_version": transformers.__version__, "app": APP_NAME,
                "gpu": GPU, "model": self.hf_id, "revision": self.loaded_revision, "revision_pin": self.revision,
                "quantization": REENCODE_QUANT, "dtype": str(getattr(self.model, "dtype", "")),
                "fp8_excluded": list(self.fp8_excluded), "n_layers": len(self.layers)}

    def _render(self, conv: list[dict], add_generation_prompt: bool = True) -> str:
        """The chat template as generation applied it (thinking on, system folded
        when the template rejects the role). Only template errors move down the ladder."""
        attempts = [(conv, {"enable_thinking": True}), (_fold_system(conv), {"enable_thinking": True}),
                    (conv, {}), (_fold_system(conv), {})]
        last = None
        for c, kw in attempts:
            try:
                return self.tok.apply_chat_template(c, add_generation_prompt=add_generation_prompt,
                                                    tokenize=False, **kw)
            except Exception as e:  # noqa: BLE001
                if not _is_template_error(e):
                    raise
                last = e
        raise last

    def _ids(self, text: str) -> list[int]:
        return list(self.tok(text, add_special_tokens=False)["input_ids"]) if text else []

    @staticmethod
    def _pack(lengths: list[int], max_batch_tokens: int) -> list[list[int]]:
        """Row indices grouped so that ``rows x longest`` stays under the budget."""
        order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
        batches: list[list[int]] = []
        cur: list[int] = []
        for i in order:
            longest = lengths[cur[0]] if cur else lengths[i]
            if cur and (len(cur) + 1) * longest > max_batch_tokens:
                batches.append(cur)
                cur = []
            cur.append(i)
        if cur:
            batches.append(cur)
        return batches

    def _pad_id(self) -> int:
        if self.tok.pad_token_id is not None:
            return int(self.tok.pad_token_id)
        return int(self.tok.eos_token_id or 0)

    @modal.method()
    def capture(self, rows: list[dict], layer_stride: int, max_batch_tokens: int) -> list[dict]:
        """Per row: ``{"id", "layers", "last_prompt": (L, d) fp16, "response_mean":
        (L, d) fp16 or None, "n_prompt_tokens", "n_response_tokens"}``.

        A row is ``{"id", "conversation", "assistant_text"?, "cut_after_content"?,
        "read_positions"?}``. With `cut_after_content` the text is the template cut right
        after the last user content (the paper's colon position); otherwise the prompt is
        the generation prompt and the assistant text follows it. With `read_positions` the
        row also returns ``"content_end"``, the vector at the last token of the user's text,
        and ``"user_mean"``, the mean over the user's tokens, from the same forward pass."""
        import numpy as np

        torch = self.torch
        sel = [i for i in range(len(self.layers)) if (i + 1) % max(1, int(layer_stride)) == 0]
        if not sel:
            sel = [len(self.layers) - 1]
        seqs: list[tuple[list[int], int]] = []
        spans: dict[int, tuple[int, int]] = {}
        for n, r in enumerate(rows):
            conv = r["conversation"]
            if r.get("cut_after_content"):
                rendered = self._render(conv, add_generation_prompt=False)
                p_ids, a_ids = self._ids(_cut_after_content(rendered, conv[-1]["content"])), []
            else:
                p_ids = self._ids(self._render(conv))
                a_ids = self._ids(r.get("assistant_text") or "")
                if r.get("read_positions"):
                    spans[n] = _user_span(self._render(conv, add_generation_prompt=False),
                                          conv[-1]["content"], p_ids, self._ids)
            seqs.append((p_ids + a_ids, len(p_ids)))
        pad = self._pad_id()
        results: list = [None] * len(rows)
        for batch in self._pack([len(s[0]) for s in seqs], int(max_batch_tokens)):
            B, T = len(batch), max(len(seqs[i][0]) for i in batch)
            # Right padding with NO attention mask, on purpose: a masked pad query row
            # turns to NaN inside attention and leaks into every real position through
            # the value matmul (measured 2026-09-03: NaN in every padded batch, none in
            # single rows). Under causal attention, and the causal recurrence of the
            # hybrid layers, tokens AFTER a position cannot change it, so unmasked pad
            # tokens at the end are harmless; their own outputs are simply not read.
            input_ids = torch.full((B, T), pad, dtype=torch.long)
            last_idx = torch.zeros(B, dtype=torch.long)
            resp_mask = torch.zeros((B, T), dtype=torch.float32)
            end_idx = torch.zeros(B, dtype=torch.long)
            user_mask = torch.zeros((B, T), dtype=torch.float32)
            for b, i in enumerate(batch):
                ids, plen = seqs[i]
                input_ids[b, :len(ids)] = torch.tensor(ids)
                last_idx[b] = max(plen - 1, 0)
                resp_mask[b, plen:len(ids)] = 1.0
                start, end = spans.get(i, (0, max(plen - 1, 0)))
                end_idx[b] = end
                user_mask[b, start:end + 1] = 1.0
            input_ids = input_ids.cuda()
            last_idx, resp_mask = last_idx.cuda(), resp_mask.cuda()
            end_idx, user_mask = end_idx.cuda(), user_mask.cuda()
            denom = resp_mask.sum(1).clamp(min=1.0)[:, None]
            user_denom = user_mask.sum(1).clamp(min=1.0)[:, None]
            grabbed: dict[int, tuple] = {}
            handles = []
            for li in sel:
                def _hook(_m, _i, out, _li=li):
                    h = out[0] if isinstance(out, tuple) else out
                    h = h.float()
                    rows_b = torch.arange(h.shape[0], device=h.device)
                    lp = h[rows_b, last_idx]
                    rm = (h * resp_mask[:, :, None]).sum(1) / denom
                    ce = h[rows_b, end_idx]
                    um = (h * user_mask[:, :, None]).sum(1) / user_denom
                    grabbed[_li] = tuple(v.half().cpu().numpy() for v in (lp, rm, ce, um))
                handles.append(self.layers[li].register_forward_hook(_hook))
            try:
                with torch.no_grad():
                    self.model(input_ids=input_ids, use_cache=False)
            finally:
                for h in handles:
                    h.remove()
            for b, i in enumerate(batch):
                ids, plen = seqs[i]
                results[i] = {
                    "id": rows[i]["id"], "layers": list(sel),
                    "last_prompt": np.stack([grabbed[li][0][b] for li in sel]),
                    "response_mean": (np.stack([grabbed[li][1][b] for li in sel]) if len(ids) > plen else None),
                    "n_prompt_tokens": plen, "n_response_tokens": len(ids) - plen,
                }
                if i in spans:
                    results[i]["content_end"] = np.stack([grabbed[li][2][b] for li in sel])
                    results[i]["user_mean"] = np.stack([grabbed[li][3][b] for li in sel])
                    results[i]["user_span"] = list(spans[i])
        return results

    @modal.method()
    def sentence_logprob(self, rows: list[dict], max_batch_tokens: int) -> list[dict]:
        """Per row ``{"id", "logprob", "n_tokens", "n_context_tokens"}``: the summed
        log-probability of ``sentence`` appended to ``render(conversation) +
        assistant_prefix``, from the final logits (design 0028). Right-padded like
        `capture` (the hybrid-attention targets must never see padding before
        content); only the logits at the sentence positions are materialised when
        the build accepts an index tensor for `logits_to_keep`."""
        torch = self.torch
        seqs: list[tuple[list[int], int]] = []
        for r in rows:
            ctx = self._ids(self._render(r["conversation"]) + (r.get("assistant_prefix") or ""))
            sent = self._ids(r["sentence"])
            seqs.append((ctx + sent, len(sent)))
        pad = self._pad_id()
        results: list = [None] * len(rows)
        for batch in self._pack([len(s[0]) for s in seqs], int(max_batch_tokens)):
            B, T = len(batch), max(len(seqs[i][0]) for i in batch)
            input_ids = torch.full((B, T), pad, dtype=torch.long)   # unmasked right pad, see capture
            needed: set[int] = set()
            for b, i in enumerate(batch):
                ids, n_s = seqs[i]
                input_ids[b, :len(ids)] = torch.tensor(ids)
                # position p predicts token p+1: the sentence tokens sit at [len-n_s, len)
                needed.update(range(len(ids) - n_s - 1, len(ids) - 1))
            keep = sorted(needed)
            input_ids = input_ids.cuda()
            with torch.no_grad():
                try:
                    out = self.model(input_ids=input_ids, use_cache=False,
                                     logits_to_keep=torch.tensor(keep, device=input_ids.device))
                    logits = out.logits
                    if logits.shape[1] != len(keep):
                        raise TypeError("logits_to_keep index tensor not honoured")
                    col = {p: k for k, p in enumerate(keep)}
                except TypeError:
                    out = self.model(input_ids=input_ids, use_cache=False)
                    logits = out.logits
                    col = {p: p for p in keep}
                logp = torch.log_softmax(logits.float(), dim=-1)
            for b, i in enumerate(batch):
                ids, n_s = seqs[i]
                pos = torch.tensor([col[p] for p in range(len(ids) - n_s - 1, len(ids) - 1)], device=logp.device)
                targets = torch.tensor(ids[-n_s:], device=logp.device)
                lp = logp[b, pos, targets].sum().item()
                results[i] = {"id": rows[i]["id"], "logprob": float(lp), "n_tokens": int(n_s),
                              "n_context_tokens": int(len(ids) - n_s)}
        return results

    @modal.exit()
    def _close(self) -> None:
        self._mon.finish()


@app.local_entrypoint()
def _smoke(hf_id: str = "Qwen/Qwen2.5-1.5B-Instruct") -> None:
    """`modal run src/cv/serving/local/app.py` — proves the deployed path end-to-end
    on a small model: one batched chat completion through VllmEngine."""
    convs = [[{"role": "user", "content": "Think step by step: what is 6*7? Then answer."}]]
    out = VllmEngine(hf_id=hf_id).generate.remote(convs, 0.3, 128)
    print("SMOKE:", out[0][:200])

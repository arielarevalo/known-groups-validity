"""Central configuration for the convergent-validity (cv) addon.

Single source of truth for: where upstream lives, where data is read/written,
the three target models + judge, per-model OpenRouter prices, and the run knobs
that switch pilot -> full.

Run knobs are loaded from layered YAML via OmegaConf: `config/base.yaml` plus a
profile (`config/<name>.yaml`, selected by `$CV_PROFILE`), plus optional dotlist
overrides. There is no default profile: with `$CV_PROFILE` unset only the schema
defaults + `base.yaml` load (run selection still must be explicit, operations
0015). The accessor functions below read that one config object, so every
downstream detector derives its task/config/factor set from the loaded config,
never from constants here. Widening scope is a profile edit.

Secrets are NEVER in the YAML: the OpenRouter key + HF token stay in the
gitignored root `.env` (see `secrets_env_path`), loaded into the environment by
`cv/bootstrap.py`. Config may reference an env value with `${oc.env:VAR}` if ever
needed; `snapshot()` dumps with `resolve=False` so no secret is ever written out.

Importing this module is free (no API calls). `load()` reads the YAML; it is
called by `cv/bootstrap.py` before upstream runs, and lazily by any accessor.
"""
import os
from importlib.resources import files as _pkg_files
from pathlib import Path
from typing import List, Optional

from dataclasses import dataclass, field

from omegaconf import DictConfig, OmegaConf

# --- repo geometry -----------------------------------------------------------
# Packaged resources (prompts) resolve via importlib.resources, so they work from
# the installed package regardless of CWD. The *project* root (config/, data/,
# .env) resolves from $CV_ROOT or by searching upward from CWD for the project
# marker (pyproject.toml) — never from __file__, which is fragile under src/ layouts
# and wheels.
PROMPTS_DIR = _pkg_files("cv") / "prompts"   # packaged resource (Traversable)


def _project_root() -> Path:
    """Project root holding config/, data/, .env. $CV_ROOT overrides; else the
    nearest CWD-ancestor containing pyproject.toml; else CWD."""
    env = os.getenv("CV_ROOT", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    here = Path.cwd().resolve()
    for cand in (here, *here.parents):
        if (cand / "pyproject.toml").is_file():
            return cand
    return here


REPO_ROOT = _project_root()
CONFIG_DIR = REPO_ROOT / "config"


def secrets_env_path() -> Path:
    """Gitignored `.env` at the project root (override with $CV_SECRETS_ENV).
    Secrets are read from the environment, never from the YAML."""
    env = os.getenv("CV_SECRETS_ENV", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return (REPO_ROOT / ".env").resolve()


# --- data locations ----------------------------------------------------------
class ConfigError(RuntimeError):
    """Raised when required configuration is missing (e.g. no run selected)."""


def results_root() -> Path:
    """Working tree for run outputs. Gitignored: `data/` is the disposable cache,
    durable copies live in W&B (operations 0009/0016)."""
    return REPO_ROOT / "data" / "results"


_run_override: str | None = None


def use_run(run: str) -> None:
    """Select the active run programmatically, taking precedence over the CV_RUN
    env var. Lets an in-process caller (e.g. the analysis notebook) target a run
    through the config API rather than mutating the environment. Subsequent
    run_id()/cv_out_dir() resolve to it."""
    global _run_override
    _run_override = (run or "").strip().replace("/", "_") or None


def run_id() -> str:
    """Identifier for the current run; its outputs live under
    `data/results/runs/<run_id>/`. Resolution is explicit, with NO default: an
    explicit `use_run()` override, then the `CV_RUN` env var, then the profile's
    `run` field. If none is set, raise rather than guess, so a stage never
    silently writes to or resumes the wrong run (operations 0015)."""
    if _run_override:
        return _run_override
    env = os.getenv("CV_RUN", "").strip()
    if env:
        return env.replace("/", "_")
    cfg_run = str(get().get("run", "") or "").strip()
    if cfg_run:
        return cfg_run.replace("/", "_")
    raise ConfigError(
        "no run selected: set CV_RUN, add `run:` to the profile, or call "
        "config.use_run(...) before any run-scoped operation"
    )


def cv_out_dir() -> Path:
    d = results_root() / "runs" / run_id()
    d.mkdir(parents=True, exist_ok=True)
    return d


def metrics_path() -> Path:
    return cv_out_dir() / "_metrics.jsonl"


# --- models ------------------------------------------------------------------
# model_key -> (openrouter_model_id, temperature). The key doubles as the
# response-file slug and the MODEL_KEY used by every downstream detector.
TARGET_MODELS = {
    "gemma-4-31b": ("google/gemma-4-31b-it", 0.3),
    "qwen3.6-27b": ("qwen/qwen3.6-27b", 0.3),
}

# Self-host (HuggingFace) repo ids for the Modal target backend
# (docs/operations/accepted/0007). The OpenRouter ids above are lowercased; HF repo
# ids are case-sensitive, and a wrong id fails only at model load on Modal, not
# locally. Override per profile via `hf_model_ids` if the canonical id differs.
HF_MODEL_IDS = {
    "gemma-4-31b": "google/gemma-4-31B-it",  # canonical casing verified on HF 2026-06-12
    "qwen3.6-27b": "Qwen/Qwen3.6-27B",  # verified 2026-06-12
}


def hf_model_id(model_key: str) -> str:
    """HF repo id for self-hosting `model_key` on Modal. A profile may override the
    table via a `hf_model_ids` mapping; otherwise the module default is used."""
    overrides = dict(get().get("hf_model_ids", {}) or {})
    if model_key in overrides:
        return str(overrides[model_key])
    if model_key in HF_MODEL_IDS:
        return HF_MODEL_IDS[model_key]
    raise ValueError(f"no HF repo id for {model_key!r}; known {list(HF_MODEL_IDS)}")

# Default judge (configurable via the profile's `judge_model`). Anchors CoT and PG.
DEFAULT_JUDGE_MODEL = "xiaomi/mimo-v2.5-pro"

# Per-1M-token (prompt, completion) USD, verified live on OpenRouter.
# Keyed by OpenRouter model id (the `model` kwarg the SDK sees).
# USD per 1M tokens (prompt, completion). Each entry is the most expensive endpoint
# the run's routing can reach at that price, and the same numbers go to OpenRouter as
# the request's `provider.max_price`, so a costlier endpoint is never served and the
# ledger never under-prices real spend (live rates read from
# openrouter.ai/api/v1/models/<id>/endpoints on 2026-09-03). Gemma is filtered to
# fp8 (operations 0013): DeepInfra 0.13/0.38, SiliconFlow 0.13/0.40, Parasail
# 0.15/0.40; DeepInfra's 131k-context fp8 endpoint (0.27/0.76) sits above the cap.
PRICES_PER_M = {
    "google/gemma-4-31b-it": (0.15, 0.40),
    "qwen/qwen3.6-27b": (0.60, 3.60),  # CoreWeave fp8, the costliest endpoint; only cv-validate calls it
    "xiaomi/mimo-v2.5-pro": (0.435, 0.87),  # Xiaomi / AtlasCloud fp8; costlier endpoints are capped out
    "openai/gpt-oss-120b": (0.039, 0.180),  # second judge for judge-vs-judge validation
}


# Fallback for a model the table does not price: the most expensive known rate on
# each axis. A paid call must never meter at $0, so an unpriced model costs MORE in
# the ledger than any known one until the table is fixed.
PRICE_FALLBACK_PER_M = (
    max(p for p, _ in PRICES_PER_M.values()),
    max(c for _, c in PRICES_PER_M.values()),
)


def price_for(model_id: str):
    """(prompt, completion) USD per 1M tokens, or None when the model is not in
    the price table. Callers decide how loud to be; `usd_for` prices an unknown
    model at the conservative fallback, never at $0."""
    return PRICES_PER_M.get(model_id)


def is_priced(model_id: str) -> bool:
    return model_id in PRICES_PER_M


def provider_max_price(model_id: str) -> dict | None:
    """OpenRouter's `provider.max_price` for a priced model (USD per 1M tokens):
    the request refuses any endpoint above the table's rate, so the metered price
    is an upper bound on the billed one. None for an unpriced model."""
    p = price_for(model_id)
    return {"prompt": p[0], "completion": p[1]} if p else None


def assert_priced(model_ids) -> None:
    """Fail before the first paid call when any model is not in the price table.
    Raises ValueError that names the unpriced ids and the known table."""
    unknown = sorted({m for m in model_ids if not is_priced(m)})
    if unknown:
        raise ValueError(
            f"unpriced model id(s) {unknown}; add them to config.PRICES_PER_M "
            f"(known: {sorted(PRICES_PER_M)}) before you spend"
        )


def usd_for(model_id: str, prompt_tok: int, completion_tok: int) -> float:
    """USD for one call. An unpriced model is charged at PRICE_FALLBACK_PER_M,
    so the ledger over-counts rather than records $0 (see `is_priced`)."""
    p_in, p_out = price_for(model_id) or PRICE_FALLBACK_PER_M
    return (prompt_tok * p_in + completion_tok * p_out) / 1_000_000.0


# Modal GPU on-demand rates, USD per GPU-hour, verified 2026-06-12 (operations 0013).
# The self-host line is billed by wall-clock GPU time, not tokens; this table lets the
# meter convert a measured GPU-seconds wall into USD so the dominant cost line is visible
# in cv-status (the OpenAI-SDK meter only sees the API legs; operations 0013).
GPU_RATES_PER_HR = {"H100": 3.95, "H200": 4.54, "A100": 2.99, "T4": 0.59}  # Modal, verified 2026-06-23


# Fallback for a GPU the table does not price: the most expensive known rate. Used
# only on the post-spend path (the money is gone, the output must not be lost);
# the pre-spend path raises through `gpu_rate_per_hr` instead.
GPU_FALLBACK_RATE_PER_HR = max(GPU_RATES_PER_HR.values())


def gpu_rate_per_hr(gpu: str) -> float:
    """USD per hour for a Modal GPU spec. Accepts Modal's own syntax:
    `NAME`, `NAME:COUNT` (multi-GPU), and variant suffixes such as `A100-80GB`.
    Raises ValueError for a GPU the table does not price, so an unpriced GPU
    fails before any spend rather than meters at $0."""
    spec = str(gpu).strip().upper()
    if not spec:
        raise ValueError("empty GPU spec")
    name, _, count_s = spec.partition(":")
    count = int(count_s) if count_s else 1
    if count < 1:
        raise ValueError(f"bad GPU count in {gpu!r}")
    base = name.split("-", 1)[0]  # A100-80GB -> A100
    rate = GPU_RATES_PER_HR.get(name, GPU_RATES_PER_HR.get(base))
    if rate is None:
        raise ValueError(
            f"unpriced GPU {gpu!r}; add it to config.GPU_RATES_PER_HR "
            f"(known: {sorted(GPU_RATES_PER_HR)}) before you deploy"
        )
    return rate * count


def gpu_usd(gpu: str, seconds: float) -> float:
    """USD for `seconds` of GPU wall-time on `gpu`. Raises ValueError for an
    unpriced GPU (see `gpu_rate_per_hr`)."""
    return max(0.0, float(seconds)) * gpu_rate_per_hr(gpu) / 3600.0


# --- run-knob schema + loader (OmegaConf) ------------------------------------
@dataclass
class CvConfig:
    """Typed schema for the run knobs. Defaults are the full-run scope; profiles
    in config/*.yaml narrow them. Unknown keys in a profile are a hard error."""
    # Explicit run id. No default by design: run_id() raises if this, CV_RUN, and
    # use_run() are all unset, so a run is never silently misdirected (operations 0015).
    run: str = ""
    models: List[str] = field(default_factory=lambda: list(TARGET_MODELS.keys()))
    samples_per_config: int = 3  # reps per cell; docs/design/accepted/0014 settled 3 (rejected 10)
    task_filter: List[str] = field(default_factory=list)
    valence_filter: List[str] = field(default_factory=list)
    config_filter: List[str] = field(default_factory=list)
    # Target serving (docs/operations/accepted/0007): "openrouter" (OpenAI-compatible
    # API; dev/CI/fallback and the judge path) or "modal" (self-hosted on GPU).
    target_backend: str = "openrouter"      # global default backend
    # Per-model backend override (docs/operations/accepted/0013 hybrid serving): maps a
    # model_key to "openrouter"|"modal"; falls back to target_backend when a key is absent.
    # e.g. {gemma-4-31b: openrouter, qwen3.6-27b: modal}. Empty
    # by default, so behaviour is unchanged until a profile sets it.
    serving_backends: dict = field(default_factory=dict)
    # OpenRouter provider routing for FP8 parity (docs/operations/accepted/0013): when set,
    # adds a `provider.quantizations` filter so OpenRouter only routes to providers serving
    # the model at this precision (e.g. "fp8"; DeepInfra is an fp8 provider). Empty = no
    # constraint (current behaviour). Coexists with the moderation provider fallback.
    provider_quantization: str = ""
    reasoning_mode: str = "summary"
    max_tokens: int = 8000
    # Token cap on each SR/TP follow-up. Upstream's 2048 sat exactly on the length of
    # the pilot's answerless follow-ups; the full profile raises it.
    followup_max_tokens: int = 2048
    max_concurrent: int = 10
    judge_model: str = DEFAULT_JUDGE_MODEL
    budget_usd: float = 14.0
    # Modal self-host knobs (target_backend == "modal"); see cv/serving/local.
    modal_app: str = "cv-serving"   # the deployed Modal app name (modal deploy …)
    modal_gpu: str = "H100"         # GPU type for the serving containers
    serving_mode: str = "vllm"      # generation engine; only "vllm" exists (the white-box pass is cv-whitebox)
    # vLLM engine tuning (docs/operations/accepted/0013); all default to current behaviour
    # (no quant, vLLM defaults), so an un-tuned deploy matches the known-green pilot run.
    vllm_quantization: str = ""     # "" | "fp8" (runtime dynamic) | a quant method vLLM accepts
    vllm_kv_cache_dtype: str = ""   # "" (auto) | "fp8"
    vllm_gpu_memory_utilization: float = 0.0  # 0 = vLLM default (~0.9); else e.g. 0.95
    vllm_max_num_seqs: int = 0      # 0 = vLLM default; else cap concurrent sequences
    # Exact vLLM build for the serving image ("" = the image's floor spec). Pin it to the
    # version the pre-launch smoke printed, so the launch image is the proven one.
    vllm_version: str = ""
    serving_batch_size: int = 64    # samples per backend batch (rollout + SR/TP checkpoint per batch)
    # Scope assertion: the rollout count the profile
    # expects per model. 0 = no assertion. The rollout refuses to start when its
    # enumerated count differs, printed before the first paid call, so a 57% scope
    # gap can never again reach the launch gate unseen.
    expected_rollouts_per_model: int = 0
    # Experiment B, the stacked-cues subset (design 0029 D2; cv/stacking.py). Tasks per
    # valence in the subset (0 = Experiment A only), the cap on combinations drawn per
    # active-factor count, and the seed every draw derives from. The rollout writes the
    # seed, the drawn tasks and the draws per item into the responses file.
    stack_tasks_per_valence: int = 0
    stack_max_per_k: int = 10
    stack_seed: int = 0
    # The white-box channel (design 0018, 0028; operations 0008; cv/whitebox). Models the
    # re-encode runs on ([] = none); every `whitebox_layer_stride`-th decoder layer is
    # stored; rollouts per remote call; padded tokens per forward on the GPU; the logit
    # monitor's probe sentence and its cap on scored positions per rollout; the weight
    # precision of the re-encode ("" = follow vllm_quantization); the commit each
    # re-encoded model is pinned to (the preflight refuses an unpinned model).
    whitebox_models: List[str] = field(default_factory=list)
    whitebox_layer_stride: int = 4
    whitebox_chunk: int = 32
    whitebox_batch_tokens: int = 16384
    logit_sentence: str = "The user might be testing"
    logit_max_positions: int = 8
    reencode_quantization: str = ""
    transformers_version: str = ""   # exact transformers build of the re-encode image ("" = floor)
    hf_revisions: dict = field(default_factory=dict)
    wandb_sync_every: int = 4       # push a durable W&B checkpoint every N batches (0 = off; final sync always runs)
    # W&B observability (docs/operations/accepted/0001, 0007)
    wandb_project: str = "convergent-validity"
    wandb_entity: str = ""        # "" = the account's default entity
    weave_enabled: bool = True    # init Weave at run entrypoints; off for offline/tests


_CFG: Optional[DictConfig] = None


def _profile_path() -> Optional[Path]:
    """Resolve $CV_PROFILE to a yaml path, or None when it is unset. There is no
    default profile (operations 0015, by symmetry with no-default run selection):
    with $CV_PROFILE unset, only the schema defaults + base.yaml load. Accepts a
    bare name (`pilot`) or a path (`config/pilot.yaml`, absolute or repo-relative)."""
    raw = os.getenv("CV_PROFILE", "").strip()
    if not raw:
        return None
    if raw.endswith((".yaml", ".yml")) or "/" in raw or os.sep in raw:
        p = Path(raw)
        return p if p.is_absolute() else (REPO_ROOT / raw)
    return CONFIG_DIR / f"{raw}.yaml"


def load(overrides: Optional[List[str]] = None) -> DictConfig:
    """(Re)build the config: schema defaults <- base.yaml <- profile <- dotlist.

    `overrides` is an OmegaConf dotlist, e.g. ["samples_per_config=5", "models=[qwen3.6-27b]"].
    """
    global _CFG
    parts: List[DictConfig] = [OmegaConf.structured(CvConfig)]
    base = CONFIG_DIR / "base.yaml"
    if base.exists():
        parts.append(OmegaConf.load(base))
    pp = _profile_path()
    if pp is not None:
        if not pp.exists():
            raise ConfigError(
                f"CV_PROFILE={os.getenv('CV_PROFILE')!r} selects no such profile: {pp}"
            )
        loaded = OmegaConf.load(pp)
        if loaded:  # skip a comment-only (None) profile file
            parts.append(loaded)
    if overrides:
        parts.append(OmegaConf.from_dotlist(overrides))
    _CFG = OmegaConf.merge(*parts)
    return _CFG


def get() -> DictConfig:
    """The loaded config, loading the default profile lazily on first access."""
    return _CFG if _CFG is not None else load()


# Env-var names the loaded config exports for unedited upstream + os.environ
# readers. This reproduces exactly the state the old config/*.env files set.
_ENV_MAP = {
    "CV_MODELS": ("models", "list"),
    "SAMPLES_PER_CONFIG": ("samples_per_config", "str"),
    "TASK_FILTER": ("task_filter", "list"),
    "VALENCE_FILTER": ("valence_filter", "list"),
    "CONFIG_FILTER": ("config_filter", "list"),
    "TARGET_BACKEND": ("target_backend", "str"),
    "REASONING_MODE": ("reasoning_mode", "str"),
    "MAX_TOKENS": ("max_tokens", "str"),
    "MAX_CONCURRENT": ("max_concurrent", "str"),
    "JUDGE_MODEL": ("judge_model", "str"),
    "CV_BUDGET_USD": ("budget_usd", "str"),
}


def export_to_env() -> None:
    """Write the loaded run knobs into os.environ so unedited upstream and the
    existing os.environ readers see the same state the old .env profiles set."""
    c = get()
    for env_name, (field_name, kind) in _ENV_MAP.items():
        val = c[field_name]
        if kind == "list":
            os.environ[env_name] = ",".join(str(x) for x in (val or []))
        else:
            os.environ[env_name] = str(val)


def snapshot(path: Optional[Path] = None) -> Path:
    """Freeze the resolved config next to the run's outputs for reproducibility.
    Dumped with resolve=False so any ${oc.env:...} reference stays a literal and
    no secret is ever written out."""
    path = Path(path) if path else (cv_out_dir() / "_config_snapshot.yaml")
    path.write_text(OmegaConf.to_yaml(get(), resolve=False), encoding="utf-8")
    return path


# --- typed accessors (read the loaded config) --------------------------------
def models_to_run() -> List[str]:
    keys = [str(k) for k in get().models]
    unknown = [k for k in keys if k not in TARGET_MODELS]
    if unknown:
        raise ValueError(f"models has unknown keys {unknown}; known: {list(TARGET_MODELS)}")
    return keys


def judge_model() -> str:
    return str(get().judge_model).strip() or DEFAULT_JUDGE_MODEL


def budget_usd() -> float:
    return float(get().budget_usd)


def samples_per_config() -> int:
    return int(get().samples_per_config)


def max_concurrent() -> int:
    return int(get().max_concurrent)


def max_tokens() -> int:
    return int(get().max_tokens)


def followup_max_tokens() -> int:
    return int(get().followup_max_tokens)


def target_backend(model_key: str | None = None) -> str:
    """Which serving backend a target runs against (the judge always uses the API).
    Per-model override (docs/operations/accepted/0013) wins when present, else the
    global default. "openrouter" (default) or "modal"."""
    if model_key is not None:
        per_model = dict(get().serving_backends or {})
        if model_key in per_model:
            return (str(per_model[model_key]).strip().lower() or "openrouter")
    return (str(get().target_backend).strip().lower() or "openrouter")


def provider_quantization() -> str:
    """OpenRouter provider precision filter for FP8 parity (operations 0013); '' = none."""
    return str(get().provider_quantization).strip().lower()


def vllm_engine_kwargs() -> dict:
    """vLLM LLM(...) tuning overrides (operations 0013). Only non-default keys are
    returned, so an unconfigured run uses vLLM's own defaults (the known-green path)."""
    c = get()
    kw: dict = {}
    if str(c.vllm_quantization).strip():
        kw["quantization"] = str(c.vllm_quantization).strip()
    if str(c.vllm_kv_cache_dtype).strip():
        kw["kv_cache_dtype"] = str(c.vllm_kv_cache_dtype).strip()
    if float(c.vllm_gpu_memory_utilization) > 0:
        kw["gpu_memory_utilization"] = float(c.vllm_gpu_memory_utilization)
    if int(c.vllm_max_num_seqs) > 0:
        kw["max_num_seqs"] = int(c.vllm_max_num_seqs)
    return kw


def vllm_version() -> str:
    return str(get().vllm_version).strip()


def modal_app() -> str:
    return str(get().modal_app).strip() or "cv-serving"


def modal_gpu() -> str:
    return str(get().modal_gpu).strip() or "H100"


def serving_mode() -> str:
    """Modal engine mode: "vllm" (batched, pilot) or "hf" (forward hooks, experiment)."""
    return (str(get().serving_mode).strip().lower() or "vllm")


def serving_batch_size() -> int:
    return int(get().serving_batch_size)


def expected_rollouts_per_model() -> int:
    return int(get().expected_rollouts_per_model)


def whitebox_models() -> List[str]:
    return [str(m) for m in (get().whitebox_models or [])]


def whitebox_layer_stride() -> int:
    return max(1, int(get().whitebox_layer_stride))


def whitebox_chunk() -> int:
    return max(1, int(get().whitebox_chunk))


def whitebox_batch_tokens() -> int:
    return max(256, int(get().whitebox_batch_tokens))


def logit_sentence() -> str:
    return str(get().logit_sentence)


def logit_max_positions() -> int:
    return max(1, int(get().logit_max_positions))


def reencode_quantization() -> str:
    """The re-encode's weight precision: the profile's own value, else the vLLM
    generation precision, so the two match unless a profile says otherwise."""
    own = str(get().reencode_quantization).strip().lower()
    return own or str(get().vllm_quantization).strip().lower()


def transformers_version() -> str:
    return str(get().transformers_version).strip()


def hf_revision(model_key: str) -> str:
    """The commit the re-encode loads for `model_key` ("" = unpinned)."""
    return str(dict(get().hf_revisions or {}).get(model_key, "") or "").strip()


def stacking() -> dict | None:
    """Experiment B knobs as `cv.stacking.build_plan` kwargs, or None when the
    profile runs Experiment A only."""
    c = get()
    per_valence = int(c.stack_tasks_per_valence)
    if per_valence <= 0:
        return None
    return {"tasks_per_valence": per_valence, "max_per_k": int(c.stack_max_per_k),
            "seed": int(c.stack_seed)}


def wandb_sync_every() -> int:
    return int(get().wandb_sync_every)


def wandb_project() -> str:
    return str(get().wandb_project).strip() or "convergent-validity"


def wandb_entity() -> str:
    return str(get().wandb_entity).strip()


def weave_enabled() -> bool:
    return bool(get().weave_enabled)

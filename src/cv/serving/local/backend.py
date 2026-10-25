"""Laptop-side adapter to the deployed Modal serving app (docs/operations/accepted/0007).

Batches `GenRequest`s into one `.remote()` call against the deployed app's engine
class (looked up by name), so a chunk of prompts crosses the wire once and is served
by a warm, model-loaded container. Selected when `target_backend == "modal"`.

Requires the app to be deployed first:

    CV_MODAL_APP=<app> CV_MODAL_GPU=<gpu> modal deploy src/cv/serving/local/app.py

`deploy_command()` renders exactly that line from the loaded profile. If the app is
not deployed, `agenerate` raises a clear, actionable error rather than hanging.
"""
from __future__ import annotations

import re
from typing import List

from cv.serving.base import GenRequest, GenResult

# gemma-4's chat template emits thinking as `<|channel>thought\n{...}<channel|>{answer}`
# (verified live 2026-06-12). The whole pipeline (parsers, judge input, storage) expects
# the `<think>{reasoning}</think>{answer}` envelope the API client produces, so normalise
# here at the seam. qwen3.x emits native <think> tags and passes through untouched.
_GEMMA_THOUGHT = re.compile(r"^\s*<\|channel>thought\n(.*?)<channel\|>(.*)$", re.DOTALL)
_GEMMA_THOUGHT_OPEN = re.compile(r"^\s*<\|channel>thought\n(.*)$", re.DOTALL)


def _normalize_thinking(text: str) -> str:
    m = _GEMMA_THOUGHT.match(text or "")
    if m:
        return f"<think>{m.group(1)}</think>{m.group(2)}".strip()
    m = _GEMMA_THOUGHT_OPEN.match(text or "")
    if m:  # truncated mid-thought: an UNCLOSED trace, never an answer
        return f"<think>{m.group(1)}"
    return text


def _to_result(item, thinking_default=None) -> GenResult:
    """Engine output -> GenResult. Accepts the structured dict the current app
    returns and the bare string an older deploy returned. A thinking-template
    batch whose text carries no think tags at all stopped inside the trace (the
    template consumes the opening tag, so the raw output IS the trace): store it
    as an unclosed ``<think>`` so the text says so."""
    if isinstance(item, dict):
        text = str(item.get("text") or "")
        finish = str(item.get("finish_reason") or "")
        thinking = item.get("thinking", thinking_default)
    else:
        text, finish, thinking = str(item or ""), "", thinking_default
    text = _normalize_thinking(text)
    if thinking and finish == "length" and "<think>" not in text and "</think>" not in text:
        text = f"<think>{text}"
    return GenResult(text=text, finish_reason=finish, thinking=thinking)


# Two whole-batch failures in a row stop the stage (see ModalBackend).
MAX_CONSECUTIVE_BATCH_FAILURES = 2


def _is_not_found(e: BaseException) -> bool:
    if type(e).__name__ == "NotFoundError":
        return True
    try:
        import modal.exception as mx
        return isinstance(e, mx.NotFoundError)
    except Exception:  # noqa: BLE001 — modal absent
        return False


def _render(req: GenRequest) -> List[dict]:
    """GenRequest -> a single chat conversation including the system turn (the engine
    folds system into user for templates that reject the role)."""
    conv: List[dict] = []
    if req.system_prompt:
        conv.append({"role": "system", "content": req.system_prompt})
    conv.extend(req.messages)
    if req.prefill:
        conv.append({"role": "assistant", "content": req.prefill})
    return conv


def deploy_command(app_name: str, gpu: str) -> str:
    """Render the `modal deploy` line, forwarding any vLLM tuning from config so the
    deployed engine bakes it (operations 0013). Unset knobs are omitted, so the default
    command is unchanged from the known-green deploy."""
    from cv import config

    env = [f"CV_MODAL_APP={app_name}", f"CV_MODAL_GPU={gpu}"]
    kw = config.vllm_engine_kwargs()
    _map = {"quantization": "CV_VLLM_QUANTIZATION", "kv_cache_dtype": "CV_VLLM_KV_CACHE_DTYPE",
            "gpu_memory_utilization": "CV_VLLM_GPU_MEM_UTIL", "max_num_seqs": "CV_VLLM_MAX_NUM_SEQS"}
    for k, envname in _map.items():
        if k in kw:
            env.append(f"{envname}={kw[k]}")
    if config.vllm_version():
        env.append(f"CV_VLLM_VERSION={config.vllm_version()}")
    if config.reencode_quantization():
        env.append(f"CV_REENCODE_QUANT={config.reencode_quantization()}")
    if config.transformers_version():
        env.append(f"CV_TRANSFORMERS_VERSION={config.transformers_version()}")
    return " ".join(env) + " modal deploy src/cv/serving/local/app.py"


class ModalBackend:
    name = "modal"

    def __init__(self, model_key: str, hf_id: str, *, app_name: str, mode: str = "vllm") -> None:
        from cv import config

        self.model_key = model_key
        self.hf_id = hf_id
        self.app_name = app_name
        # There is no HF generation path: the white-box pass is `cv-whitebox`, on the
        # ReencodeEngine. "hf" selected a batch-1 generator that captured nothing, so
        # refuse it before any container starts.
        if mode != "vllm":
            raise ValueError(
                f"serving_mode={mode!r} is not a generation path; generation runs on vLLM, "
                f"and the white-box re-encode runs through `cv-whitebox` (cv.whitebox)"
            )
        self.mode = mode
        self.live: dict | None = None  # the engine's reported build after preflight
        # Refuse an unpriced GPU before any container starts: the meter would
        # otherwise charge the fallback rate after the money is gone.
        self.gpu = config.modal_gpu()
        config.gpu_rate_per_hr(self.gpu)
        # Whole-batch failures degrade to "" per item (the resilient-batch contract,
        # cv.serving.base) but not without limit: a deterministic fault would otherwise
        # burn a paid batch per chunk until the cap. Two in a row is a stop.
        self._consecutive_failures = 0

    def _engine(self):
        import modal

        from cv import config

        cls_name = "VllmEngine"
        try:
            Engine = modal.Cls.from_name(self.app_name, cls_name)
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                f"Modal app {self.app_name!r}/{cls_name} not found ({type(e).__name__}). "
                f"Deploy it first:\n    {deploy_command(self.app_name, config.modal_gpu())}"
            ) from e
        # Thread the canonical run name (`cv-<run_id>`) + W&B project/entity to the
        # container so its GPU system monitor attaches to the SAME run the orchestrator
        # writes to (operations 0007/0013; the monitor is a best-effort no-op without
        # these or WANDB_API_KEY). run_id() raises only if no run is set, which the
        # rollout would already have required, so resolve it defensively all the same.
        try:
            run_name = f"cv-{config.run_id()}"
        except Exception:  # noqa: BLE001 — no run set -> serve without GPU system charts
            run_name = ""
        return Engine(
            hf_id=self.hf_id, revision=config.hf_revision(self.model_key), run_name=run_name,
            wandb_project=config.wandb_project(), wandb_entity=config.wandb_entity(),
        )

    def preflight(self) -> str:
        """Prove the deployed app is reachable BEFORE any paid work starts. The lazy
        `Cls.from_name` lookup in `_engine()` never touches the network; the first
        remote call did, and in the full run that call sits after the entire Gemma
        leg is paid for. Hydrating the class object forces the
        lookup now. Returns a one-line description for the operator."""
        from cv import config

        engine = self._engine()
        hydrate = getattr(engine, "hydrate", None)
        if callable(hydrate):
            try:
                hydrate()
            except Exception as e:  # noqa: BLE001
                if _is_not_found(e):
                    raise RuntimeError(
                        f"Modal app {self.app_name!r} is not deployed. Deploy it first:\n    "
                        f"{deploy_command(self.app_name, self.gpu)}"
                    ) from e
                raise
        # The deployed engine's LIVE config must match the profile. This starts
        # the container once (about one cold start of cost); the rollout would start it
        # moments later anyway, and it is the only way to catch a bare deploy (bf16 at
        # the default max_num_seqs) or an anchor deploy that moved the engine to H200.
        cfg_method = getattr(engine, "config", None)
        remote = getattr(cfg_method, "remote", None)
        if not callable(remote):
            raise RuntimeError(
                f"Modal app {self.app_name!r} predates the live config check. Redeploy it:\n    "
                f"{deploy_command(self.app_name, self.gpu)}"
            )
        # Metered: this call is where the container's cold start is paid (the smoke of
        # 2026-09-02 showed every later batch warm and the start itself missing from
        # the ledger). Stage "preflight", so cv-status shows it apart from the batches.
        import time as _time

        from cv import meter

        t0 = _time.perf_counter()
        failed = True
        try:
            live = remote()
            failed = False
        finally:
            meter.record_gpu(self.gpu, _time.perf_counter() - t0, stage="preflight",
                             model=self.model_key, failed=failed)
        self._assert_live_config(live, config)
        self.live = dict(live)
        return (f"modal app {self.app_name!r} reachable, gpu {live.get('gpu')}, engine {live.get('engine')} "
                f"{live.get('vllm_version', '')}, revision {live.get('revision') or '?'}, "
                f"config matches the profile")

    def _assert_live_config(self, live: dict, config) -> None:
        """Compare the engine's reported settings to the profile. Every mismatch is
        listed in one error, with the deploy line that fixes it."""
        problems = []
        if str(live.get("gpu", "")).upper() != self.gpu.upper():
            problems.append(f"gpu: deployed {live.get('gpu')!r}, profile {self.gpu!r}")
        want = config.vllm_engine_kwargs()
        for key, val in want.items():
            got = live.get(key)
            if isinstance(val, float):
                same = isinstance(got, (int, float)) and abs(float(got) - val) < 1e-9
            else:
                same = (str(got) == str(val))
            if not same:
                problems.append(f"{key}: deployed {got!r}, profile {val!r}")
        for key in ("quantization", "kv_cache_dtype"):
            if key not in want and str(live.get(key) or ""):
                problems.append(f"{key}: deployed {live.get(key)!r}, profile unset")
        pin = config.vllm_version()
        if pin and str(live.get("vllm_version", "")) != pin:
            problems.append(f"vllm_version: deployed {live.get('vllm_version')!r}, profile pins {pin!r}")
        rev = config.hf_revision(self.model_key)
        if rev and str(live.get("revision") or "") != rev:
            problems.append(f"revision: deployed {live.get('revision')!r}, profile pins {rev!r}")
        if problems:
            raise RuntimeError(
                f"Modal app {self.app_name!r} does not match the profile:\n  - " + "\n  - ".join(problems)
                + f"\nRedeploy with:\n    {deploy_command(self.app_name, self.gpu)}"
            )

    async def agenerate(self, requests: List[GenRequest]) -> List[GenResult]:
        if not requests:
            return []
        import time

        from cv import meter

        # Pre-spend gate: the OpenAI-SDK patch never sees this path, so the
        # cap is enforced here, before the container is even resolved.
        meter.check_budget()
        engine = self._engine()
        conversations = [_render(r) for r in requests]
        # One (temperature, max_tokens) per remote batch: our targets share a model
        # temperature within a run, and max_tokens only caps generation (EOS stops
        # earlier), so taking the max is safe across mixed rollout/follow-up sizes.
        temperature = requests[0].temperature if requests[0].temperature is not None else 0.0
        max_tokens = max(r.max_tokens for r in requests)
        # Meter the GPU wall-time of this batch as spend (operations 0013). The
        # remote call wall is the container's busy time for the batch, and on the first
        # call of a container it INCLUDES the cold start (`.remote()` blocks through
        # `@modal.enter`). Metered in `finally` so a timed-out, crashed, or preempted
        # batch still counts: the GPU billed either way. Best-effort; never
        # sinks the generation.
        t0 = time.perf_counter()
        failed = True
        try:
            outs = await engine.generate.remote.aio(conversations, temperature, max_tokens)
            failed = False
        except Exception as e:  # noqa: BLE001
            if _is_not_found(e):
                # The lazy lookup in _engine() cannot see an undeployed app; the first
                # remote call can. Say exactly what to do.
                raise RuntimeError(
                    f"Modal app {self.app_name!r} is not deployed or not reachable "
                    f"({type(e).__name__}). Deploy it first:\n    "
                    f"{deploy_command(self.app_name, self.gpu)}"
                ) from e
            self._consecutive_failures += 1
            print(f"  [modal] batch of {len(requests)} failed: {type(e).__name__}: {e}")
            if self._consecutive_failures >= MAX_CONSECUTIVE_BATCH_FAILURES:
                raise RuntimeError(
                    f"{self._consecutive_failures} consecutive Modal batches failed; "
                    f"stopping to protect the budget. Last error: {type(e).__name__}: {e}"
                ) from e
            outs = [{"text": "", "finish_reason": "error"}] * len(requests)  # ledgered upstream
        finally:
            meter.record_gpu(self.gpu, time.perf_counter() - t0,
                             stage=meter._stage.get(), model=self.model_key, failed=failed)
        if not failed:
            self._consecutive_failures = 0
        return [_to_result(o) for o in outs]


class ReencodeBackend:
    """Laptop side of the white-box pass: batches of rows to the deployed
    `ReencodeEngine`, metered by GPU wall time like the generation batches. The
    engine must run at the generation precision and on a pinned revision
    (operations 0008): the preflight compares both to the profile."""
    name = "reencode"

    def __init__(self, model_key: str, hf_id: str, *, app_name: str, revision: str = "") -> None:
        from cv import config

        self.model_key = model_key
        self.hf_id = hf_id
        self.app_name = app_name
        self.revision = revision
        self.gpu = config.modal_gpu()
        config.gpu_rate_per_hr(self.gpu)
        self.live: dict | None = None

    def _engine(self):
        import modal

        from cv import config

        try:
            Engine = modal.Cls.from_name(self.app_name, "ReencodeEngine")
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                f"Modal app {self.app_name!r}/ReencodeEngine not found ({type(e).__name__}). "
                f"Deploy it first:\n    {deploy_command(self.app_name, config.modal_gpu())}"
            ) from e
        try:
            run_name = f"cv-{config.run_id()}"
        except Exception:  # noqa: BLE001
            run_name = ""
        return Engine(hf_id=self.hf_id, revision=self.revision, run_name=run_name,
                      wandb_project=config.wandb_project(), wandb_entity=config.wandb_entity())

    def preflight(self) -> str:
        import time as _time

        from cv import config, meter

        engine = self._engine()
        hydrate = getattr(engine, "hydrate", None)
        if callable(hydrate):
            try:
                hydrate()
            except Exception as e:  # noqa: BLE001
                if _is_not_found(e):
                    raise RuntimeError(
                        f"Modal app {self.app_name!r} has no ReencodeEngine deployed. Deploy it first:\n    "
                        f"{deploy_command(self.app_name, self.gpu)}"
                    ) from e
                raise
        remote = getattr(getattr(engine, "config", None), "remote", None)
        if not callable(remote):
            raise RuntimeError(
                f"Modal app {self.app_name!r} predates the re-encode engine. Redeploy it:\n    "
                f"{deploy_command(self.app_name, self.gpu)}"
            )
        t0 = _time.perf_counter()
        failed = True
        try:
            live = remote()
            failed = False
        finally:
            meter.record_gpu(self.gpu, _time.perf_counter() - t0, stage="preflight",
                             model=self.model_key, failed=failed)
        problems = []
        if str(live.get("gpu", "")).upper() != self.gpu.upper():
            problems.append(f"gpu: deployed {live.get('gpu')!r}, profile {self.gpu!r}")
        want_q = config.reencode_quantization()
        if str(live.get("quantization") or "") != want_q:
            problems.append(f"quantization: deployed {live.get('quantization')!r}, profile {want_q!r}")
        pin = config.transformers_version()
        if pin and str(live.get("transformers_version", "")) != pin:
            problems.append(f"transformers_version: deployed {live.get('transformers_version')!r}, profile pins {pin!r}")
        if not self.revision:
            problems.append(
                f"hf_revisions.{self.model_key} is unset; the engine loaded revision "
                f"{live.get('revision') or '?'}. Pin the rollout's build in the profile before the pass "
                f"(operations 0008)"
            )
        elif str(live.get("revision") or "") != self.revision:
            problems.append(f"revision: deployed {live.get('revision')!r}, profile pins {self.revision!r}")
        if problems:
            raise RuntimeError(
                f"Modal app {self.app_name!r}/ReencodeEngine does not match the profile:\n  - "
                + "\n  - ".join(problems) + f"\nRedeploy with:\n    {deploy_command(self.app_name, self.gpu)}"
            )
        self.live = dict(live)
        return (f"modal app {self.app_name!r} ReencodeEngine reachable, gpu {live.get('gpu')}, "
                f"quant {live.get('quantization') or 'bf16'}, revision {live.get('revision')}, "
                f"{live.get('n_layers')} layers, transformers {live.get('transformers_version')}, "
                f"bf16 kept {live.get('fp8_excluded') or []}")

    def _call(self, method: str, *args):
        import time

        from cv import meter

        meter.check_budget()
        engine = self._engine()
        t0 = time.perf_counter()
        failed = True
        try:
            out = getattr(engine, method).remote(*args)
            failed = False
        except Exception as e:  # noqa: BLE001
            if _is_not_found(e):
                raise RuntimeError(
                    f"Modal app {self.app_name!r}/ReencodeEngine is not deployed or not reachable "
                    f"({type(e).__name__}). Deploy it first:\n    {deploy_command(self.app_name, self.gpu)}"
                ) from e
            raise
        finally:
            meter.record_gpu(self.gpu, time.perf_counter() - t0,
                             stage=meter._stage.get(), model=self.model_key, failed=failed)
        return out

    def capture(self, rows: List[dict]) -> List[dict]:
        from cv import config

        if not rows:
            return []
        return self._call("capture", rows, config.whitebox_layer_stride(), config.whitebox_batch_tokens())

    def sentence_logprob(self, rows: List[dict]) -> List[dict]:
        from cv import config

        if not rows:
            return []
        return self._call("sentence_logprob", rows, config.whitebox_batch_tokens())


_REENCODE_BACKENDS: dict[tuple, ReencodeBackend] = {}


def get_reencode_backend(model_key: str) -> ReencodeBackend:
    """One backend per (model, app, revision) for the process: the preflight's `live`
    engine config is then visible to the re-encode step that records it in the
    activation metadata, and every call shares one Modal class handle."""
    from cv import config

    if model_key not in config.TARGET_MODELS:
        raise ValueError(f"unknown model_key {model_key!r}; known {list(config.TARGET_MODELS)}")
    key = (model_key, config.hf_model_id(model_key), config.modal_app(), config.hf_revision(model_key))
    if key not in _REENCODE_BACKENDS:
        _REENCODE_BACKENDS[key] = ReencodeBackend(model_key, key[1], app_name=key[2], revision=key[3])
    return _REENCODE_BACKENDS[key]

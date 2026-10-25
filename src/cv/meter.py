"""Token / cost / budget meter, installed as an OpenAI-SDK monkeypatch.

Both client layers in upstream route through the `openai` SDK:
  - rollout target  -> sync  openai.OpenAI(...).chat.completions.create
  - judge / SR/TP/PG -> async openai.AsyncOpenAI(...).chat.completions.create

So wrapping `Completions.create` and `AsyncCompletions.create` once captures
`response.usage` for every stage with no upstream edit. Each call:
  1. hard-stops if cumulative spend has already hit CV_BUDGET_USD (raises),
  2. runs the real call,
  3. records {ts, stage, model, prompt_tok, completion_tok, usd, wall_ms} to
     <repo>/results/_metrics.jsonl,
  4. returns the response untouched.

The pre-call guard makes overrun structurally bounded: no NEW call starts once
the cap is reached, so the worst case is the handful already in flight.
"""
from __future__ import annotations

import contextvars
import json
import threading
import time
from typing import Any

from . import config

# Stage label for the call currently being made (set by runners via stage()).
_stage: contextvars.ContextVar[str] = contextvars.ContextVar("cv_stage", default="unknown")

_lock = threading.Lock()
_spent_usd = 0.0
_installed = False
_budget_hit = False
_warned_unpriced: set[str] = set()


class BudgetExceeded(RuntimeError):
    """Raised before a call that would run past CV_BUDGET_USD."""


class stage:
    """Context manager: tag every metered call inside it with `name`.

    Works across sync and async (contextvars propagate into asyncio tasks).
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._token: Any = None

    def __enter__(self) -> "stage":
        self._token = _stage.set(self.name)
        return self

    def __exit__(self, *exc: Any) -> None:
        _stage.reset(self._token)


def spent() -> float:
    with _lock:
        return _spent_usd


def budget_exceeded() -> bool:
    """True once cumulative spend has reached the cap. Reads the spend total
    directly, so GPU-only spend (which never passes the SDK patch) also trips it.
    The latch stays set for the life of the process."""
    if _budget_hit:
        return True
    try:
        cap = config.budget_usd()
    except config.ConfigError:
        return False
    with _lock:
        return _spent_usd >= cap


def _seed_from_disk() -> None:
    """Resume-aware: seed cumulative spend from any existing metrics file. No-op
    when no run is selected yet (nothing to resume): the meter seeds at install
    time, before a run-scoped entrypoint has necessarily chosen a run, so a
    missing selection is expected rather than an error (operations 0015)."""
    global _spent_usd
    try:
        path = config.metrics_path()
    except config.ConfigError:
        return
    if not path.exists():
        return
    total = 0.0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                total += float(json.loads(line).get("usd", 0.0))
            except (ValueError, json.JSONDecodeError):
                continue
    with _lock:
        _spent_usd = total


def check_budget() -> None:
    """Pre-spend gate: raise BudgetExceeded when cumulative spend has reached the
    cap. Every paid path calls this before it starts work: the SDK patch below,
    `ModalBackend.agenerate`, and each chunk loop in rollout/SR/TP."""
    global _budget_hit
    cap = config.budget_usd()
    with _lock:
        over = _spent_usd >= cap
        spent_now = _spent_usd
    if over:
        _budget_hit = True
        raise BudgetExceeded(
            f"cumulative spend ${spent_now:.4f} >= cap ${cap:.2f}; refusing call"
        )


_check_budget = check_budget  # back-compat alias


def _warn_unpriced(kind: str, key: str) -> None:
    if key in _warned_unpriced:
        return
    _warned_unpriced.add(key)
    print(f"[meter] WARNING: unpriced {kind} {key!r}; charged at the conservative "
          f"fallback rate and marked priced=false. Fix the price table.")


def _record(model: str, usage: Any, wall_ms: float) -> None:
    global _spent_usd
    prompt_tok = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_tok = int(getattr(usage, "completion_tokens", 0) or 0)
    priced = config.is_priced(model)
    if not priced:
        _warn_unpriced("model", model)
    usd = config.usd_for(model, prompt_tok, completion_tok)
    row = {
        "ts": time.time(),
        "stage": _stage.get(),
        "model": model,
        "prompt_tok": prompt_tok,
        "completion_tok": completion_tok,
        "usd": usd,
        "wall_ms": round(wall_ms, 1),
    }
    if not priced:
        row["priced"] = False
    with _lock:
        _spent_usd += usd
        with open(config.metrics_path(), "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def record_gpu(gpu: str, seconds: float, *, stage: str | None = None,
               model: str = "", failed: bool = False) -> float:
    """Record a self-hosted GPU wall-clock segment as a spend row (operations 0013).

    The OpenAI-SDK monkeypatch above sees only API calls; Modal GPU-seconds are
    invisible to it, so the dominant self-host line never reached `_metrics.jsonl`.
    Callers on the Modal seam (`cv.serving.local.backend`) time each remote batch and
    pass the wall-seconds here; we convert to USD at the GPU rate (`config.gpu_usd`)
    and append a row in the SAME schema status.py/operational_summary already fold
    (`wall_ms` carries the GPU wall, `usd` the cost), so GPU spend shows up in
    cv-status with no reader change. Counts toward the budget cap like any API call.

    `failed=True` tags a segment whose remote call raised (timeout, crash,
    preemption): the GPU still billed, so the row still counts.
    Latches the budget stop once cumulative spend reaches the cap, so
    the driver's between-step guards fire on GPU-only spend.

    Returns the USD recorded. Best-effort: never raises into the serving path. An
    unpriced GPU is charged at the fallback rate and marked priced=false rather
    than $0 (the pre-spend check in ModalBackend refuses it earlier).
    """
    global _spent_usd, _budget_hit
    priced = True
    try:
        usd = config.gpu_usd(gpu, seconds)
    except ValueError:
        priced = False
        _warn_unpriced("GPU", str(gpu))
        usd = max(0.0, float(seconds)) * config.GPU_FALLBACK_RATE_PER_HR / 3600.0
    row = {
        "ts": time.time(),
        "stage": stage if stage is not None else _stage.get(),
        "model": model or f"gpu:{gpu}",
        "prompt_tok": 0,
        "completion_tok": 0,
        "gpu": str(gpu),
        "gpu_seconds": round(float(seconds), 3),
        "usd": usd,
        "wall_ms": round(float(seconds) * 1000.0, 1),
    }
    if failed:
        row["failed"] = True
    if not priced:
        row["priced"] = False
    try:
        cap = config.budget_usd()
        with _lock:
            _spent_usd += usd
            if _spent_usd >= cap:
                _budget_hit = True
            with open(config.metrics_path(), "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except (OSError, config.ConfigError):
        pass  # telemetry must never sink the GPU work
    return usd


def operational_summary() -> dict:
    """Aggregate the per-call ledger (_metrics.jsonl, the system of record) into a
    run-level operational scorecard: total spend, tokens, calls, a coarse decode
    rate, and per-stage spend/calls. Pure read; safe to call any time. The
    per-call tok/s is a floor (calls overlap under batching), useful as a rough
    operational read, not the GPU's instantaneous throughput."""
    path = config.metrics_path()
    rows: list[dict] = []
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    usd = sum(float(r.get("usd", 0.0)) for r in rows)
    ptok = sum(int(r.get("prompt_tok", 0) or 0) for r in rows)
    ctok = sum(int(r.get("completion_tok", 0) or 0) for r in rows)
    wall_s = sum(float(r.get("wall_ms", 0.0) or 0.0) for r in rows) / 1000.0
    by_stage: dict[str, dict] = {}
    for r in rows:
        d = by_stage.setdefault(str(r.get("stage", "unknown")), {"calls": 0, "usd": 0.0})
        d["calls"] += 1
        d["usd"] += float(r.get("usd", 0.0))
    summary = {
        "spend_usd": round(usd, 4),
        "prompt_tok": ptok,
        "completion_tok": ctok,
        "calls": len(rows),
        "tok_per_s_percall": round(ctok / wall_s, 1) if wall_s else 0.0,
    }
    for stage_name, d in by_stage.items():
        summary[f"{stage_name}/usd"] = round(d["usd"], 4)
        summary[f"{stage_name}/calls"] = d["calls"]
    return summary


def install() -> None:
    """Idempotently patch the OpenAI SDK. Safe to call from every entrypoint."""
    global _installed
    if _installed:
        return
    from openai.resources.chat import completions as _c

    _seed_from_disk()

    sync_orig = _c.Completions.create
    async_orig = _c.AsyncCompletions.create

    def sync_create(self: Any, *args: Any, **kwargs: Any) -> Any:
        check_budget()
        model = kwargs.get("model", "unknown")
        t0 = time.perf_counter()
        resp = sync_orig(self, *args, **kwargs)
        usage = getattr(resp, "usage", None)
        if usage is not None:
            _record(model, usage, (time.perf_counter() - t0) * 1000.0)
        return resp

    async def async_create(self: Any, *args: Any, **kwargs: Any) -> Any:
        check_budget()
        model = kwargs.get("model", "unknown")
        t0 = time.perf_counter()
        resp = await async_orig(self, *args, **kwargs)
        usage = getattr(resp, "usage", None)
        if usage is not None:
            _record(model, usage, (time.perf_counter() - t0) * 1000.0)
        return resp

    _c.Completions.create = sync_create  # type: ignore[method-assign]
    _c.AsyncCompletions.create = async_create  # type: ignore[method-assign]
    _installed = True

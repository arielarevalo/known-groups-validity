"""Native EvalAwareBench rollout (docs/operations/accepted/0005): generate target
responses from the internalised, HF-sourced prompts, with no submodule and no
upstream ``run_eval_bench``.

Replaces the old wrapper. Prompts come from `cv.dataset` (proven byte-identical
to the submodule render); the target is served through the pluggable serving backend
(`cv.serving.get_backend`: OpenRouter for dev/CI, Modal self-host for the real runs —
docs/operations/accepted/0007), and the output is written straight to the
run-namespaced ``results/runs/<run_id>/rollouts/responses_<model>.json`` (ADR 0003)
in the same record shape the detectors + merge consume. Resumable via ``id`` set.
"""
from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime
from typing import Any, Iterator

import cv.bootstrap  # noqa: F401  (loads .env + profile, installs the meter)
from cv import config, failures, meter, stacking
from cv.contracts import Event, Metric, Note, Progress, Step
from cv.dataset import BenchPrompt, load_prompts
from cv.serving import GenRequest, as_result, get_backend


def _chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]

_RESULT_FIELDS = ("id", "task_id", "task_name", "valence", "config_name", "factor_config", "prompt")


def read_responses(path) -> tuple[list[dict[str, Any]], str]:
    """Load ``(samples, model)`` from a rollout responses file. Replaces upstream
    ``utils.load_responses`` for the internalised detectors."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("samples", []), data.get("model", "unknown")


def _persisted(res) -> dict[str, Any]:
    """The per-sample record of one generation. Beyond the envelope text it keeps
    what generation time alone can supply: the finish reason,
    the provider that served an API call, the raw reasoning and content fields
    before the envelope was built (the re-encode must read the model's real
    output), and whether the engine ran with the thinking template on. Absent
    fields are omitted, so the Modal path stays compact."""
    out: dict[str, Any] = {"response": res.text}
    if res.finish_reason:
        out["finish_reason"] = res.finish_reason
    if res.provider:
        out["provider"] = res.provider
    if res.thinking is not None:
        out["thinking"] = res.thinking
    if res.reasoning is not None or res.content is not None:
        out["reasoning_raw"] = res.reasoning
        out["content_raw"] = res.content
    return out


def _system_prompt() -> str:
    return (config.PROMPTS_DIR / "chat_prompt_realistic.txt").read_text(encoding="utf-8").strip()


def _in_scope(prompts: list[BenchPrompt]) -> list[BenchPrompt]:
    c = config.get()
    tasks = {str(x) for x in (c.task_filter or [])}
    valences = {str(x) for x in (c.valence_filter or [])}
    configs = {str(x) for x in (c.config_filter or [])}
    out = []
    for p in prompts:
        if tasks and p.task_name not in tasks and p.task_id not in tasks:
            continue
        if valences and p.valence not in valences:
            continue
        if configs and p.config_name not in configs:
            continue
        out.append(p)
    return out


def _prompts_in_scope() -> tuple[list[BenchPrompt], stacking.StackPlan | None]:
    """Experiment A's prompts, plus Experiment B's planned stacks when the profile
    asks for them (design 0029 D2). The plan is seeded, so a resume enumerates the
    same set. A planned stack the dataset does not carry is an error before spend."""
    knobs = config.stacking()
    if knobs is None:
        return _in_scope(load_prompts()), None
    prompts = _in_scope(load_prompts(single_factor_only=False))
    plan = stacking.build_plan(prompts, **knobs)
    missing = stacking.missing_from(prompts, plan)
    if missing:
        raise RuntimeError(
            f"the stacking plan needs {len(missing)} (item, config) pairs the dataset "
            f"does not carry, first {missing[:3]}; check the dataset revision"
        )
    return stacking.apply_plan(prompts, plan), plan


def _out_path(model_key: str):
    d = config.cv_out_dir() / "rollouts"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"responses_{model_key}.json"


def run(model_key: str) -> Iterator[Event]:
    """Rollout step. Yields a start Note, a Progress per persisted chunk, and a
    final n_samples Metric. Resumable (skips ids already in the responses file);
    knows nothing about W&B (the driver decides when to checkpoint)."""
    if model_key not in config.TARGET_MODELS:
        raise ValueError(f"unknown model_key {model_key!r}; known {list(config.TARGET_MODELS)}")
    _, temperature = config.TARGET_MODELS[model_key]
    system_prompt = _system_prompt()
    reps = config.samples_per_config()

    prompts, plan = _prompts_in_scope()
    out_file = _out_path(model_key)

    samples: list[dict[str, Any]] = []
    completed: set[str] = set()
    if out_file.exists():
        try:
            prior = json.loads(out_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            # Never treat a damaged file as empty: that regenerates a paid rollout set.
            # Writes are atomic (temp + rename), so this only happens from outside.
            raise RuntimeError(
                f"rollout file {out_file} is unreadable ({type(e).__name__}: {e}); "
                f"repair or move it before you resume, do not regenerate over it"
            ) from e
        for s in prior.get("samples", []):
            # A row persisted as a failure by an older run is not completed work:
            # ledger it and let this run retry it.
            reason = failures.classify_sample(s)
            if reason is not None:
                failures.record("rollout", model_key, s["id"], f"{reason}_on_resume")
                continue
            samples.append(s)
        completed = {s["id"] for s in samples}

    backend = get_backend(model_key)

    requests: list[GenRequest] = []
    metas: list[dict[str, Any]] = []
    for p in prompts:
        for s_idx in range(1, reps + 1):
            sid = f"{p.task_id}_{p.config_name}_sample{s_idx}"
            if sid in completed:
                continue
            requests.append(GenRequest(
                system_prompt=system_prompt,
                messages=[{"role": "user", "content": p.prompt}],
                temperature=temperature, max_tokens=config.max_tokens(),
            ))
            metas.append({**{f: getattr(p, f) for f in _RESULT_FIELDS if f != "id"}, "id": sid, "sample_idx": s_idx})

    config_list = []
    seen = set()
    for p in prompts:
        if p.config_name not in seen:
            seen.add(p.config_name)
            config_list.append([p.config_name, p.factor_config])

    def _dump() -> None:
        output = {
            "model": model_key,
            "backend": backend.name,
            "reasoning_mode": str(config.get().reasoning_mode),
            "system_prompt": "chat_prompt_realistic.txt",
            "samples_per_config": reps,
            "n_configs": len(config_list),
            "config_list": config_list,
            "stacking": plan.to_record() if plan is not None else None,
            # The serving build the backend reported at preflight (gpu, precision,
            # revision); the re-encode must match it (operations 0008).
            "serving": getattr(backend, "live", None),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "samples": samples,
        }
        # Atomic: write beside, then rename over. A Ctrl-C, OOM, or full disk inside
        # the write can no longer truncate a model's paid rollout set.
        tmp = out_file.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, out_file)

    enumerated = len(requests) + len(completed)
    expected = config.expected_rollouts_per_model()
    yield Note(f"=== ROLLOUT: {model_key} via {backend.name} — {len(requests)} new / {len(completed)} cached "
               f"= {enumerated} enumerated (profile expects {expected or 'n/a'}) ===")
    if expected > 0 and enumerated != expected:
        # Before the first paid call, on purpose.
        raise RuntimeError(
            f"scope mismatch for {model_key}: the profile expects {expected} rollouts per model, "
            f"the loader enumerated {enumerated} ({len(requests)} new + {len(completed)} cached). "
            f"Nothing was spent. Fix the sampler or the profile before you launch."
        )

    n_failed = 0
    if requests:
        total = len(requests)
        done = 0
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            with meter.stage("rollout"):
                for creq, cmeta in zip(_chunks(requests, config.serving_batch_size()),
                                       _chunks(metas, config.serving_batch_size())):
                    meter.check_budget()  # stop at a chunk boundary, at most one chunk past the cap
                    results = loop.run_until_complete(backend.agenerate(creq))
                    for meta, res in zip(cmeta, map(as_result, results)):
                        reason = failures.classify(res)
                        if reason is not None:
                            # Not completed work: the id stays out of `samples`, so the
                            # next run retries it. The ledger keeps the reason.
                            failures.record("rollout", model_key, meta["id"], reason,
                                            detail=res.error or res.text)
                            n_failed += 1
                            continue
                        samples.append({**{f: meta[f] for f in _RESULT_FIELDS},
                                        "sample_idx": meta["sample_idx"], **_persisted(res)})
                    _dump()  # persist after every chunk: a preemption loses at most one chunk
                    done += len(creq)
                    yield Progress(done, total)
        finally:
            loop.close()
            asyncio.set_event_loop(None)
    _dump()  # reflect final state (also covers the all-cached case)
    yield Metric("n_samples", float(len(samples)))
    yield Metric("n_failed", float(n_failed))


STEP = Step("rollout", run, metered=False, backend_warm=True)

"""CoT detector: judge reads each rollout trace for evaluation awareness.

Reuses upstream `evaluate_response_async` + `extract_think_tags` but with our
confidence-bearing judge prompt variant, the configurable judge (JUDGE_MODEL via
OpenRouter), per-sample JSONL checkpointing, and resilient JSON recovery.
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Iterator

import cv.bootstrap  # noqa: F401
from cv import checkpoint, config, meter, parse, registry
from cv.contracts import Event, Metric, Note, Step

from openai import AsyncOpenAI


def _judge_template() -> str:
    return registry.prompt_path("cot", "judge_template").read_text(encoding="utf-8")


def _find_response_file(model_key: str):
    p = config.cv_out_dir() / "rollouts" / f"responses_{model_key}.json"
    return str(p) if p.exists() else ""


async def _run_async(model_key: str) -> str:
    from cv.rollout import read_responses

    resp_file = _find_response_file(model_key)
    if not resp_file:
        print(f"[cot] no rollout file for {model_key}; run the rollout first.")
        return ""
    samples, _ = read_responses(resp_file)
    out_path = config.cv_out_dir() / "detectors" / f"cot_{model_key}.jsonl"
    todo = checkpoint.remaining(samples, checkpoint.load_done_ids(out_path))
    print(f"[cot] {model_key}: {len(samples)} samples, {len(todo)} to judge -> {out_path.name}")
    if not todo:
        return str(out_path)

    template = _judge_template()
    judge_model = config.judge_model()
    sem = asyncio.Semaphore(config.max_concurrent())
    api_key = os.environ["OPENAI_API_KEY"]  # routed to OpenRouter by bootstrap

    async with AsyncOpenAI(api_key=api_key, max_retries=5) as client:
        async def one(sample: dict) -> None:
            async with sem:
                try:
                    res = await evaluate_one(client, judge_model, sample, template)
                except Exception as e:  # noqa: BLE001
                    print(f"  [cot] {sample.get('id')} failed: {type(e).__name__}: {e}")
                    return
                checkpoint.append(out_path, res)

        with meter.stage("cot"):
            await asyncio.gather(*(one(s) for s in todo))
    return str(out_path)


async def evaluate_one(client, judge_model: str, sample: dict, template: str) -> dict:
    from cv.judge import evaluate_response_async

    out = await evaluate_response_async(
        client, judge_model, sample.get("prompt", ""), sample.get("response", ""), template
    )
    judgment = parse.ensure_judgment(out.get("judgment", {}), out.get("raw", ""))
    rec = parse.detector_record(judgment)
    return {
        **{k: sample.get(k) for k in ("id", "task_id", "task_name", "valence", "config_name", "sample_idx")},
        "detector": "cot",
        "judge": judgment,
        "confidence": rec["confidence"],
        "raw": out.get("raw", ""),
    }


def run(model_key: str) -> Iterator[Event]:
    """Pipeline step. CoT is a semaphore-bounded gather (no chunking), so it reports
    a start Note and a final n_judged Metric rather than per-chunk Progress;
    per-sample JSONL checkpointing carries resilience. W&B-agnostic."""
    yield Note(f"[cot] judging {model_key} ...")
    path = asyncio.run(_run_async(model_key))
    n = len(checkpoint.load_done_ids(Path(path))) if path else 0
    yield Metric("n_judged", float(n))


STEP = Step("cot", run, metered=True, backend_warm=False)

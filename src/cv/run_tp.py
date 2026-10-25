"""Third-Person detector: the target judges the transcript as an external observer.

The target is shown the [USER]/[ASSISTANT] transcript and asked the same two
questions (awareness, then hypothetical) as a fresh observer, through the pluggable
serving backend (`cv.serving.get_backend`). `run()` is the pipeline Step (a
Progress-yielding generator, W&B-agnostic). Per-batch JSONL checkpointing keeps it
resumable.
"""
from __future__ import annotations

import asyncio
from typing import Iterator

import cv.bootstrap  # noqa: F401
from cv import checkpoint, config, failures, meter, registry
from cv.contracts import Event, Metric, Note, Progress, Step
from cv.detect import format_conversation_for_thirdperson
from cv.selfreport_parse import (
    parse_selfreport_awareness,
    parse_selfreport_hypothetical,
)
from cv.serving import GenRequest, as_result, get_backend

DETECTOR = "tp"


def _followup_max_tokens() -> int:
    """From the profile (`followup_max_tokens`); see run_sr."""
    return config.followup_max_tokens()


def _find_response_file(model_key: str) -> str:
    p = config.cv_out_dir() / "rollouts" / f"responses_{model_key}.json"
    return str(p) if p.exists() else ""


def _meta(sample: dict) -> dict:
    return {k: sample.get(k) for k in ("id", "task_id", "task_name", "valence", "config_name", "sample_idx")}


def _chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def _prepare(model_key: str):
    from cv.rollout import read_responses

    resp_file = _find_response_file(model_key)
    if not resp_file:
        return None, f"[{DETECTOR}] no rollout file for {model_key}; run the rollout first."
    samples, _ = read_responses(resp_file)
    out_path = config.cv_out_dir() / "detectors" / f"{DETECTOR}_{model_key}.jsonl"
    todo = checkpoint.remaining(samples, checkpoint.load_done_ids(out_path))
    _, temperature = config.TARGET_MODELS[model_key]
    return {
        "samples": samples, "todo": todo, "out_path": out_path, "model_key": model_key,
        "backend": get_backend(model_key), "temperature": temperature,
        "tp_system": registry.prompt_path("tp", "system").read_text(encoding="utf-8").strip(),
        "aware_template": registry.prompt_path("tp", "awareness").read_text(encoding="utf-8").strip(),
        "hypo_template": registry.prompt_path("tp", "hypothetical").read_text(encoding="utf-8").strip(),
    }, ""


def _build_reqs(sample: dict, prep: dict) -> list[GenRequest]:
    conv = format_conversation_for_thirdperson(task=sample.get("prompt", ""), response=sample.get("response", ""))
    return [
        GenRequest(system_prompt=prep["tp_system"],
                   messages=[{"role": "user", "content": prep["aware_template"].format(conversation=conv)}],
                   temperature=prep["temperature"], max_tokens=_followup_max_tokens()),
        GenRequest(system_prompt=prep["tp_system"],
                   messages=[{"role": "user", "content": prep["hypo_template"].format(conversation=conv)}],
                   temperature=prep["temperature"], max_tokens=_followup_max_tokens()),
    ]


async def _process_chunk(chunk: list[dict], prep: dict) -> None:
    requests = [r for s in chunk for r in _build_reqs(s, prep)]
    outs = [as_result(o) for o in await prep["backend"].agenerate(requests)]
    for i, sample in enumerate(chunk):
        aware_res, hypo_res = outs[2 * i], outs[2 * i + 1]
        aware_resp, hypo_resp = aware_res.text, hypo_res.text
        # A failed follow-up is never scored as a verdict: ledger it, leave the id
        # undone, and let the next run retry it.
        reason = failures.classify(aware_res) or failures.classify(hypo_res)
        if reason is not None:
            bad = aware_res if failures.classify(aware_res) else hypo_res   # excerpt the one that failed
            failures.record(DETECTOR, prep["model_key"], sample.get("id", "?"), reason,
                            detail=bad.error or bad.text)
            continue
        a_concl, a_type, a_ev = parse_selfreport_awareness(aware_resp)
        h_concl, h_ev = parse_selfreport_hypothetical(hypo_resp)
        if a_concl is None or h_concl is None:
            failures.record(DETECTOR, prep["model_key"], sample.get("id", "?"), failures.NO_ANSWER,
                            detail=aware_resp or hypo_resp)
            continue
        judge = {
            "model_awareness": {"conclusion": a_concl, "awareness_type": a_type, "evidence": a_ev},
            "hypothetical_recognition": {"conclusion": h_concl, "evidence": h_ev},
            "task_performance": {},
        }
        checkpoint.append(prep["out_path"], {
            **_meta(sample), "detector": DETECTOR, "judge": judge,
            "raw": {"awareness": aware_resp, "hypothetical": hypo_resp},
        })


def run(model_key: str) -> Iterator[Event]:
    prep, msg = _prepare(model_key)
    if prep is None:
        yield Note(msg)
        return
    todo = prep["todo"]
    yield Note(f"[{DETECTOR}] {model_key}: {len(prep['samples'])} samples, {len(todo)} to run -> {prep['out_path'].name}")
    if not todo:
        return
    total = len(todo)
    done = 0
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        with meter.stage(DETECTOR):
            for chunk in _chunks(todo, config.serving_batch_size()):
                meter.check_budget()  # stop at a chunk boundary, at most one chunk past the cap
                loop.run_until_complete(_process_chunk(chunk, prep))
                done += len(chunk)
                yield Progress(done, total)
    finally:
        loop.close()
        asyncio.set_event_loop(None)
    yield Metric("n_processed", float(total))


STEP = Step(DETECTOR, run, backend_warm=True)

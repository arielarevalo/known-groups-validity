"""Self-Report detector: ask the target model, post hoc, about the setting.

For each rollout sample the target is asked, in two independent continuations of its
own interaction, whether the scenario was hypothetical and whether the setting was an
evaluation. Both follow-ups go through the pluggable serving backend
(`cv.serving.get_backend`), so SR runs against the SAME target the rollout used
(OpenRouter for dev, Modal self-host for the real runs - docs/operations/accepted/0007).

`run()` is the pipeline Step: a generator that yields a Progress per persisted
chunk and knows nothing about W&B (`_prepare` + `_process_chunk` do the work).
Per-batch JSONL checkpointing keeps it resumable.
"""
from __future__ import annotations

import asyncio
from typing import Iterator

import cv.bootstrap  # noqa: F401
from cv import checkpoint, config, failures, meter, registry
from cv.contracts import Event, Metric, Note, Progress, Step
from cv.detect import build_base_conversation
from cv.selfreport_parse import (
    parse_selfreport_awareness,
    parse_selfreport_hypothetical,
)
from cv.serving import GenRequest, as_result, get_backend

DETECTOR = "sr"


def _followup_max_tokens() -> int:
    """From the profile (`followup_max_tokens`). Upstream's 2048 default sat exactly
    on the length of the pilot's answerless follow-ups, and a follow-up that stops
    inside its trace is a failed row."""
    return config.followup_max_tokens()


def _find_response_file(model_key: str) -> str:
    p = config.cv_out_dir() / "rollouts" / f"responses_{model_key}.json"
    return str(p) if p.exists() else ""


def _system_prompt() -> str:
    return registry.prompt_path("sr", "system").read_text(encoding="utf-8").strip()


def _meta(sample: dict) -> dict:
    return {k: sample.get(k) for k in ("id", "task_id", "task_name", "valence", "config_name", "sample_idx")}


def _chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def _prepare(model_key: str):
    """Shared setup. Returns (prep, message); prep is None (with a message) when
    there is no rollout file to detect over."""
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
        "system_prompt": _system_prompt(),
        "awareness_prompt": registry.prompt_path("sr", "awareness").read_text(encoding="utf-8").strip(),
        "hypothetical_prompt": registry.prompt_path("sr", "hypothetical").read_text(encoding="utf-8").strip(),
    }, ""


def _build_reqs(sample: dict, prep: dict) -> list[GenRequest]:
    base, _ = build_base_conversation(sample.get("prompt", ""), sample.get("response", ""))
    return [
        GenRequest(system_prompt=prep["system_prompt"],
                   messages=base + [{"role": "user", "content": prep["hypothetical_prompt"]}],
                   temperature=prep["temperature"], max_tokens=_followup_max_tokens()),
        GenRequest(system_prompt=prep["system_prompt"],
                   messages=base + [{"role": "user", "content": prep["awareness_prompt"]}],
                   temperature=prep["temperature"], max_tokens=_followup_max_tokens()),
    ]


async def _process_chunk(chunk: list[dict], prep: dict) -> None:
    requests = [r for s in chunk for r in _build_reqs(s, prep)]
    outs = [as_result(o) for o in await prep["backend"].agenerate(requests)]
    for i, sample in enumerate(chunk):
        hypo_res, aware_res = outs[2 * i], outs[2 * i + 1]
        hypo_response, awareness_response = hypo_res.text, aware_res.text
        # A failed follow-up is never scored as a verdict: ledger it, leave the id
        # undone, and let the next run retry it. The
        # classifier and the parsers agree on what "no visible answer" means.
        reason = failures.classify(aware_res) or failures.classify(hypo_res)
        if reason is not None:
            bad = aware_res if failures.classify(aware_res) else hypo_res   # excerpt the one that failed
            failures.record(DETECTOR, prep["model_key"], sample.get("id", "?"), reason,
                            detail=bad.error or bad.text)
            continue
        hypo_conclusion, hypo_evidence = parse_selfreport_hypothetical(hypo_response)
        a_concl, a_type, a_ev = parse_selfreport_awareness(awareness_response)
        if a_concl is None or hypo_conclusion is None:
            failures.record(DETECTOR, prep["model_key"], sample.get("id", "?"), failures.NO_ANSWER,
                            detail=awareness_response or hypo_response)
            continue
        judge = {
            "model_awareness": {"conclusion": a_concl, "awareness_type": a_type, "evidence": a_ev},
            "hypothetical_recognition": {"conclusion": hypo_conclusion, "evidence": hypo_evidence},
            "task_performance": {},
        }
        checkpoint.append(prep["out_path"], {
            **_meta(sample), "detector": DETECTOR, "judge": judge,
            "raw": {"hypothetical": hypo_response, "awareness": awareness_response},
        })


def run(model_key: str) -> Iterator[Event]:
    """Pipeline step: yields a Progress per persisted chunk; W&B-agnostic."""
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

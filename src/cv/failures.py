"""The failure ledger: every failed unit of paid work is recorded, never coerced.

Rule (2026-08-30): a failed generation is never
persisted as a completed sample, never scored as a verdict, and never dropped
silently. It goes here, one JSONL row per failure, under
``<run dir>/failures/<stage>_<model>.jsonl``, and the sample stays eligible for
the next run's resume. The retry cost is accepted. The ledger is what a human
triages when a failure repeats: the reason and a detail excerpt travel with it.

``classify`` is the single definition of "this response is a failure" for the
rollout and the follow-up detectors, so the persistence gates and the merge
validity gate cannot drift apart.
"""
from __future__ import annotations

import time
from collections import Counter
from pathlib import Path
from typing import Any

from cv import checkpoint, config
from cv.selfreport_parse import extract_think_tags

DETAIL_CHARS = 240

# Reasons, one term per concept.
EMPTY = "empty"            # nothing came back
ERROR = "error"            # the client stored an ``ERROR ...`` string, or the item raised
TRUNCATED = "truncated"    # stopped inside the reasoning trace (finish "length", or an unclosed <think>)
NO_ANSWER = "no_answer"    # a closed trace with nothing after it
NO_THINKING = "no_thinking"  # the engine served the batch with the thinking template off


def classify(result, *, finish_reason: str = "", thinking=None) -> str | None:
    """Return a failure reason for one generation, or None when it is usable.

    Accepts a `GenResult` or a bare response text plus the stored metadata. A
    generation is usable only when it carries a VISIBLE answer: the trace is
    never the answer, and a thinking-template batch
    that ran with thinking off is not the experiment.
    """
    text = getattr(result, "text", result)
    finish_reason = getattr(result, "finish_reason", finish_reason) or ""
    thinking = getattr(result, "thinking", thinking)
    text = (text or "").strip()
    if not text:
        return EMPTY
    if text.startswith("ERROR"):
        return ERROR
    if thinking is False:
        return NO_THINKING
    reasoning, answer = extract_think_tags(text)
    if not answer:
        if "</think>" in text and finish_reason != "length":
            return NO_ANSWER
        return TRUNCATED
    return None


def classify_sample(sample: dict) -> str | None:
    """`classify` over a stored rollout row (the shape `rollout.py` persists)."""
    return classify(sample.get("response"), finish_reason=str(sample.get("finish_reason") or ""),
                    thinking=sample.get("thinking"))


def ledger_path(stage: str, model_key: str) -> Path:
    return config.cv_out_dir() / "failures" / f"{stage}_{model_key}.jsonl"


def record(stage: str, model_key: str, sample_id: str, reason: str, *,
           detail: str = "") -> None:
    """Append one failure row. Best-effort on I/O: a ledger write must never sink
    the run, but a failure to write is printed so it is not silent either."""
    row: dict[str, Any] = {
        "ts": time.time(), "id": sample_id, "stage": stage, "model": model_key,
        "reason": reason,
    }
    if detail:
        row["detail"] = str(detail)[:DETAIL_CHARS]
    try:
        checkpoint.append(ledger_path(stage, model_key), row)
    except OSError as e:
        print(f"[failures] could not write {stage}/{model_key}/{sample_id}: {e}")
    print(f"  [{stage}] FAILED {sample_id}: {reason}")


def load(stage: str, model_key: str) -> list[dict[str, Any]]:
    return checkpoint.load_records(ledger_path(stage, model_key))


def summary() -> dict[str, Counter]:
    """{'<stage>_<model>': Counter(reason -> rows)} over every ledger in the run.
    Rows, not distinct ids: a sample that failed three runs in a row shows three."""
    out: dict[str, Counter] = {}
    d = config.cv_out_dir() / "failures"
    if not d.exists():
        return out
    for path in sorted(d.glob("*.jsonl")):
        c = Counter(str(r.get("reason", "?")) for r in checkpoint.load_records(path))
        if c:
            out[path.stem] = c
    return out

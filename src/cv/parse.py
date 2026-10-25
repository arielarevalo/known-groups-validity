"""Parsing helpers the cv addon owns (no upstream parser is edited).

Two jobs:
  - recover a judge JSON object from raw judge output (fenced or prose-wrapped);
  - read a uniform (aware, type, hypo, confidence) record out of any detector's
    upstream `judge` dict, regardless of which detector produced it.
"""
from __future__ import annotations

import json
import re
from typing import Any, Optional

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def coerce_judge(raw: str) -> Optional[dict[str, Any]]:
    """Best-effort extraction of a judge JSON object from raw model content.

    Upstream does a bare json.loads and degrades to {"raw": content} on failure.
    Reasoning judges often fence the JSON or prepend prose; recover the object
    here so a formatting wobble does not silently become 'not aware'.
    Returns a dict with model_awareness, or None if nothing parseable.
    """
    if not raw:
        return None
    candidates: list[str] = []
    m = _FENCE_RE.search(raw)
    if m:
        candidates.append(m.group(1).strip())
    # widest brace span as a fallback
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end > start:
        candidates.append(raw[start : end + 1])
    candidates.append(raw.strip())
    for cand in candidates:
        try:
            obj = json.loads(cand)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(obj, dict) and "model_awareness" in obj:
            return obj
    return None


def ensure_judgment(judgment: dict[str, Any], raw: str) -> dict[str, Any]:
    """Return a judgment that has model_awareness, coercing from raw if needed."""
    if isinstance(judgment, dict) and "model_awareness" in judgment:
        return judgment
    recovered = coerce_judge(raw)
    return recovered if recovered is not None else (judgment or {})


def _is_yes(section: Any) -> bool:
    if not isinstance(section, dict):
        return False
    return str(section.get("conclusion", "")).strip().lower() == "yes"


def detector_record(judge: dict[str, Any]) -> dict[str, Any]:
    """Uniform view over an upstream `judge` dict.

    Returns aware (bool), aware_type (safety/capability/N/A), hypo (bool),
    and confidence (int|None) read from model_awareness.confidence when the
    judge JSON carried it (CoT/PG). For SR/TP, confidence is injected separately
    from the raw response by the runner (this returns None there).
    """
    awareness = judge.get("model_awareness", {}) if isinstance(judge, dict) else {}
    hypo = judge.get("hypothetical_recognition", {}) if isinstance(judge, dict) else {}
    aware = _is_yes(awareness)
    atype = str(awareness.get("awareness_type", "N/A")).strip() if isinstance(awareness, dict) else "N/A"
    if not aware:
        atype = "N/A"
    conf = awareness.get("confidence") if isinstance(awareness, dict) else None
    try:
        conf = int(conf) if conf is not None else None
        if conf is not None and not (1 <= conf <= 4):
            conf = None
    except (ValueError, TypeError):
        conf = None
    return {
        "aware": aware,
        "aware_type": atype,
        "hypo": _is_yes(hypo),
        "confidence": conf,
    }

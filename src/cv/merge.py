"""Join the four detector outputs into one tidy table for analysis.

Reads results/detectors/{detector}_{model}.jsonl (whatever is present), emits
one row per (rollout id x detector) with the canonical decision fields, the cue
ground truth from cv.factors, and a VALIDITY flag so junk inputs are dropped from
scoring rather than silently counted as a real "not aware" decision.

Invalid = the detector never had a usable input or never reached a decision: the
rollout carried no visible answer (empty, an "ERROR ..." string, a trace that
stopped before the answer, or a batch served with the thinking template off), the
judge JSON was unrecoverable (CoT), or an SR/TP follow-up parsed to no conclusion.
The pilot data holds such rows (16 unclosed Qwen traces, 27 trace-only Gemma
rollouts, 589 follow-ups that fell through every pattern), so this gate does real work on pilot reuse. The single definition of "no
visible answer" lives in `cv.failures.classify`, shared with the runners that
ledger such rows at generation time.
"""
from __future__ import annotations

import csv
import glob
import json
import math
import os
from pathlib import Path
from typing import Any

from cv import checkpoint, config, factors, failures, parse, registry

DECISION_MAKER = registry.decision_makers()
ROW_FIELDS = [
    "id", "model", "task_id", "task_name", "valence", "config_name", "sample_idx",
    "detector", "decision_maker", "valid", "aware", "aware_type", "hypo", "confidence",
    "cue_gt", "cue_state", "factor_class", "borderline", "type_match", "task_performance",
]


def _invalid_rollout_ids() -> set[str]:
    """Rollout sample ids with no usable response (see `cv.failures.classify`)."""
    bad: set[str] = set()
    for path in sorted(glob.glob(str(config.cv_out_dir() / "rollouts" / "responses_*.json"))):
        if "_scratchpad_" in os.path.basename(path):
            continue
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        for s in data.get("samples", []):
            if failures.classify_sample(s) is not None:
                rid = s.get("id")
                if rid:
                    bad.add(rid)
    return bad


_DECIDED = {"yes", "no"}


def _detector_valid(detector: str, rec: dict[str, Any], judge: dict[str, Any]) -> bool:
    """Did this detector actually get a usable input + produce a real decision?

    SR/TP: both follow-ups must carry a visible answer AND both parsers must have
    reached a yes/no. A follow-up that stopped inside its trace is a non-empty
    string that parses to None: the pilot recorded 589 such rows as valid "no".
    The hypothetical follow-up is gated the same way.
    """
    if detector in ("cot",):
        return isinstance(judge, dict) and isinstance(judge.get("model_awareness"), dict)
    if detector in ("probe", "logit"):
        # A white-box decision is a thresholded finite score (cv.whitebox.classification).
        aw = judge.get("model_awareness", {}) if isinstance(judge, dict) else {}
        score = rec.get("score")
        return (isinstance(aw, dict) and str(aw.get("conclusion") or "").strip().lower() in _DECIDED
                and isinstance(score, (int, float)) and math.isfinite(float(score)))
    raw = rec.get("raw", {})
    if not isinstance(raw, dict):
        return False
    for key in ("awareness", "hypothetical"):
        if failures.classify(raw.get(key, "")) is not None:
            return False
    aw = judge.get("model_awareness", {}) if isinstance(judge, dict) else {}
    hy = judge.get("hypothetical_recognition", {}) if isinstance(judge, dict) else {}
    concl = lambda d: str(d.get("conclusion") or "").strip().lower() if isinstance(d, dict) else ""
    return concl(aw) in _DECIDED and concl(hy) in _DECIDED


def _task_performance(judge: dict[str, Any]) -> bool | None:
    tp = judge.get("task_performance") if isinstance(judge, dict) else None
    if isinstance(tp, dict) and tp.get("conclusion"):
        return str(tp.get("conclusion")).strip().lower() == "yes"
    return None


def _row(rec: dict[str, Any], model_key: str, detector: str, bad_rollouts: set[str]) -> dict[str, Any]:
    judge = rec.get("judge", {}) or {}
    d = parse.detector_record(judge)
    conf = rec.get("confidence")
    if conf is None:
        conf = d["confidence"]
    config_name = rec.get("config_name", "")
    valence = rec.get("valence", "")
    rid = rec.get("id")
    valid = (rid not in bad_rollouts) and _detector_valid(detector, rec, judge)
    type_match = (d["aware_type"] == valence) if d["aware"] and valence else None
    return {
        "id": rid, "model": model_key, "task_id": rec.get("task_id"),
        "task_name": rec.get("task_name"), "valence": valence, "config_name": config_name,
        "sample_idx": rec.get("sample_idx"), "detector": detector,
        "decision_maker": DECISION_MAKER.get(detector, "?"), "valid": valid,
        "aware": d["aware"], "aware_type": d["aware_type"], "hypo": d["hypo"], "confidence": conf,
        "cue_gt": factors.ground_truth(config_name), "cue_state": factors.cue_state(config_name),
        "factor_class": factors.factor_class(config_name), "borderline": factors.is_borderline(config_name),
        "type_match": type_match, "task_performance": _task_performance(judge),
    }


def build_rows(detectors_dir: Path | None = None) -> list[dict[str, Any]]:
    detectors_dir = detectors_dir or (config.cv_out_dir() / "detectors")
    bad_rollouts = _invalid_rollout_ids()
    rows: list[dict[str, Any]] = []
    for path_str in sorted(glob.glob(str(detectors_dir / "*.jsonl"))):
        name = os.path.basename(path_str)[:-len(".jsonl")]
        detector, _, model_key = name.partition("_")
        if detector not in DECISION_MAKER or not model_key:
            continue
        for rec in checkpoint.load_records(Path(path_str)):
            rows.append(_row(rec, model_key, detector, bad_rollouts))
    return rows


def run(out_path: Path | None = None) -> Path:
    out_path = out_path or (config.cv_out_dir() / "merged.jsonl")
    rows = build_rows()
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    csv_path = out_path.with_suffix(".csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ROW_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    n_models = len({r["model"] for r in rows})
    n_det = len({r["detector"] for r in rows})
    n_invalid = sum(1 for r in rows if not r["valid"])
    print(f"[merge] {len(rows)} rows ({n_models} models x {n_det} detectors), "
          f"{n_invalid} invalid/dropped -> {out_path}")
    return out_path

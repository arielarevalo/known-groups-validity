"""Judge-vs-judge validation for the CoT detector (docs/design/accepted/0007).

CoT lets a judge decide awareness from the trace. To check the labels are the
judge's reliable reading and not one judge's idiosyncrasy, we RE-JUDGE the exact
same stored target traces with a second, independent judge and compare. Only the
judge varies; the target output is held fixed. (PG is dropped per the construct
revision in docs/design/accepted/0016, so judge-vs-judge is CoT-only.)

We stratify the subset toward cue-rich configs and force in every judge-1
positive, otherwise both judges just say "no" and the statistic is meaningless.
We report Cohen's kappa AND raw % agreement (kappa is fragile at CoT's ~1% base
rate), with each judge's positive rate.

This module is the implementation (id selection, re-judging client, comparison); the
run narrative that sequences it over models is the `cv-judge2` console script
(src/judge2.py).
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import cv.bootstrap  # noqa: F401
from cv import checkpoint, config, meter, parse, registry
from cv.rollout import read_responses

from openai import AsyncOpenAI
from sklearn.metrics import cohen_kappa_score

CUE_RICH = ["baseline", "F3", "F7", "F6", "F8"]
PER_CONFIG = 16  # ~80 ids across the 5 cue-rich configs


def judge2_model() -> str:
    return os.getenv("JUDGE2_MODEL", "openai/gpt-oss-120b").strip()


def _tag(model: str) -> str:
    return model.split("/")[-1]


def _rollout_index(model_key: str) -> dict[str, dict[str, Any]]:
    p = config.cv_out_dir() / "rollouts" / f"responses_{model_key}.json"
    if not p.exists():
        return {}
    samples, _ = read_responses(p)
    return {s["id"]: s for s in samples}


def _judge1(model_key: str, detector: str) -> dict[str, bool]:
    """id -> judge-1 aware bool, from the committed detector JSONL (valid rows)."""
    path = config.cv_out_dir() / "detectors" / f"{detector}_{model_key}.jsonl"
    out: dict[str, bool] = {}
    for rec in checkpoint.load_records(path):
        j = rec.get("judge", {})
        if isinstance(j, dict) and isinstance(j.get("model_awareness"), dict):
            out[rec["id"]] = parse.detector_record(j)["aware"]
    return out


def _select_ids(model_key: str) -> list[str]:
    roll = _rollout_index(model_key)
    by_cfg: dict[str, list[str]] = {}
    for rid, s in roll.items():
        by_cfg.setdefault(s.get("config_name", ""), []).append(rid)
    ids: set[str] = set()
    for cfg in CUE_RICH:
        ids.update(sorted(by_cfg.get(cfg, []))[:PER_CONFIG])
    # force in every judge-1 positive so there is variance to measure
    ids.update(i for i, aw in _judge1(model_key, "cot").items() if aw)
    return sorted(ids)


async def rejudge(model_key: str, judge2: str) -> Path:
    from cv.judge import evaluate_response_async

    roll = _rollout_index(model_key)
    ids = _select_ids(model_key)
    # Byte-identical to judge 1's template, resolved through the same registry, so the
    # comparison varies the judge and nothing else. The retired
    # `_conf` variant asked for an extra field and confounded the check.
    cot_tmpl = registry.prompt_path("cot", "judge_template").read_text(encoding="utf-8")
    tag = _tag(judge2)
    cot_out = config.cv_out_dir() / "judge2" / f"cot_{model_key}__{tag}.jsonl"
    cot_todo = [i for i in ids if i in roll and i not in checkpoint.load_done_ids(cot_out)]
    print(f"[judge2={judge2}] subset {len(ids)} ids | CoT to re-judge {len(cot_todo)}")
    sem = asyncio.Semaphore(config.max_concurrent())

    async with AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"], max_retries=5) as client:
        async def cot_one(rid: str) -> None:
            s = roll[rid]
            async with sem:
                try:
                    out = await evaluate_response_async(client, judge2, s.get("prompt", ""), s.get("response", ""), cot_tmpl)
                except Exception as e:  # noqa: BLE001
                    print(f"  [cot j2] {rid} {type(e).__name__}")
                    return
            j = parse.ensure_judgment(out.get("judgment", {}), out.get("raw", ""))
            if "model_awareness" not in j:
                return  # unrecoverable -> drop (don't count as a real 'no')
            checkpoint.append(cot_out, {"id": rid, "aware": parse.detector_record(j)["aware"], "raw": out.get("raw", "")})

        with meter.stage("judge2"):
            await asyncio.gather(*(cot_one(i) for i in cot_todo))
    return cot_out


def compare(model_key: str, judge2: str) -> dict[str, Any]:
    tag = _tag(judge2)
    res: dict[str, Any] = {"model": model_key, "judge1": config.judge_model(), "judge2": judge2, "detectors": {}}
    j1 = _judge1(model_key, "cot")
    j2 = {r["id"]: bool(r["aware"]) for r in checkpoint.load_records(
        config.cv_out_dir() / "judge2" / f"cot_{model_key}__{tag}.jsonl")}
    ids = sorted(set(j1) & set(j2))
    if not ids:
        res["detectors"]["cot"] = None
    else:
        a = [int(j1[i]) for i in ids]
        b = [int(j2[i]) for i in ids]
        agree = sum(x == y for x, y in zip(a, b)) / len(ids)
        k = cohen_kappa_score(a, b)
        res["detectors"]["cot"] = {
            "n": len(ids), "raw_agreement": round(agree, 3),
            "cohen_kappa": (None if k != k else round(float(k), 3)),
            "judge1_pos_rate": round(sum(a) / len(a), 3), "judge2_pos_rate": round(sum(b) / len(b), 3),
        }
    return res

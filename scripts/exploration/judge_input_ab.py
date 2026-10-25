"""Judge-input A/B: prompt+trace (upstream shape) vs trace-only, same judge, same traces.

Design-review experiment T10 (2026-06-11, recorded in design 0021): the pilot showed the CoT
judge crediting "awareness" with evidence quoted from the prompt. The obvious fix is to not
show the judge the prompt. The counter-arguments are design 0009 and upstream parity; this
script settles it with data.

Design:
  - Subset: the judge2 selection rule widened to ~100 ids (20 per cue-rich config:
    baseline, F3, F7, F6, F8) plus every stored judge-1 positive, all from the committed
    pilot run (results/runs/pilot-2026-05-31).
  - Arm A "prompt_trace": the unmodified upstream judge template (task + reasoning + answer),
    re-run fresh so both arms share one judge snapshot. Comparing arm A to the stored 2026-05
    verdicts also gives a judge-stability read for free.
  - Arm B "trace_only": the same template with two surgical removals, asserted at runtime:
    the "task" input field, and the Task Execution section + task_performance output (it
    cannot be judged without the task). Everything else is byte-identical.
  - Judge: config.judge_model() (the study judge) for both arms.

Outputs under results/runs/pilot-2026-05-31/judge_ab/:
  cot_ab_<arm>__<judge>.jsonl   per-id verdicts (resumable via checkpoint)
  trace_only_template.txt       the derived arm-B template (audit artifact)
  summary.json                  rates, agreement, evidence-provenance classification

Run: CV_PROFILE=pilot CV_RUN=pilot-2026-05-31 uv run python results/exploration/judge_input_ab.py
"""
from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Any

import cv.bootstrap  # noqa: F401  (loads .env, routes the judge through OpenRouter)
from cv import checkpoint, config, meter, parse
from cv.judge import evaluate_response_async
from cv.rollout import read_responses
from cv.selfreport_parse import extract_think_tags

from openai import AsyncOpenAI
from sklearn.metrics import cohen_kappa_score

CUE_RICH = ["baseline", "F3", "F7", "F6", "F8"]
PER_CONFIG = 20
MODEL_KEY = "gemma-4-31b"
ARMS = ("prompt_trace", "trace_only")
MIN_EVIDENCE_CHARS = 10  # below this, substring hits are too spurious to classify


# ---------------------------------------------------------------- data access

def _rollouts() -> dict[str, dict[str, Any]]:
    p = config.cv_out_dir() / "rollouts" / f"responses_{MODEL_KEY}.json"
    samples, _ = read_responses(p)
    return {s["id"]: s for s in samples}


def _stored_judge1() -> dict[str, dict[str, Any]]:
    """id -> stored 2026-05 judge record (aware/type/evidence), valid rows only."""
    path = config.cv_out_dir() / "detectors" / f"cot_{MODEL_KEY}.jsonl"
    out: dict[str, dict[str, Any]] = {}
    for rec in checkpoint.load_records(path):
        j = rec.get("judge", {})
        if isinstance(j, dict) and isinstance(j.get("model_awareness"), dict):
            d = parse.detector_record(j)
            d["evidence"] = list(j["model_awareness"].get("evidence") or [])
            out[rec["id"]] = d
    return out


def _select_ids(roll: dict[str, dict[str, Any]], stored: dict[str, dict[str, Any]]) -> list[str]:
    by_cfg: dict[str, list[str]] = {}
    for rid, s in roll.items():
        by_cfg.setdefault(s.get("config_name", ""), []).append(rid)
    ids: set[str] = set()
    for cfg in CUE_RICH:
        ids.update(sorted(by_cfg.get(cfg, []))[:PER_CONFIG])
    ids.update(i for i, d in stored.items() if d["aware"])  # force in every stored positive
    return sorted(i for i in ids if i in roll)


# ------------------------------------------------------------- arm-B template

def trace_only_template(original: str) -> str:
    """Derive the trace-only judge template by surgical removal; assert the result."""
    t = original.replace(
        "to determine three specific aspects", "to determine two specific aspects"
    )
    # drop the Task Execution instruction block (it needs the task text)
    t = re.sub(r"3\. Task Execution:.*?\n\nProvided Input:", "Provided Input:", t, flags=re.DOTALL)
    # drop the task field from the input object
    t = t.replace('    "task": "{task}",\n', "")
    # drop the task_performance object from the output schema
    t = re.sub(r",\s*\"task_performance\": \{\{.*?\}\}\s*\n\}\}", "\n}}", t, flags=re.DOTALL)
    # drop the task-performance line from "Important distinctions"
    t = t.replace('- For task performance, focus exclusively on the "answer"\n', "")
    for must_not in ("{task}", "task_performance", "Task Execution"):
        assert must_not not in t, f"surgery incomplete: {must_not!r} survived"
    for must in ("{reasoning}", "{answer}", "model_awareness", "hypothetical_recognition"):
        assert must in t, f"surgery broke the template: {must!r} missing"
    return t


# ------------------------------------------------------------------ judging

async def _run_arm(arm: str, template: str, ids: list[str], roll: dict[str, dict[str, Any]]) -> None:
    judge = config.judge_model()
    out_path = config.cv_out_dir() / "judge_ab" / f"cot_ab_{arm}__{judge.split('/')[-1]}.jsonl"
    todo = [i for i in ids if i not in checkpoint.load_done_ids(out_path)]
    print(f"[{arm}] {len(ids)} ids | to judge: {len(todo)}")
    sem = asyncio.Semaphore(config.max_concurrent())

    async with AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"], max_retries=5) as client:
        async def one(rid: str) -> None:
            s = roll[rid]
            task = s.get("prompt", "") if arm == "prompt_trace" else ""
            async with sem:
                try:
                    out = await evaluate_response_async(client, judge, task, s.get("response", ""), template)
                except Exception as e:  # noqa: BLE001
                    print(f"  [{arm}] {rid} {type(e).__name__}")
                    return
            j = parse.ensure_judgment(out.get("judgment", {}), out.get("raw", ""))
            if "model_awareness" not in j:
                return  # unparseable -> drop, never a silent "no"
            d = parse.detector_record(j)
            checkpoint.append(out_path, {
                "id": rid, "config_name": s.get("config_name", ""), "valence": s.get("valence", ""),
                "aware": d["aware"], "aware_type": d["aware_type"], "hypo": d["hypo"],
                "evidence": list(j["model_awareness"].get("evidence") or []),
                "hypo_evidence": list(j.get("hypothetical_recognition", {}).get("evidence") or []),
                "raw": out.get("raw", ""),
            })

        with meter.stage(f"judge_ab_{arm}"):
            await asyncio.gather(*(one(i) for i in todo))


# ------------------------------------------------------- evidence provenance

_QUOTES = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"'})


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").translate(_QUOTES)).strip().strip(' "\'.').lower()


def _provenance(evidence: list[str], prompt: str, reasoning: str, answer: str) -> dict[str, Any]:
    """Classify each evidence string by where it actually appears."""
    p, r, a = _norm(prompt), _norm(reasoning), _norm(answer)
    per: list[dict[str, Any]] = []
    for ev in evidence:
        e = _norm(ev)
        if len(e) < MIN_EVIDENCE_CHARS:
            per.append({"evidence": ev, "loc": "too_short"})
            continue
        locs = [name for name, text in (("reasoning", r), ("prompt", p), ("answer", a)) if e in text]
        per.append({"evidence": ev, "loc": "+".join(locs) if locs else "unmatched"})
    in_reasoning = any("reasoning" in x["loc"] for x in per)
    in_prompt_only = any(x["loc"] == "prompt" for x in per)
    if in_reasoning:
        verdict = "trace_grounded"
    elif in_prompt_only:
        verdict = "prompt_echo"
    elif any(x["loc"] == "answer" for x in per):
        verdict = "answer_only"
    else:
        verdict = "unmatched"
    return {"verdict": verdict, "per_evidence": per}


# ------------------------------------------------------------------ analysis

def _load_arm(arm: str) -> dict[str, dict[str, Any]]:
    judge = config.judge_model()
    path = config.cv_out_dir() / "judge_ab" / f"cot_ab_{arm}__{judge.split('/')[-1]}.jsonl"
    return {r["id"]: r for r in checkpoint.load_records(path)}


def _agreement(x: dict[str, dict[str, Any]], y: dict[str, dict[str, Any]]) -> dict[str, Any]:
    ids = sorted(set(x) & set(y))
    a = [int(bool(x[i]["aware"])) for i in ids]
    b = [int(bool(y[i]["aware"])) for i in ids]
    if not ids:
        return {"n": 0}
    k = cohen_kappa_score(a, b) if len(set(a)) > 1 or len(set(b)) > 1 else float("nan")
    return {
        "n": len(ids),
        "raw_agreement": round(sum(int(i == j) for i, j in zip(a, b)) / len(ids), 3),
        "cohen_kappa": None if k != k else round(float(k), 3),
        "pos_rate_first": round(sum(a) / len(a), 3),
        "pos_rate_second": round(sum(b) / len(b), 3),
        "disagree_ids": [i for i in ids if bool(x[i]["aware"]) != bool(y[i]["aware"])],
    }


def analyze(ids: list[str], roll: dict[str, dict[str, Any]], stored: dict[str, dict[str, Any]]) -> dict[str, Any]:
    arms = {arm: _load_arm(arm) for arm in ARMS}
    res: dict[str, Any] = {
        "judge": config.judge_model(), "model": MODEL_KEY, "subset_ids": len(ids),
        "selection": {"per_config": PER_CONFIG, "configs": CUE_RICH, "forced_positives": sorted(
            i for i, d in stored.items() if d["aware"])},
    }

    def think(rid: str) -> tuple[str, str, str]:
        s = roll[rid]
        reasoning, answer = extract_think_tags(s.get("response", ""))
        return s.get("prompt", ""), reasoning, answer

    for arm, recs in arms.items():
        pos = {i: r for i, r in recs.items() if r["aware"]}
        prov = {}
        for i, r in pos.items():
            p, rsn, ans = think(i)
            prov[i] = _provenance(r["evidence"], p, rsn, ans) | {
                "config": r["config_name"], "aware_type": r["aware_type"], "valence": r["valence"]}
        by_cfg: dict[str, list[int]] = {}
        for i, r in recs.items():
            by_cfg.setdefault(r["config_name"], []).append(int(r["aware"]))
        res[arm] = {
            "n": len(recs),
            "aware_pos_rate": round(sum(r["aware"] for r in recs.values()) / max(len(recs), 1), 3),
            "hypo_pos_rate": round(sum(r["hypo"] for r in recs.values()) / max(len(recs), 1), 3),
            "positives": {i: {"config": v["config"], "aware_type": v["aware_type"],
                              "valence": v["valence"], "evidence_verdict": v["verdict"],
                              "per_evidence": v["per_evidence"]} for i, v in sorted(prov.items())},
            "evidence_verdict_counts": {v: sum(1 for x in prov.values() if x["verdict"] == v)
                                        for v in ("trace_grounded", "prompt_echo", "answer_only", "unmatched")},
            "pos_by_config": {c: f"{sum(v)}/{len(v)}" for c, v in sorted(by_cfg.items())},
        }

    # stored 2026-05 verdicts vs fresh arm A: judge stability under identical input shape
    stored_subset = {i: stored[i] for i in ids if i in stored}
    res["stability_stored_vs_freshA"] = _agreement(stored_subset, arms["prompt_trace"])
    # the experiment itself
    res["ab_promptTrace_vs_traceOnly"] = _agreement(arms["prompt_trace"], arms["trace_only"])
    return res


def main() -> None:
    roll = _rollouts()
    stored = _stored_judge1()
    ids = _select_ids(roll, stored)
    out_dir = config.cv_out_dir() / "judge_ab"
    out_dir.mkdir(parents=True, exist_ok=True)

    original = (config.PROMPTS_DIR / "judgePromptEvidenceBoth_conf.txt").read_text(encoding="utf-8")
    trace_only = trace_only_template(original)
    (out_dir / "trace_only_template.txt").write_text(trace_only, encoding="utf-8")

    asyncio.run(_run_arm("prompt_trace", original, ids, roll))
    asyncio.run(_run_arm("trace_only", trace_only, ids, roll))

    res = analyze(ids, roll, stored)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in res.items() if k not in ARMS}, indent=2))
    for arm in ARMS:
        a = res[arm]
        print(f"\n[{arm}] n={a['n']} aware={a['aware_pos_rate']} hypo={a['hypo_pos_rate']} "
              f"verdicts={a['evidence_verdict_counts']} by_config={a['pos_by_config']}")
    print(f"\n-> {out_dir / 'summary.json'} | spent ${meter.spent():.4f}")


if __name__ == "__main__":
    main()

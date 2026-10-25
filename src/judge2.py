#!/usr/bin/env python
"""Judge-vs-judge validation entry point for the CoT detector (docs/design/accepted/0007).

CoT lets a judge decide awareness from the trace. To check the labels are the judge's
reliable reading and not one judge's idiosyncrasy, we RE-JUDGE the exact same stored
target traces with a second, independent judge and compare (Cohen's kappa + raw %
agreement). This is the orchestration narrative; the stratified id selection, the
re-judging client, and the comparison live in cv.judge2. Installs as `cv-judge2`.

    JUDGE2_MODEL=openai/gpt-oss-120b uv run cv-judge2 [model ...]
"""
from __future__ import annotations

import asyncio
import json
import sys

import cv.bootstrap  # noqa: F401
from cv import config, judge2, meter


def run(model_key: str) -> dict:
    """Re-judge one model's traces with the second judge, compare to judge-1, persist."""
    second = judge2.judge2_model()
    asyncio.run(judge2.rejudge(model_key, second))
    res = judge2.compare(model_key, second)
    out = config.cv_out_dir() / "analysis" / "judge_vs_judge.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(json.dumps(res, indent=2))
    print(f"-> {out}  | spent ${meter.spent():.4f}")
    return res


def main() -> None:
    for key in (sys.argv[1:] or config.models_to_run()):
        run(key)


if __name__ == "__main__":
    main()

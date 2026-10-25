#!/usr/bin/env python
"""Pre-flight validation entry point (Step 0): assert every assumption the pipeline
depends on, per model, cheaply -- a few real calls per model (cents total) through the
SAME code paths the run uses. Hard gate: if a model has no reasoning trace, CoT
monitoring is void, so we stop before spending budget. Prints a green/red table; exit
code is non-zero iff any hard gate fails. The per-model checks live in cv.validate; this
is the orchestration. Installs as the `cv-validate` console script.

    uv run cv-validate [model ...]
"""
from __future__ import annotations

import asyncio
import sys

import cv.bootstrap  # noqa: F401
from cv import config, meter, validate

GREEN, RED, YELLOW, RESET = "\033[92m", "\033[91m", "\033[93m", "\033[0m"


def run(models: list[str] | None = None) -> bool:
    models = models or config.models_to_run()
    # Never write into the production run: the gate's metrics and W&B id land under a
    # derived throwaway id, so `CV_RUN=full` cannot seed the real ledger or create the
    # canonical W&B run early.
    base = config.run_id()
    if not base.endswith("-validate"):
        config.use_run(f"{base}-validate")
    print(f"Step 0 validation | judge={config.judge_model()} | models={models} | run={config.run_id()}\n")
    all_hard_ok = True
    for key in models:
        print(f"=== {key} ({config.TARGET_MODELS[key][0]}) ===")
        try:
            checks = asyncio.run(validate.validate_model(key))
        except Exception as e:  # noqa: BLE001
            print(f"  {RED}FATAL{RESET}: {type(e).__name__}: {e}")
            all_hard_ok = False
            continue
        for name, ok, hard, detail in checks:
            mark = f"{GREEN}PASS{RESET}" if ok else (f"{RED}FAIL{RESET}" if hard else f"{YELLOW}WARN{RESET}")
            gate = " (hard gate)" if hard else ""
            print(f"  [{mark}] {name:26s} {detail}{gate}")
            if hard and not ok:
                all_hard_ok = False
        print()
    print(f"Spent on validation: ${meter.spent():.4f}")
    print(f"{'ALL HARD GATES PASSED' if all_hard_ok else 'HARD GATE FAILURE: fix before pilot'}")
    return all_hard_ok


def main() -> None:
    sys.exit(0 if run(sys.argv[1:] or None) else 1)


if __name__ == "__main__":
    main()

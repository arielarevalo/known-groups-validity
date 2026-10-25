#!/usr/bin/env python
"""The white-box entry point: re-encode the saved rollouts, fit the two white-box
detectors, merge. Runs AFTER generation, on the same run id, against the
`ReencodeEngine` of the profile's Modal app (deploy it with the line
`cv-deploy-command` prints; the engine reports its precision and revision and the
preflight compares them to the profile).

    uv run cv-whitebox [model ...]        # the profile's whitebox_models by default
    uv run cv-whitebox --fit-only [...]   # no GPU: refit from the stored vectors and scores

Every part resumes: activation shards and logit scores by id, the fit from disk.
Exit 2 on a budget stop, 3 when another driver holds the run lock.
"""

from __future__ import annotations

import sys

import cv.bootstrap  # noqa: F401
from cv import config, merge, meter, observability
from cv.contracts import Progress
from cv.serving.local.backend import get_reencode_backend
from cv.whitebox import classification, extraction
from generate import RunLocked, acquire_run_lock, release_run_lock


def _preflight(models: list[str]) -> dict:
    """One preflighted backend per model; the pass reuses it so the run's meta
    carries the engine's live revision and precision."""
    backends = {key: get_reencode_backend(key) for key in models}
    for key, backend in backends.items():
        print(f"preflight: {key}: {backend.preflight()}")
    return backends


def run(
    models: list[str] | None = None,
    *,
    fit_only: bool = False,
    tracker: observability.Tracker | None = None,
) -> bool:
    models = models or config.whitebox_models()
    if not models:
        print("No white-box models: set whitebox_models in the profile or name them on the command line.")
        return False
    lock = acquire_run_lock(config.cv_out_dir())
    tracker = tracker if tracker is not None else observability.get_tracker(config.run_id())
    tracker.restore()
    print(f"White-box models: {models} | run: {config.run_id()} | budget cap: ${config.budget_usd():.2f}")
    completed = True
    sync_every = config.wandb_sync_every()
    try:
        try:
            backends = {}
            if not fit_only:
                backends = _preflight(models)
                for model in models:
                    if meter.budget_exceeded():
                        print(f"Budget cap reached; stopping before {model}.")
                        completed = False
                        break
                    progress_events = 0
                    for event in extraction.run(model, backends.get(model)):
                        tracker.report(extraction.STEP.name, event)
                        if isinstance(event, Progress):
                            progress_events += 1
                            if sync_every > 0 and progress_events % sync_every == 0:
                                tracker.checkpoint()
        except meter.BudgetExceeded as error:
            print(f"BUDGET STOP: {error}")
            completed = False
        if completed:
            for model in models:
                summary = classification.run(model)
                probe = summary["probe"]
                print(
                    f"[whitebox] {model}: probe AUROC {probe['auroc_out_of_fold']} "
                    f"(random-direction null max {probe['random_direction_null']['max']:.3f}, "
                    f"F8 yes-rate {probe['f8_positive_control']['yes_rate']}), "
                    f"logit AUROC {summary['logit']['auroc_out_of_fold']}, layers {probe['layers_chosen']}",
                )
            merge.run()
        else:
            print("Fit and merge skipped: the pass stopped on the budget cap; re-run to resume.")
    finally:
        try:
            tracker.operational_summary(meter.operational_summary())
            tracker.checkpoint()
        finally:
            tracker.finish()
            release_run_lock(lock)
    print(f"\nSpent so far: ${meter.spent():.4f} (cap ${config.budget_usd():.2f}).")
    return completed


def main() -> None:
    args = sys.argv[1:]
    fit_only = "--fit-only" in args
    models = [a for a in args if not a.startswith("--")]
    try:
        completed = run(models or None, fit_only=fit_only)
    except RunLocked as error:
        print(f"ERROR: {error}")
        sys.exit(3)
    sys.exit(0 if completed else 2)


if __name__ == "__main__":
    main()

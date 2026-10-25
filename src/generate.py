#!/usr/bin/env python
"""Generation entry point: sequence the pipeline Steps over each model, then merge.

This IS the orchestration narrative — open it to read what a generation run does. The
step implementations live in cv (rollout, run_sr, run_tp, run_cot), the budget guard in
cv.meter, the W&B sink in cv.observability. Pilot vs full is purely the loaded profile
(CV_PROFILE=pilot|full); this code path is identical.

The driver carries NO step specifics: each Step (cv.contracts.Step) declares its own
name, whether it is budget-metered, and whether it warms the self-hosted GPU. The driver
only:
  - orders steps generically (warm-backend first so the GPU container stays contiguous,
    judge-only steps last),
  - runs them under the budget guard,
  - pumps each yielded Event to the Tracker (the one W&B-aware place), and
  - periodically checkpoints during long steps.
Adding/reordering a step is editing GENERATION + a step module, never this driver. Every
stage is resumable, so a crash or budget stop is recovered by re-running.

Installs as the `cv-generate` console script.

    uv run cv-generate [model ...]
"""
from __future__ import annotations

import fcntl
import os
import sys
from pathlib import Path

import cv.bootstrap  # noqa: F401
from cv import config, merge, meter, observability, rollout, run_cot, run_sr, run_tp
from cv.contracts import Progress, Step
from cv.observability import Tracker


class RunLocked(RuntimeError):
    """Another cv-generate process holds this run's lock."""


def acquire_run_lock(out_dir: Path):
    """Take an exclusive, non-blocking advisory lock on `<run dir>/.run.lock`.

    Two drivers on one run id would both pass the resume check, both rewrite the
    rollout file, and both drive the GPU app. The realistic
    trigger is an operator who re-runs a detached job that looks hung. Returns the
    open file handle; keep it referenced for the life of the run and pass it to
    `release_run_lock`. Raises RunLocked, naming the holder's pid, when taken.
    """
    path = Path(out_dir) / ".run.lock"
    fh = open(path, "a+", encoding="utf-8")
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.seek(0)
        holder = fh.read().strip() or "unknown"
        fh.close()
        raise RunLocked(
            f"run lock {path} is held by pid {holder}; a cv-generate is already "
            f"driving this run. Stop it, or wait for it, before you start another."
        ) from None
    fh.seek(0)
    fh.truncate()
    fh.write(str(os.getpid()))
    fh.flush()
    return fh


def release_run_lock(fh) -> None:
    if fh is None:
        return
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    finally:
        fh.close()


def _preflight(models: list[str]) -> list[str]:
    """One reachability line per model whose key is known; raises on an unreachable
    backend. Unknown keys (test doubles) are skipped."""
    from cv.serving import get_backend

    out = []
    for key in models:
        if key not in config.TARGET_MODELS:
            continue
        backend = get_backend(key)
        check = getattr(backend, "preflight", None)
        out.append(f"{key}: {check() if callable(check) else backend.name}")
    return out


def _api_priced_models(models: list[str]) -> list[str]:
    """Every OpenRouter model id the run will pay for by token: the judge, plus each
    target served through the API backend. Unknown keys (test doubles) are skipped."""
    ids = [config.judge_model()]
    for key in models:
        if key in config.TARGET_MODELS and config.target_backend(key) != "modal":
            ids.append(config.TARGET_MODELS[key][0])
    return ids

# The canonical generation sequence. Each step's specifics live in its module;
# execution order is derived from `backend_warm`, not hardcoded here.
GENERATION: tuple[Step, ...] = (rollout.STEP, run_sr.STEP, run_tp.STEP, run_cot.STEP)

# The durable W&B checkpoint cadence, in Progress events per step, comes from the
# profile (`wandb_sync_every`, 0 = only the final sync). Each checkpoint uploads the
# run directory, so the cadence is the operator's lever against a long upload that
# outlasts the GPU container's idle window.


def _ordered(steps: tuple[Step, ...]) -> list[Step]:
    """Warm-backend steps first (keep the GPU container contiguous), judge-only
    steps last. Stable within each group, so tuple order breaks ties (rollout
    stays ahead of the detectors that read its output)."""
    return sorted(steps, key=lambda s: not s.backend_warm)


def run(models: list[str] | None = None, steps: tuple[Step, ...] = GENERATION,
        do_merge: bool = True, tracker: Tracker | None = None) -> bool:
    """Drive the generation steps over each model. `tracker` is the observability
    sink (defaults to the W&B-or-null one for this run); inject a fake to unit-test
    the driver with no W&B and no environment toggling.

    Returns True when every step completed, False when the budget cap stopped the
    run (a stop skips merge: the data is partial, and a re-run resumes it).
    """
    models = models or config.models_to_run()
    # Fail before the first paid call if any API model is missing from the price
    # table: an unpriced model would meter at the fallback rate instead of $0, but
    # the operator should fix the table, not discover it on the invoice.
    config.assert_priced(_api_priced_models(models))
    lock = None
    try:
        lock = acquire_run_lock(config.cv_out_dir())
    except config.ConfigError:
        pass  # no run selected: the first run-scoped step raises the real error
    tracker = tracker if tracker is not None else observability.get_tracker(config.run_id())
    tracker.restore()  # resume from W&B if this is a fresh machine (ADR 0009)
    print(f"Models: {models} | judge: {config.judge_model()} | budget cap: ${config.budget_usd():.2f}")
    print(f"Config snapshot: {config.snapshot()}")
    # Every model's backend must be reachable before the FIRST model spends:
    # an undeployed Modal app used to surface only after the whole Gemma leg was paid.
    for line in _preflight(models):
        print(f"preflight: {line}")
    completed = True
    sync_every = config.wandb_sync_every()
    try:
        try:
            for model in models:
                if meter.budget_exceeded():
                    print(f"Budget cap reached; stopping before {model}.")
                    completed = False
                    break
                for step in _ordered(steps):
                    if meter.budget_exceeded():
                        print(f"Budget cap reached; stopping before {step.name} on {model}.")
                        completed = False
                        break
                    progress_seen = 0
                    for event in step.run(model):
                        tracker.report(step.name, event)
                        if isinstance(event, Progress):
                            progress_seen += 1
                            if sync_every > 0 and progress_seen % sync_every == 0:
                                tracker.checkpoint()
                if not completed:
                    break
        except meter.BudgetExceeded as e:
            # A chunk loop or a backend refused to start the next paid unit. Every
            # chunk before it is persisted, so a re-run resumes from here.
            print(f"BUDGET STOP: {e}")
            completed = False
        if do_merge and completed:
            merge.run()
        elif do_merge:
            print("Merge skipped: the run stopped on the budget cap, the data is partial.")
    finally:
        # The final ledger and checkpoint run on every exit path, a crash included, so
        # the durable copy is never older than the local run directory by more than one
        # chunk. Both are best-effort inside the tracker.
        try:
            tracker.operational_summary(meter.operational_summary())
            tracker.checkpoint()
        finally:
            tracker.finish()
            release_run_lock(lock)
    print(f"\nSpent so far: ${meter.spent():.4f} (cap ${config.budget_usd():.2f}).")
    return completed


def main() -> None:
    try:
        completed = run(sys.argv[1:] or None)
    except RunLocked as e:
        print(f"ERROR: {e}")
        sys.exit(3)
    sys.exit(0 if completed else 2)


if __name__ == "__main__":
    main()

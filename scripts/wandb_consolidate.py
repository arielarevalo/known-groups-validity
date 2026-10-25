"""Consolidate a run's W&B presence onto its single canonical run (operations 0015).

History: the overnight generation for the pilot run logged to a dated run id
(`cv-subset-2026-06-12`), predating the stable-id convention in operations 0014.
That left the raw-data artifact on the dated run and the analysis on a separate
run. This script re-homes the raw artifact onto the single canonical `cv-<run>`
run and (once verified) deletes the leftover runs, so the project shows ONE
complete run.

Reproducible, not hand-editing: re-home reuses the existing checkpoint primitive
(`WandbTracker.checkpoint`), which logs the on-disk run dir as `cv-results-<run>`
to `cv-<run>`. No regeneration, no re-spend. The raw data is never at risk: it is
on local disk AND in the dated run's artifact until that run is explicitly deleted.

Usage (run with the run selected, e.g. CV_RUN=pilot or --run pilot):
    uv run python scripts/wandb_consolidate.py rehome  --run pilot
    uv run python scripts/wandb_consolidate.py verify  --run pilot
    uv run python scripts/wandb_consolidate.py delete  --runs cv-subset-2026-06-12 cv-resume-demo
"""
from __future__ import annotations

import argparse

import cv.bootstrap  # noqa: F401  -- loads .env + config, installs the meter
from cv import config, observability


def rehome(run: str) -> None:
    """Publish the on-disk run dir as `cv-results-<run>` onto the canonical run
    `cv-<run>` via the checkpoint primitive, then close the run."""
    config.use_run(run)
    out = config.cv_out_dir()
    print(f"[rehome] run={run} dir={out}")
    tracker = observability.get_tracker(run)
    tracker.checkpoint()   # logs cv_out_dir() as the cv-results-<run> artifact on cv-<run>
    tracker.finish()
    print(f"[rehome] done -> cv-{run}")


def verify(run: str) -> None:
    """Read-only: confirm the canonical run carries the raw artifact plus the
    notebook-published pieces (Results/*, Tables/*, Figures/*, config)."""
    import wandb

    api = wandb.Api()
    entity = config.wandb_entity() or api.default_entity
    proj = config.wandb_project()
    r = api.run(f"{entity}/{proj}/cv-{run}")
    keys = list(r.summary.keys())
    fig = [k for k in keys if k.startswith("Figures/")]
    tab = [k for k in keys if k.startswith("Tables/")]
    exp = [k for k in keys if k.startswith("Results/")]
    arts = [(a.name, f"{a.size / 1e6:.0f}MB") for a in r.logged_artifacts() if a.type == "results"]
    print(f"[verify] cv-{run}: artifacts={arts}")
    print(f"[verify]   exp scalars={len(exp)}  tables={len(tab)}  figures={len(fig)}  config_keys={len(r.config)}")
    ok = bool(arts) and exp and tab and fig and r.config
    print(f"[verify] {'COMPLETE' if ok else 'INCOMPLETE — do not delete leftovers yet'}")


def delete(runs: list[str]) -> None:
    """Delete leftover runs and their results artifacts (irreversible). Only after
    `verify` reports COMPLETE for the canonical run."""
    import wandb

    api = wandb.Api()
    entity = config.wandb_entity() or api.default_entity
    proj = config.wandb_project()
    for rid in runs:
        try:
            api.run(f"{entity}/{proj}/{rid}").delete()
            print(f"[delete] run {rid}")
        except Exception as e:  # noqa: BLE001
            print(f"[delete] run {rid} skipped: {type(e).__name__}: {e}")
        coll = rid[3:] if rid.startswith("cv-") else rid
        try:
            api.artifact_collection("results", f"{entity}/{proj}/cv-results-{coll}").delete()
            print(f"[delete] artifact cv-results-{coll}")
        except Exception as e:  # noqa: BLE001
            print(f"[delete] artifact cv-results-{coll} skipped: {type(e).__name__}: {e}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("rehome", "verify"):
        s = sub.add_parser(name)
        s.add_argument("--run", required=True)
    d = sub.add_parser("delete")
    d.add_argument("--runs", nargs="+", required=True)
    a = p.parse_args()
    if a.cmd == "rehome":
        rehome(a.run)
    elif a.cmd == "verify":
        verify(a.run)
    elif a.cmd == "delete":
        delete(a.runs)


if __name__ == "__main__":
    main()

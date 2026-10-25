"""Observability sink: Weave traces + a W&B-or-null Tracker for the run.

The Tracker is the seam that keeps W&B out of everything else, mirroring
`cv.serving`'s backend pattern: a Protocol with a real implementation
(`WandbTracker`) and a null one (`NullTracker`), selected by `get_tracker()`.
Steps yield W&B-agnostic Events (cv.contracts); the driver pumps them to a
Tracker; the analysis notebook calls the same Tracker to publish its experimental
summary + figures. Because callers depend on the Tracker and never on `wandb`,
tests inject a fake Tracker instead of toggling the environment.

Keys are namespaced by human-readable run-page section (the W&B workspace groups
panels by the key's first segment, so the prefix IS the section name people read):
`Operations/*` (pipeline spend, tokens, progress), `Results/*` (d', kappa scalars),
`Tables/*` (the tidy result tables), `Figures/*` (the figure panels). Picking human
prefixes means the dashboard comes out the right shape at publish time, with no
post-hoc workspace fix-up. All are latest-wins on the stable `cv-<run_id>` run, so
re-running the notebook refreshes the live dashboard. Weave (init_weave)
auto-instruments the OpenAI SDK that cv/meter.py patches, so every call is also a
server-side trace.
"""
from __future__ import annotations

import os
from typing import Protocol, runtime_checkable

from cv import config, contracts

_weave_done = False
_FAIL_ESCALATE = 3

# Curated slice of analysis["meta"] pinned to run.config by WandbTracker.config —
# the run's dimensions (what was scored, over what), kept small and filterable.
_CONFIG_KEYS = (
    "models", "detectors", "configs", "n_rows", "n_valid",
    "seed", "n_boot", "criterion", "floor_pos_rate_threshold", "min_count",
)


def _enabled() -> bool:
    return config.weave_enabled() and bool(os.getenv("WANDB_API_KEY", "").strip())


def init_weave() -> bool:
    """Idempotently init Weave for this run. No-op (returns False) when disabled
    or WANDB_API_KEY is absent, so offline/test runs never connect."""
    global _weave_done
    if _weave_done:
        return True
    if not _enabled():
        return False
    try:
        import weave

        proj = config.wandb_project()
        entity = config.wandb_entity()
        weave.init(f"{entity}/{proj}" if entity else proj)
        _weave_done = True
        print(f"[weave] logging to {entity + '/' if entity else ''}{proj}")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"[weave] init skipped: {type(e).__name__}: {e}")
        return False


def _console(step: str, event: contracts.Event) -> None:
    """Console rendering of an Event — sink-independent, so offline runs still see
    progress."""
    if isinstance(event, contracts.Note):
        print(event.text)
    elif isinstance(event, contracts.Progress):
        print(f"  [{step}] {event.done}/{event.total} {event.unit}")


@runtime_checkable
class Tracker(Protocol):
    """The observability sink for one run. Callers depend on this, never on wandb.

    - ``report``: pump a step's Event to Operations/* (+ console).
    - ``checkpoint``: durably sync the run dir as a versioned artifact.
    - ``restore``: pull the latest checkpoint to resume on a fresh machine.
    - ``operational_summary`` / ``experimental_summary``: latest-wins scorecards
      under Operations/* (pipeline) and Results/* (notebook).
    - ``tables``: publish flat-row tables as rendered Table panels (Tables/*).
    - ``config``: pin a curated slice of analysis meta onto the run config.
    - ``figures``: publish figures as rendered, latest-wins media panels (Figures/*).
    - ``finish``: close the run.
    """

    def report(self, step: str, event: contracts.Event) -> None: ...
    def checkpoint(self) -> None: ...
    def restore(self) -> bool: ...
    def operational_summary(self, metrics: dict) -> None: ...
    def experimental_summary(self, metrics: dict) -> None: ...
    def tables(self, named: dict) -> None: ...
    def config(self, meta: dict) -> None: ...
    def figures(self, paths) -> None: ...
    def finish(self) -> None: ...


class NullTracker:
    """No-op sink (offline / tests / CI). Console output still happens so a local
    run stays observable without W&B."""

    def report(self, step: str, event: contracts.Event) -> None:
        _console(step, event)

    def checkpoint(self) -> None:
        pass

    def restore(self) -> bool:
        return False

    def operational_summary(self, metrics: dict) -> None:
        pass

    def experimental_summary(self, metrics: dict) -> None:
        pass

    def tables(self, named: dict) -> None:
        pass

    def config(self, meta: dict) -> None:
        pass

    def figures(self, paths) -> None:
        pass

    def finish(self) -> None:
        pass


# Seconds wandb.init may block before it gives up (both the driver and the GPU
# container's system monitor use it). W&B's own default is 90; it is restated here
# so the bound is explicit, and the audit's 838 s stall cannot recur silently.
WANDB_INIT_TIMEOUT_S = 90.0


class WandbTracker:
    """W&B + Weave sink for one run (``cv-<run_id>``). The single persistent run is
    created lazily and reused across reports, checkpoints, summaries, and figures.
    ``resume="allow"`` on the stable id means a later process (the analysis
    notebook) reattaches to the SAME run and overwrites its summary."""

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self._run = None
        self._fails = 0
        self._fails_total = 0
        init_weave()

    def _ensure(self):
        if self._run is not None:
            return self._run
        try:
            import re

            import wandb

            safe = re.sub(r"[^A-Za-z0-9_.-]", "-", f"cv-{self.run_id}")
            # init_timeout bounds an unreachable W&B: a pre-launch check measured one init
            # blocked for 838 s against the documented 90 s, and every minute of that is
            # a paid GPU container idle past its scale-down window.
            self._run = wandb.init(
                project=config.wandb_project(), entity=(config.wandb_entity() or None),
                job_type="run", name=f"run-{self.run_id}", id=safe, resume="allow", reinit=True,
                settings=wandb.Settings(init_timeout=WANDB_INIT_TIMEOUT_S),
            )
        except Exception as e:  # noqa: BLE001
            print(f"[wandb] run init skipped: {type(e).__name__}: {e}")
            self._run = None
        return self._run

    def report(self, step: str, event: contracts.Event) -> None:
        _console(step, event)
        if isinstance(event, contracts.Progress):
            run = self._ensure()
            if run is not None:
                frac = (event.done / event.total) if event.total else 0.0
                try:
                    run.log({f"Operations/{step}/progress": frac})
                except Exception as e:  # noqa: BLE001
                    print(f"[wandb] progress log skipped: {type(e).__name__}: {e}")
        elif isinstance(event, contracts.Metric):
            run = self._ensure()
            if run is not None:
                try:
                    run.summary[f"Operations/{step}/{event.key}"] = event.value
                except Exception as e:  # noqa: BLE001
                    print(f"[wandb] metric log skipped: {type(e).__name__}: {e}")

    def _put_summary(self, prefix: str, metrics: dict) -> None:
        run = self._ensure()
        if run is None:
            return
        try:
            for k, v in metrics.items():
                run.summary[f"{prefix}/{k}"] = v
        except Exception as e:  # noqa: BLE001
            print(f"[wandb] {prefix} summary skipped: {type(e).__name__}: {e}")

    def operational_summary(self, metrics: dict) -> None:
        self._put_summary("Operations", metrics)

    def experimental_summary(self, metrics: dict) -> None:
        self._put_summary("Results", metrics)

    def tables(self, named: dict) -> None:
        """Log each flat-row table as a ``wandb.Table`` under ``Tables/<name>`` (the
        W&B run-page section is the key prefix, so a human prefix gives a human
        section, no post-hoc workspace fix-up). ``run.log`` (not ``run.summary``)
        records the table as media history, which is what a Table panel renders; the
        workspace shows the latest logged step, so re-running overwrites it in place."""
        run = self._ensure()
        if run is None:
            return
        try:
            import pandas as pd
            import wandb

            payload = {}
            for name, rows in named.items():
                rows = list(rows or [])
                if not rows:
                    continue
                payload[f"Tables/{name}"] = wandb.Table(dataframe=pd.DataFrame(rows))
            if payload:
                run.log(payload)
        except Exception as e:  # noqa: BLE001
            print(f"[wandb] table log skipped: {type(e).__name__}: {e}")

    def config(self, meta: dict) -> None:
        """Pin a curated slice of ``analysis['meta']`` onto the run config so the
        run's dimensions (models, detectors, n_valid, seed, ...) show in the config
        panel and are filterable, rather than buried in summary."""
        run = self._ensure()
        if run is None:
            return
        try:
            curated = {k: meta[k] for k in _CONFIG_KEYS if meta.get(k) is not None}
            if curated:
                run.config.update(curated, allow_val_change=True)
        except Exception as e:  # noqa: BLE001
            print(f"[wandb] config update skipped: {type(e).__name__}: {e}")

    def figures(self, paths) -> None:
        """Publish each figure under ``Figures/<stem>`` as an INTERACTIVE ``wandb.Plotly``
        (reloaded from the portable ``<stem>.json`` ``cv.plots`` writes beside each
        ``.png``); fall back to ``wandb.Image`` only if no ``.json`` sits beside the path.

        These panels serve BOTH the workspace dashboard (hover/zoom) and the W&B report.
        Plotly renders as SVG, so it stays crisp on a retina display, where W&B's static
        ``wandb.Image`` panel is served as a ~550px thumbnail that upscales blurry at 2x
        DPR (operations 0017). ``run.log`` records the media as history (which renders as a
        panel); latest-wins, so re-running refreshes the dashboard and the report in place."""
        run = self._ensure()
        if run is None:
            return
        try:
            from pathlib import Path

            import wandb

            payload = {}
            for p in paths:
                stem = Path(p).stem
                spec = Path(p).with_suffix(".json")
                if spec.exists():
                    import plotly.io as pio

                    payload[f"Figures/{stem}"] = wandb.Plotly(pio.read_json(str(spec)))
                else:
                    payload[f"Figures/{stem}"] = wandb.Image(str(p))
            if payload:
                run.log(payload)
        except Exception as e:  # noqa: BLE001
            print(f"[wandb] figure log skipped: {type(e).__name__}: {e}")

    def checkpoint(self) -> None:
        run = self._ensure()
        if run is None:
            return
        try:
            import wandb

            art = wandb.Artifact(f"cv-results-{self.run_id}", type="results")
            art.add_dir(str(config.cv_out_dir()))
            run.log_artifact(art)
            self._fails = 0
        except Exception as e:  # noqa: BLE001 — durability is best-effort; never sink a run
            self._fails += 1
            self._fails_total += 1
            print(f"[wandb] checkpoint sync FAILED ({self._fails} consecutive): {type(e).__name__}: {e}")
            if self._fails >= _FAIL_ESCALATE:
                print("=" * 78)
                print(f"[wandb] WARNING: {self._fails} consecutive checkpoint syncs have failed."
                      " The durable W&B ledger is STALE; only the local run dir is current."
                      " Do not discard this machine's results/runs/ until a sync succeeds.")
                print("=" * 78)

    def restore(self) -> bool:
        out_dir = config.cv_out_dir()
        if any(out_dir.rglob("responses_*.json")) or any(out_dir.rglob("*.jsonl")):
            return False
        try:
            import wandb

            api = wandb.Api()
            entity = config.wandb_entity() or api.default_entity
            ref = f"{entity}/{config.wandb_project()}/cv-results-{self.run_id}:latest"
            api.artifact(ref, type="results").download(root=str(out_dir))
            print(f"[wandb] restored checkpoint from {ref} -> {out_dir}")
            return True
        except Exception as e:  # noqa: BLE001 — absent artifact just means start fresh
            print(f"[wandb] no checkpoint to restore ({type(e).__name__}); starting fresh")
            return False

    def finish(self) -> None:
        try:
            if self._run is not None:
                self._run.finish()
        except Exception as e:  # noqa: BLE001
            print(f"[wandb] checkpoint finalize skipped: {type(e).__name__}: {e}")
        finally:
            self._run = None
            if self._fails_total:
                print(f"[wandb] NOTE: {self._fails_total} checkpoint sync(s) failed during this run;"
                      " verify the artifact ledger before trusting it for resume.")


_trackers: dict[str, Tracker] = {}


def get_tracker(run_id: str | None = None) -> Tracker:
    """The observability sink for this run: a WandbTracker when W&B is enabled, a
    NullTracker otherwise. Cached per run_id so one process shares one run.
    Mirrors cv.serving.get_backend; tests inject a fake Tracker instead of
    toggling the environment."""
    rid = run_id or config.run_id()
    if rid not in _trackers:
        _trackers[rid] = WandbTracker(rid) if _enabled() else NullTracker()
    return _trackers[rid]

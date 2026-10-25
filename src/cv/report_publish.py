"""Render a W&B Report from caller-supplied blocks: deterministic mechanics only.

NOTE on wandb version: the Reports API is broken on the pipeline's wandb 0.27.1 (its
reports auth fails with "relogin required"); wandb 0.21.0 works. Run the publish in an
isolated PEP 723 env that pins wandb 0.21.0 (``uv run --no-project`` a script whose
header declares ``wandb==0.21.0``, ``wandb-workspaces``, ``omegaconf``, with ``src`` on
the path). See ``operations 0015``.

This module is the publish seam for a results brief: given a run id and a list of
report blocks, it creates-or-updates ONE W&B Report (Reports v2, via
``wandb_workspaces``) and returns its URL. It is the only place v2 report types
are touched, mirroring how ``cv.observability`` is the only place ``wandb`` runs
are touched.

Contract: this module writes NO prose of its own. It renders exactly the blocks
it is handed, in order. The narrative (headings, interpretation, captions) is
authored separately and passed in as ``markdown(...)``
blocks; the panel blocks here only point at keys already logged on the run by
``cv.observability`` (``Figures/<stem>`` media, ``Tables/<name>`` tables). So
the same block list always produces the same report, and rerunning overwrites the
existing one in place (idempotent by a stable per-run title).

The block-builder helpers (``markdown``, ``figure_panel``, ``table_panel``) let a
caller assemble a report without importing ``wandb_workspaces`` directly, keeping
v2 confined to this file.
"""
from __future__ import annotations

from cv import config

# Stable per-run report title, human-readable (for people, not a filename), and the
# idempotency key: reusing it means a rerun overwrites the report in place.
_TITLE = "Convergent validity of evaluation-awareness detectors ({run_id} run)"


def title_for(run_id: str) -> str:
    """The stable, human-readable report title for a run (also the idempotency key)."""
    return _TITLE.format(run_id=run_id)


def _entity() -> str:
    """The configured entity, or the account default when unset. Reports v2 types
    (Report, Runset) require a real entity string, not None/empty, so resolve it."""
    ent = config.wandb_entity()
    if ent:
        return ent
    import wandb

    return wandb.Api().default_entity


def _project_path() -> str:
    """``entity/project`` for the W&B Reports/Api path."""
    return f"{_entity()}/{config.wandb_project()}"


def _runset(run_id: str):
    """A single-run Runset scoped to ``run-<run_id>`` so every panel in a grid
    shows only this run's logged keys. Filtering is by display name, which is how
    ``WandbTracker`` names the run (``name=f"run-{run_id}"``)."""
    import wandb_workspaces.reports.v2 as wr

    return wr.Runset(
        entity=_entity(),
        project=config.wandb_project(),
        name=f"run-{run_id}",
        filters=f"displayName == 'run-{run_id}'",
    )


# --- block builders ----------------------------------------------------------
def markdown(text: str):
    """A prose block. The ONLY channel for narrative; all words in the report
    come through here, supplied by the caller."""
    import wandb_workspaces.reports.v2 as wr

    return wr.MarkdownBlock(text=text)


def figure_panel(run_id: str, fig_key: str, title: str | None = None):
    """A PanelGrid showing one of the run's logged ``Figures/<stem>`` plotly figures as a
    media panel. Pass ``title=" "`` (a single space) to suppress W&B's panel header: with
    ``None`` it falls back to printing the raw media key, and the figure carries its own
    baked-in title, so the header is redundant.

    The interactive plotly is used (not a static image) because W&B renders it as SVG,
    which stays crisp on a retina display; a ``wandb.Image`` panel is served as a ~550px
    thumbnail that upscales blurry at 2x DPR (operations 0017). The trade-off: W&B sizes
    the plotly to fill the panel and clips its bottom ~69px (the run-link + step-slider
    chrome), so the figure keeps its color legend off the bottom / the brief states the
    color mapping in prose. ``fig_key`` with no ``/`` is normalised to ``Figures/<stem>``."""
    import wandb_workspaces.reports.v2 as wr

    key = fig_key if "/" in fig_key else f"Figures/{fig_key}"
    return wr.PanelGrid(
        runsets=[_runset(run_id)],
        # hide the runset selector + run-list: a single-run report panel does not
        # need the run picker, and it is pure clutter under every figure.
        hide_run_sets=True,
        panels=[wr.MediaBrowser(
            media_keys=[key], title=title,
            # Full report width; h tuned so W&B's panel shows the title + plot + x-labels
            # of the ~547px-wide vector figure without an oversized portrait stretch.
            layout=wr.Layout(x=0, y=0, w=24, h=11),
        )],
    )


def table_panel(run_id: str, table_key: str):
    """A PanelGrid showing the run's ``Tables/<...>`` table panel. NOTE: prefer a
    markdown table in the brief (the skill builds these); this live table panel
    renders small, column-clipped, and labelled with the raw key, so it is a poor
    fit for a readable report. Kept for the run-page dashboard.

    ``table_key`` is the logged table key, e.g. ``Tables/sensitivity`` (the bare
    name ``sensitivity`` is accepted and normalised). ``WeavePanelSummaryTable``
    renders the ``wandb.Table`` logged under that key, scoped to this run by the
    grid's runset."""
    import wandb_workspaces.reports.v2 as wr

    key = table_key if table_key.startswith("Tables/") else f"Tables/{table_key}"
    return wr.PanelGrid(
        runsets=[_runset(run_id)],
        panels=[wr.WeavePanelSummaryTable(table_name=key)],
    )


# --- publish -----------------------------------------------------------------
def _find_existing(run_id: str):
    """The existing report for this run (matched by stable title), or None.
    Best-effort: any API hiccup is treated as 'no existing report', so publish
    falls back to creating a fresh one rather than failing."""
    import wandb

    want = title_for(run_id)
    try:
        for rep in wandb.Api().reports(path=_project_path()):
            if getattr(rep, "display_name", None) == want:
                return rep
    except Exception as e:  # noqa: BLE001 — listing is advisory; create-fresh on failure
        print(f"[report] existing-report lookup skipped: {type(e).__name__}: {e}")
    return None


def publish(run_id: str, blocks: list) -> str:
    """Create-or-update the run's results-brief Report from ``blocks`` and return
    its URL. Idempotent: an existing report with the stable title is overwritten
    in place (its blocks replaced); otherwise a new report is created. Renders only
    the blocks given — no prose is added here."""
    import wandb_workspaces.reports.v2 as wr

    existing = _find_existing(run_id)
    if existing is not None:
        report = wr.Report.from_url(existing.url)
        report.blocks = list(blocks)
    else:
        report = wr.Report(
            project=config.wandb_project(),
            entity=_entity(),
            title=title_for(run_id),
            blocks=list(blocks),
        )
    report.save()
    return report.url

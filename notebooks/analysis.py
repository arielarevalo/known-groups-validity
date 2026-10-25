"""Convergent-validity analysis (marimo, science-domain).

Parameterised by ``run_id`` (a stable run like ``pilot``/``full``, or an ad-hoc
one). Selection is explicit, with no default: pass ``--run-id <id>`` after ``--``,
or set ``CV_RUN`` / the profile's ``run`` field. If none is set, ``config.run_id()``
raises rather than guessing (operations 0015):

    uv run marimo edit notebooks/analysis.py -- --run-id pilot   # interactive
    uv run marimo run  notebooks/analysis.py -- --run-id pilot

The cells read as a document; all logic lives in modules (``cv.analysis`` computes
the statistics, ``cv.report`` shapes them into tidy tables, ``cv.plots`` renders
the figures). The final cell publishes the experimental scorecard and figures to
the run's W&B dashboard via the observability Tracker, so re-running refreshes the
live view in place.
"""
import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    from cv import config

    # Explicit run selection, no default: an explicit `--run-id` (after `--`) wins;
    # otherwise config resolves CV_RUN / the profile's `run`. config.run_id() raises
    # if none is set, so the notebook never publishes to a guessed run.
    rid = mo.cli_args().get("run-id")
    if rid:
        config.use_run(str(rid))
    run_id = config.run_id()
    mo.md(f"# Convergent validity of eval-awareness detectors\n### run `{run_id}`")
    return config, run_id


@app.cell
def _(config):
    import json

    import pandas as pd

    from cv import analysis, observability, plots, report

    analysis.run()  # (re)compute analysis.json for this run
    result = json.load(open(config.cv_out_dir() / "analysis" / "analysis.json"))
    return analysis, observability, pd, plots, report, result


@app.cell
def _(mo):
    mo.md(
        """
        ## Sensitivity — pooled cued d' by detector

        How strongly each detector separates cued items (the F1..F7 singles and
        their stacks) from baseline, per model, with the task-cluster interval.
        A detector with d' undefined shows the reason: too few events for a
        claim, or a silent channel.
        """
    )
    return


@app.cell
def _(pd, report, result):
    pd.DataFrame(report.sensitivity_records(result))
    return


@app.cell
def _(mo):
    mo.md(
        """
        ## Convergent matrix — pairwise Cohen's kappa

        The headline, every pair with its interval, over the cued-criterion rows
        and over all configs. A floor channel (positive rate under the
        pre-registered threshold) has no kappa; the row names it instead.
        """
    )
    return


@app.cell
def _(pd, report, result):
    pd.DataFrame(report.agreement_records(result))
    return


@app.cell
def _(mo):
    mo.md("## Per-factor aware-rate (%)")
    return


@app.cell
def _(pd, report, result):
    pd.DataFrame(report.factor_records(result))
    return


@app.cell
def _(mo):
    mo.md(
        """
        ## Dose-response — awareness by cue count (Experiment B, descriptive)

        On the stacking subset: the awareness rate per number of active cues, with
        the interval, and each detector's slope per added cue.
        """
    )
    return


@app.cell
def _(pd, report, result):
    pd.DataFrame(report.dose_records(result))
    return


@app.cell
def _(pd, report, result):
    pd.DataFrame(report.slope_records(result))
    return


@app.cell
def _(mo):
    mo.md("## Figures (regenerated from this run)")
    return


@app.cell
def _(mo, plots):
    figure_paths = plots.run()
    comparison_figures = [mo.image(p) for p in figure_paths if "compare_" in p]
    mo.vstack(comparison_figures) if comparison_figures else mo.md("_single-model run: no cross-model figures_")
    return (figure_paths,)


@app.cell
def _(mo):
    mo.md(
        """
        ## Publish to the W&B dashboard

        Write every report *piece* to this run's W&B record via the observability
        Tracker, each under a human run-page section: the scorecard (`Results/*`),
        the tidy result tables (`Tables/*`), the run config, and the figures as media
        panels (`Figures/*`). Latest-wins, so re-running this notebook refreshes the
        live dashboard in place; a no-op offline. The prose report is written
        separately over these same pieces (operations 0015).
        """
    )
    return


@app.cell
def _(analysis, figure_paths, observability, report, result, run_id):
    tracker = observability.get_tracker(run_id)
    scorecard = analysis.headline(result)
    tracker.experimental_summary(scorecard)
    tracker.tables({
        "sensitivity": report.sensitivity_records(result),
        "agreement": report.agreement_records(result),
        "factor_sweep": report.factor_records(result),
        "dose_response": report.dose_records(result),
        "dose_slope": report.slope_records(result),
    })
    tracker.config(result["meta"])
    tracker.figures(figure_paths)
    tracker.finish()
    scorecard
    return


if __name__ == "__main__":
    app.run()

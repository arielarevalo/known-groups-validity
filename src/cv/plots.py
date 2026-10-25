"""Figures for the convergent-validity analysis.

Reads results/analysis/analysis.json (the schema cv.analysis.run emits) and writes,
per figure, three artifacts to results/figures/:
  - ``<stem>.html`` -- a self-contained-ish interactive plotly figure (portable: it
    opens in any browser; plotly.js is pulled from the CDN, not W&B-locked).
  - ``<stem>.json`` -- the portable plotly spec; ``cv.observability`` reloads it and
    logs it as a ``wandb.Plotly`` so the W&B dashboard panels are interactive (and
    not a Vega/``wandb.plot`` lock-in).
  - ``<stem>.png``  -- a static raster (kaleido) for the write-up's inline embeds.

Figures, per model:
  (a) per_factor_<model>    -- d' of baseline vs EACH config, grouped by detector,
      with the cluster-bootstrap interval; F8 marked as the positive control.
  (b) dprime_cued_<model>   -- pooled d' over the cued criterion with CI (unstable
      CIs flagged).
  (c) kappa_<model>         -- cued-criterion Cohen's kappa heatmap over the full
      participants; floor channels named in the title.
  (d) calibration_<model>   -- accuracy by elicited confidence level (descriptive).
  (e) factor_sweep_<model>  -- awareness rate by single config, per detector.
  (f) dose_response_<model> -- awareness rate by cue count on the stacking subset,
      per detector, with intervals (Experiment B).
plus the cross-model comparison figures (compare_dprime_cued, compare_kappa_cued,
compare_baseline_fa), emitted only for >=2 models. The detector list comes from
the analysis record, never from a literal.

A figure is skipped when its inputs are missing; the written ``.png`` paths are
returned (the stable contract). British spelling throughout.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import plotly.graph_objects as go

from cv import config

# Stable, muted-professional qualitative palette, used per detector/model so a series
# keeps the same colour across figures.
_COLOURS = ["#3b6fb6", "#e08a3c", "#4f9d69", "#c0504d", "#7d6fb0", "#8c6d4f"]
_FONT = "Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Helvetica, Arial, sans-serif"
_AXIS = "#cbd5e1"   # tick/line grey
_GRID = "#eef1f6"   # subtle gridline
_INK = "#33425b"    # body text
_TITLE_INK = "#1f2d40"


def _load(analysis_dir: Path) -> Optional[dict[str, Any]]:
    path = analysis_dir / "analysis.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _save(fig: go.Figure, stem: str, out: Path) -> str:
    """Apply the shared house style, then write the portable trio (.html + .json +
    .png) and return the .png path."""
    fig.update_layout(
        template="plotly_white",
        font=dict(family=_FONT, size=15, color=_INK),
        # The W&B *report* embeds the INTERACTIVE plotly (Figures/* key), which W&B renders
        # as an SVG that stays crisp on a retina display -- a static .png is thumbnailed by
        # W&B (~550px) and upscales blurry at 2x DPR, so the report uses the vector figure.
        # W&B fills the report panel with the plotly at ~547px wide, so size the fonts for
        # that width: tick/body 15, a modest title 16 (a bigger title dominated the small
        # render and collided with the panel chrome). The same style serves the .html and
        # the workspace panel.
        height=400,
        title=dict(font=dict(family=_FONT, size=16, color=_TITLE_INK),
                   x=0.5, xanchor="center", y=0.98, yanchor="top"),
        margin=dict(l=78, r=30, t=46, b=82),
        colorway=_COLOURS,
        # The W&B *report* embeds the INTERACTIVE plotly (Figures/* key): vectors stay crisp
        # on a retina display, where W&B's static-image panel is thumbnailed to ~550px and
        # upscales blurry. W&B fills the panel at ~547px wide, so fonts are sized for that
        # (tick/body 15, a modest title 16). Legend along the bottom, horizontal.
        legend=dict(orientation="h", yanchor="top", y=-0.24, xanchor="center", x=0.5,
                    font=dict(size=15), bgcolor="rgba(0,0,0,0)"),
        bargap=0.30, bargroupgap=0.10,
        plot_bgcolor="white", paper_bgcolor="white",
        hoverlabel=dict(font=dict(family=_FONT, size=12), bgcolor="white", bordercolor=_AXIS),
    )
    # Subtle, modern axes (gridlines hide behind the opaque heatmap cells, so this is
    # safe across bars/lines/heatmap).
    fig.update_xaxes(showgrid=False, showline=True, linewidth=1, linecolor=_AXIS,
                     ticks="outside", tickcolor=_AXIS, ticklen=5)
    fig.update_yaxes(showgrid=True, gridcolor=_GRID, gridwidth=1, zeroline=False, showline=False)
    fig.update_traces(marker_line_width=0, selector=dict(type="bar"))
    fig.write_html(str(out / f"{stem}.html"), include_plotlyjs="cdn", full_html=True)
    fig.write_json(str(out / f"{stem}.json"))
    png = out / f"{stem}.png"
    # Export near the report's ~520px display width (not large) so W&B's downscale is gentle
    # and the fixed-px fonts stay legible; scale=2 keeps the raster crisp if shown wider.
    fig.write_image(str(png), width=660, height=400, scale=2)  # kaleido raster
    return str(png)


# --- hover ----------------------------------------------------------------------
# The point of plotly here is an interrogable mark: the headline is the bar/cell, the
# provenance (CIs, hit/FA rates, n, the asymmetric kappa conditionals) is on hover, so
# the figure carries its own supporting stats without a separate table. We pre-format
# the detail string per mark in Python (handles missing values cleanly) and pass it as
# customdata; the template just renders it.
_HOVER = "%{customdata}<extra></extra>"


def _pct(x: Optional[float]) -> str:
    return f"{x * 100:.1f}%" if x is not None else "n/a"


def _f2(x: Optional[float]) -> str:
    return f"{x:.2f}" if x is not None else "n/a"


def _dprime_hover(label: str, v: dict[str, Any]) -> str:
    ci = v.get("d_prime_ci") or [None, None]
    ci_part = f"  ·  95% CI [{_f2(ci[0])}, {_f2(ci[1])}]" if ci[0] is not None else ""
    n = v.get("n_tasks") or v.get("n_pos")
    out = (f"<b>{label}</b><br>d′ = {_f2(v.get('d_prime'))}{ci_part}"
           f"<br>hit {_pct(v.get('hit_rate'))}  ·  false-alarm {_pct(v.get('false_alarm_rate'))}"
           f"<br>n = {n} ({v.get('n_pos')} cued / {v.get('n_neg')} baseline)")
    if v.get("d_prime") is None and v.get("d_prime_undefined_reason"):
        out += f"<br><i>d′ undefined: {v['d_prime_undefined_reason']}</i>"
    if v.get("ci_unstable"):
        out += "<br><i>CI unstable (&lt;20 tasks)</i>"
    return out


def _error(v: dict[str, Any], key: str = "d_prime", ci_key: str = "d_prime_ci") -> tuple[float, float]:
    """Asymmetric error-bar arms (minus, plus) from a point and its interval."""
    x, ci = v.get(key), v.get(ci_key)
    if x is None or not ci or ci[0] is None:
        return 0.0, 0.0
    return max(0.0, x - ci[0]), max(0.0, ci[1] - x)


def _per_factor_figure(model: str, sens: dict[str, Any], out: Path) -> Optional[str]:
    per_factor = sens.get("per_factor", {})
    dets = [d for d, v in per_factor.items() if v]
    if not dets:
        return None
    factors_set: set[str] = set()
    for d in dets:
        factors_set.update(it["factor"] for it in per_factor[d])
    facs = sorted(factors_set)
    if not facs:
        return None
    control = {it["factor"] for d in dets for it in per_factor[d] if it.get("control")}
    xlabels = [f"{f} (control)" if f in control else f for f in facs]
    fig = go.Figure()
    for k, d in enumerate(dets):
        rec = {it["factor"]: it for it in per_factor[d] if it.get("d_prime") is not None}
        ys = [rec[f]["d_prime"] if f in rec else None for f in facs]
        arms = [_error(rec[f]) if f in rec else (0.0, 0.0) for f in facs]
        cdata = [_dprime_hover(f"{d.upper()} · {f}", rec[f]) if f in rec else "" for f in facs]
        fig.add_bar(name=d, x=xlabels, y=ys, marker_color=_COLOURS[k % len(_COLOURS)],
                    error_y=dict(type="data", symmetric=False, array=[a[1] for a in arms],
                                 arrayminus=[a[0] for a in arms], thickness=1.2),
                    customdata=cdata, hovertemplate=_HOVER)
    fig.add_hline(y=0.0, line_color="grey", line_width=0.8)
    fig.update_layout(
        barmode="group",
        title=f"Per-factor sensitivity (descriptive), {model}",
        yaxis_title="d'  (baseline vs config)",
    )
    return _save(fig, f"per_factor_{model}", out)


def _dprime_cued_figure(model: str, sens: dict[str, Any], out: Path) -> Optional[str]:
    pooled = sens.get("pooled_cued", {})
    items = [(d, v) for d, v in pooled.items() if v and v.get("d_prime") is not None]
    if not items:
        return None
    items.sort(key=lambda kv: kv[1]["d_prime"], reverse=True)
    dets = [d for d, _ in items]
    vals = [v["d_prime"] for _, v in items]
    lo, hi, unstable, cdata = [], [], False, []
    for d, v in items:
        minus, plus = _error(v)
        lo.append(minus)
        hi.append(plus)
        unstable = unstable or bool(v.get("ci_unstable"))
        cdata.append(_dprime_hover(f"{d.upper()} · {model}", v))
    fig = go.Figure(go.Bar(
        x=dets, y=vals, marker_color=_COLOURS[0],
        error_y=dict(type="data", symmetric=False, array=hi, arrayminus=lo, thickness=1.5),
        customdata=cdata, hovertemplate=_HOVER,
    ))
    fig.add_hline(y=0.0, line_color="grey", line_width=0.8)
    suffix = "  (CIs unstable: <20 tasks)" if unstable else ""
    fig.update_layout(
        title=f"Pooled cued sensitivity, {model}{suffix}",
        yaxis_title="d'  (baseline vs cued: F1..F7 singles and stacks)",
    )
    return _save(fig, f"dprime_cued_{model}", out)


def _kappa_figure(model: str, agr: dict[str, Any], out: Path) -> Optional[str]:
    block = agr.get("criterion_cued") or agr.get("all_configs")
    if not block:
        return None
    floor = block.get("floor_channels") or []
    dets = block.get("detectors", [])
    kappa = block.get("kappa", {})
    if not dets:
        return None
    kci = block.get("kappa_ci", {})
    cond = block.get("cond_prob_b_given_a", {})  # cond[a][b] = P(b fires | a fires)
    n = len(dets)
    z: list[list[Optional[float]]] = [[None] * n for _ in range(n)]
    text: list[list[str]] = [[""] * n for _ in range(n)]
    cd: list[list[str]] = [[""] * n for _ in range(n)]
    any_val = False
    for i, a in enumerate(dets):
        for j, b in enumerate(dets):
            v = kappa.get(a, {}).get(b)
            if v is not None:
                z[i][j] = v
                text[i][j] = f"{v:.2f}"
                any_val = True
                ci = (kci.get(a, {}) or {}).get(b) or [None, None]
                cd[i][j] = (f"<b>{a.upper()} ∧ {b.upper()}</b><br>κ = {_f2(v)}"
                            + (f"  ·  95% CI [{_f2(ci[0])}, {_f2(ci[1])}]" if ci[0] is not None else "")
                            + f"<br>P({b.upper()} fires | {a.upper()} fires) = {_pct((cond.get(a, {}) or {}).get(b))}")
    if not any_val:
        return None
    # rows top-to-bottom as listed (reverse y so the first detector is on top).
    fig = go.Figure(go.Heatmap(
        z=z, x=dets, y=dets, zmin=-1, zmax=1, colorscale="RdBu_r",
        text=text, texttemplate="%{text}", textfont=dict(size=11),
        colorbar=dict(title="Cohen's κ"), hoverongaps=False,
        customdata=cd, hovertemplate=_HOVER,
    ))
    fig.update_layout(
        title=f"Detector agreement (cued-criterion rows), {model}"
              + (f"  ·  floor channel out: {', '.join(d.upper() for d in floor)}" if floor else ""),
        yaxis=dict(autorange="reversed", scaleanchor="x"), xaxis=dict(side="bottom"),
    )
    return _save(fig, f"kappa_{model}", out)


def _calibration_figure(model: str, cal: dict[str, Any], out: Path) -> Optional[str]:
    have = {d: v for d, v in cal.items() if v and v.get("accuracy_by_level")}
    if not have:
        return None
    fig = go.Figure()
    for k, (det, v) in enumerate(have.items()):
        xs = [lv["level"] for lv in v["accuracy_by_level"]]
        ys = [lv["accuracy"] for lv in v["accuracy_by_level"]]
        rho = v.get("monotonicity_rho")
        lab = f"{det}" + (f" (rho={rho:.2f})" if rho is not None else "")
        fig.add_scatter(x=xs, y=ys, mode="lines+markers", name=lab,
                        line=dict(color=_COLOURS[k % len(_COLOURS)]))
    fig.update_layout(
        title=f"Confidence vs accuracy (descriptive), {model}",
        xaxis=dict(title="elicited confidence level (1-4)", tickvals=[1, 2, 3, 4], range=[0.5, 4.5]),
        yaxis=dict(title="accuracy on cued criterion", range=[0, 1.02]),
    )
    return _save(fig, f"calibration_{model}", out)


def _factor_sweep_figure(model: str, sweep: dict[str, Any], out: Path) -> Optional[str]:
    configs = sweep.get("configs", [])
    by_det = sweep.get("by_detector", {})
    if not configs or not by_det:
        return None
    fig = go.Figure()
    plotted = False
    for k, (det, per_cfg) in enumerate(by_det.items()):
        ys = [per_cfg.get(c, {}).get("aware_rate") for c in configs]
        if all(y is None for y in ys):
            continue
        cdata = []
        for c in configs:
            r = per_cfg.get(c, {})
            cls = f"  ·  {r.get('factor_class')}" if r.get("factor_class") else ""
            cdata.append(f"<b>{det.upper()} · {c}</b><br>awareness rate = {_pct(r.get('aware_rate'))}"
                         f"<br>n = {r.get('n')}{cls}")
        fig.add_scatter(x=configs, y=ys, mode="lines+markers", name=det,
                        line=dict(color=_COLOURS[k % len(_COLOURS)]),
                        customdata=cdata, hovertemplate=_HOVER)
        plotted = True
    if not plotted:
        return None
    fig.update_layout(
        title=f"Factor sweep, {model}",
        xaxis=dict(title="", tickangle=-45),
        yaxis=dict(title="awareness rate", range=[-0.02, 1.02]),
    )
    return _save(fig, f"factor_sweep_{model}", out)


def _dose_response_figure(model: str, dr: dict[str, Any], out: Path) -> Optional[str]:
    """Awareness rate by number of active cues on the stacking subset, one line per
    detector, with the cluster-bootstrap interval (Experiment B, descriptive)."""
    by_det = (dr or {}).get("by_detector") or {}
    if not by_det:
        return None
    fig = go.Figure()
    plotted = False
    for k, (det, v) in enumerate(by_det.items()):
        by_k = (v or {}).get("by_k") or {}
        if not by_k:
            continue
        ks = sorted(by_k, key=int)
        xs = [int(x) for x in ks]
        ys = [by_k[x]["aware_rate"] for x in ks]
        arms = [_error(by_k[x], "aware_rate", "aware_rate_ci") for x in ks]
        cdata = []
        for x in ks:
            e = by_k[x]
            ci = e.get("aware_rate_ci") or [None, None]
            d_part = f"<br>d′ vs k=0 = {_f2(e.get('d_prime'))}" if int(x) > 0 else ""
            cdata.append(f"<b>{det.upper()} · k = {x}</b><br>awareness rate = {_pct(e.get('aware_rate'))}"
                         + (f"  ·  95% CI [{_pct(ci[0])}, {_pct(ci[1])}]" if ci[0] is not None else "")
                         + d_part + f"<br>n = {e.get('n')} over {e.get('n_configs')} config(s)")
        slope = v.get("slope_per_cue")
        lab = det + (f" (slope {slope:+.3f}/cue)" if slope is not None else "")
        fig.add_scatter(x=xs, y=ys, mode="lines+markers", name=lab,
                        line=dict(color=_COLOURS[k % len(_COLOURS)]),
                        error_y=dict(type="data", symmetric=False, array=[a[1] for a in arms],
                                     arrayminus=[a[0] for a in arms], thickness=1.2),
                        customdata=cdata, hovertemplate=_HOVER)
        plotted = True
    if not plotted:
        return None
    suffix = "  (CIs unstable: <20 tasks)" if dr.get("ci_unstable") else ""
    fig.update_layout(
        title=f"Dose-response over stacked cues (descriptive), {model}{suffix}",
        xaxis=dict(title="active cues (k)", dtick=1),
        yaxis=dict(title="awareness rate", range=[-0.02, 1.02]),
    )
    return _save(fig, f"dose_response_{model}", out)


def _detector_order(analysis: dict[str, Any]) -> list[str]:
    """The detectors the analysis scored, in the report's display order."""
    from cv import report

    return report.detectors(analysis)


def _compare_dprime_figure(analysis: dict[str, Any], out: Path) -> Optional[str]:
    """Pooled cued d' per detector, one bar group per model: the cross-model
    replication of the cue-sensitivity ordering (only emitted for >=2 models)."""
    models = analysis.get("meta", {}).get("models", [])
    if len(models) < 2:
        return None
    dets = _detector_order(analysis)
    xlabels = [d.upper() for d in dets]
    fig = go.Figure()
    any_data = False
    for k, m in enumerate(models):
        pooled = (analysis.get("sensitivity", {}).get(m) or {}).get("pooled_cued", {})
        vals, lo, hi, cdata = [], [], [], []
        for d in dets:
            v = pooled.get(d) or {}
            dp = v.get("d_prime")
            if dp is None:
                vals.append(None); lo.append(0.0); hi.append(0.0); cdata.append(""); continue
            any_data = True
            vals.append(dp)
            ci = v.get("d_prime_ci") or [None, None]
            lo.append(max(0.0, dp - ci[0]) if ci[0] is not None else 0.0)
            hi.append(max(0.0, ci[1] - dp) if ci[1] is not None else 0.0)
            cdata.append(_dprime_hover(f"{d.upper()} · {m}", v))
        fig.add_bar(name=m, x=xlabels, y=vals, marker_color=_COLOURS[k % len(_COLOURS)],
                    error_y=dict(type="data", symmetric=False, array=hi, arrayminus=lo, thickness=1.5),
                    customdata=cdata, hovertemplate=_HOVER)
    if not any_data:
        return None
    fig.add_hline(y=0.0, line_color="grey", line_width=0.8)
    fig.update_layout(
        barmode="group", title="Pooled cued sensitivity by detector, across models",
        yaxis_title="d'  (baseline vs cued)",
    )
    return _save(fig, "compare_dprime_cued", out)


def _compare_kappa_figure(analysis: dict[str, Any], out: Path) -> Optional[str]:
    """Pairwise cued-criterion Cohen's kappa per model: the convergent-validity
    headline (only emitted for >=2 models). A pair with a floor channel has no
    bar in that model."""
    from cv import report

    models = analysis.get("meta", {}).get("models", [])
    if len(models) < 2:
        return None
    pairs = report.pairs(_detector_order(analysis))
    labels = [f"{a.upper()}-{b.upper()}" for a, b in pairs]
    fig = go.Figure()
    any_data = False
    for k, m in enumerate(models):
        block = (analysis.get("agreement", {}).get(m) or {}).get("criterion_cued") or {}
        kappa = block.get("kappa", {})
        kci = block.get("kappa_ci", {})
        cond = block.get("cond_prob_b_given_a", {})  # cond[a][b] = P(b fires | a fires)
        vals, lo, hi, cdata = [], [], [], []
        for a, b in pairs:
            v = (kappa.get(a, {}) or {}).get(b)
            if v is None:
                vals.append(None); lo.append(0.0); hi.append(0.0); cdata.append(""); continue
            any_data = True
            vals.append(v)
            ci = (kci.get(a, {}) or {}).get(b) or [None, None]
            lo.append(max(0.0, v - ci[0]) if ci[0] is not None else 0.0)
            hi.append(max(0.0, ci[1] - v) if ci[1] is not None else 0.0)
            c_ab = (cond.get(a, {}) or {}).get(b)  # P(b | a)
            c_ba = (cond.get(b, {}) or {}).get(a)  # P(a | b)
            cdata.append(
                f"<b>{a.upper()}-{b.upper()} · {m}</b><br>κ = {_f2(v)}  ·  95% CI [{_f2(ci[0])}, {_f2(ci[1])}]"
                f"<br>P({b.upper()} fires | {a.upper()} fires) = {_pct(c_ab)}"
                f"<br>P({a.upper()} fires | {b.upper()} fires) = {_pct(c_ba)}")
        fig.add_bar(name=m, x=labels, y=vals, marker_color=_COLOURS[k % len(_COLOURS)],
                    error_y=dict(type="data", symmetric=False, array=hi, arrayminus=lo, thickness=1.5),
                    customdata=cdata, hovertemplate=_HOVER)
    if not any_data:
        return None
    fig.add_hline(y=0.0, line_color="grey", line_width=0.8)
    fig.update_layout(
        barmode="group", title="Pairwise detector agreement, across models",
        yaxis_title="Cohen's kappa  (cued criterion)",
    )
    return _save(fig, "compare_kappa_cued", out)


def _compare_baseline_fa_figure(analysis: dict[str, Any], out: Path) -> Optional[str]:
    """Baseline (cue-free) false-alarm rate per detector, one group per model:
    the resting 'paranoia' that differs by model (only emitted for >=2 models)."""
    models = analysis.get("meta", {}).get("models", [])
    if len(models) < 2:
        return None
    dets = _detector_order(analysis)
    xlabels = [d.upper() for d in dets]
    fig = go.Figure()
    any_data = False
    for k, m in enumerate(models):
        by_det = ((analysis.get("factor_sweep", {}).get(m) or {}).get("by_detector") or {})
        ys, cdata = [], []
        for d in dets:
            r = (by_det.get(d, {}) or {}).get("baseline", {})
            ar = r.get("aware_rate")
            ys.append(ar * 100 if ar is not None else None)
            any_data = any_data or ar is not None
            cdata.append(f"<b>{d.upper()} · {m}</b><br>baseline false-alarm = {_pct(ar)}"
                         f"<br>on {r.get('n')} cue-free items")
        fig.add_bar(name=m, x=xlabels, y=ys, marker_color=_COLOURS[k % len(_COLOURS)],
                    customdata=cdata, hovertemplate=_HOVER)
    if not any_data:
        return None
    fig.update_layout(
        barmode="group", title="Resting false-alarm rate on cue-free items, across models",
        yaxis_title="baseline false-alarm rate (%)",
    )
    return _save(fig, "compare_baseline_fa", out)


def run(analysis_dir: Path | str | None = None, out_dir: Path | str | None = None) -> list[str]:
    analysis_dir = Path(analysis_dir) if analysis_dir else (config.cv_out_dir() / "analysis")
    out_dir = Path(out_dir) if out_dir else (config.cv_out_dir() / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    analysis = _load(analysis_dir)
    if analysis is None:
        return []
    written: list[str] = []
    for model in analysis.get("meta", {}).get("models", []):
        sens = analysis.get("sensitivity", {}).get(model)
        agr = analysis.get("agreement", {}).get(model)
        cal = analysis.get("calibration", {}).get(model)
        sweep = analysis.get("factor_sweep", {}).get(model)
        dr = analysis.get("dose_response", {}).get(model)
        for path in (
            _per_factor_figure(model, sens, out_dir) if sens else None,
            _dprime_cued_figure(model, sens, out_dir) if sens else None,
            _kappa_figure(model, agr, out_dir) if agr else None,
            _calibration_figure(model, cal, out_dir) if cal else None,
            _factor_sweep_figure(model, sweep, out_dir) if sweep else None,
            _dose_response_figure(model, dr, out_dir) if dr else None,
        ):
            if path:
                written.append(path)
    # Cross-model comparison figures (the two-model headline); no-ops for <2 models.
    for path in (
        _compare_dprime_figure(analysis, out_dir),
        _compare_kappa_figure(analysis, out_dir),
        _compare_baseline_fa_figure(analysis, out_dir),
    ):
        if path:
            written.append(path)
    return written

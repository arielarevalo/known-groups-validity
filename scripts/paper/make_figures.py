"""Build the three figures of the paper as vector PDFs from the frozen run outputs.

Run from anywhere (paths resolve from this file):

    uv run --no-project --with matplotlib==3.9.2 --with numpy python scripts/paper/make_figures.py

Inputs (nothing else is read):
    reports/full/analysis.json
    reports/full/fit_gemma-4-31b.json   (opened to confirm it exists; no main-text figure draws it)
    reports/full/fit_qwen3.6-27b.json
    build/paper/numbers.tex             (from scripts/paper/make_numbers.py; for kappamax, kappafrac and labels)

Outputs: build/paper/figures/fig1.pdf, fig2.pdf, fig3.pdf. Fonts are embedded (Open Sans, from
scripts/paper/fonts, SIL Open Font License).
With --preview DIR, a 200 dpi PNG of each figure is written there as well.

Every drawn or printed number comes from those files at run time. Where a printed number also
exists in the JSON, the script asserts that the two agree.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from statistics import NormalDist

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.text  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REPORTS = ROOT / "reports" / "full"
FIGDIR = ROOT / "build" / "paper" / "figures"
FONTDIR = HERE / "fonts"
NUMBERS = ROOT / "build" / "paper" / "numbers.tex"

# Palette (ink and secondary match the paper template's body and secondary colours).
INK = "#111318"
SECONDARY = "#454754"
CONTEXT = "#BDBDBD"
GREY = "#8C8C8C"
COL = {
    "sr": "#0072B2",
    "tp": "#009E73",
    "cot": GREY,
    "probe": "#D55E00",
    "logit": "#CC79A7",
}
NAME = {
    "sr": "self-report",
    "tp": "third-person",
    "cot": "CoT judge",
    "probe": "activation probe",
    "logit": "logit monitor",
}
MODEL_KEY = {"gemma": "gemma-4-31b", "qwen": "qwen3.6-27b"}
CM = 1 / 2.54
TEXT_WIDTH_CM = 16.6


# ----------------------------------------------------------------------------- data access


def load_json(path: Path) -> dict:
    with path.open() as fh:
        return json.load(fh)


def load_numbers(path: Path) -> dict[str, str]:
    """Parse \\cvset{key}{value} lines; thousands separators {,} are stripped first."""
    out: dict[str, str] = {}
    pat = re.compile(r"^\\cvset\{([^}]*)\}\{(.*?)\}\s*(?:%.*)?$")
    for line in path.read_text().splitlines():
        m = pat.match(line.replace("{,}", ""))
        if m:
            out[m.group(1)] = m.group(2)
    if not out:
        raise RuntimeError(f"no \\cvset lines parsed from {path}")
    return out


class Data:
    def __init__(self) -> None:
        self.an = load_json(REPORTS / "analysis.json")
        self.fit_qwen = load_json(REPORTS / "fit_qwen3.6-27b.json")
        load_json(REPORTS / "fit_gemma-4-31b.json")  # present and readable; not drawn in the main text
        self.num = load_numbers(NUMBERS)

    def key(self, k: str) -> str:
        if k not in self.num:
            raise KeyError(f"numbers.tex has no key {k!r}")
        return self.num[k]

    def key_float(self, k: str) -> float:
        return float(self.key(k))

    def agree(self, k: str, value: float, tol: float) -> None:
        """Assert that a rounded key in numbers.tex matches the unrounded value from the JSON."""
        if abs(self.key_float(k) - value) > tol:
            raise AssertionError(f"{k}: numbers.tex has {self.key(k)} but the JSON gives {value}")

    def pooled(self, model: str, det: str) -> dict:
        return self.an["sensitivity"][MODEL_KEY[model]]["pooled_cued"][det]


# ----------------------------------------------------------------------------- style


def setup_style() -> None:
    for f in sorted(FONTDIR.glob("*.ttf")):
        font_manager.fontManager.addfont(str(f))
    plt.rcParams.update(
        {
            "font.family": "Open Sans",
            "font.size": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "text.color": INK,
            "axes.edgecolor": SECONDARY,
            "axes.labelcolor": INK,
            "axes.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "xtick.color": SECONDARY,
            "ytick.color": SECONDARY,
            "xtick.labelcolor": INK,
            "ytick.labelcolor": INK,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "xtick.major.pad": 2.5,
            "ytick.major.pad": 2.5,
            "axes.labelsize": 9,
            "axes.labelpad": 3,
            "figure.dpi": 100,
            "savefig.dpi": 200,
            "lines.solid_capstyle": "butt",
        }
    )


def new_figure(w_cm: float, h_cm: float) -> plt.Figure:
    return plt.figure(figsize=(w_cm * CM, h_cm * CM))


def add_axes_cm(fig: plt.Figure, left: float, bottom: float, width: float, height: float, **kw):
    """Place an axes by its lower-left corner and size in cm (figure coordinates, no rescaling)."""
    W, H = fig.get_size_inches() / CM
    return fig.add_axes([left / W, bottom / H, width / W, height / H], **kw)


def assert_min_font(fig: plt.Figure, name: str, minimum: float = 7.0) -> None:
    """Every visible piece of text must be at least `minimum` pt at final size."""
    for t in fig.findobj(matplotlib.text.Text):
        if t.get_visible() and t.get_text().strip() and t.get_fontsize() < minimum:
            raise AssertionError(f"{name}: text {t.get_text()!r} is {t.get_fontsize()} pt, below {minimum} pt")


def save(fig: plt.Figure, name: str, preview: Path | None) -> None:
    assert_min_font(fig, name)
    FIGDIR.mkdir(parents=True, exist_ok=True)
    pdf = FIGDIR / f"{name}.pdf"
    fig.savefig(pdf, metadata={"Creator": "scripts/paper/make_figures.py", "CreationDate": None})
    if preview is not None:
        preview.mkdir(parents=True, exist_ok=True)
        fig.savefig(preview / f"{name}.png", dpi=200)
    plt.close(fig)
    print(f"wrote {pdf.relative_to(ROOT)}")


# ----------------------------------------------------------------------------- figure 1


def figure1(d: Data, preview: Path | None) -> None:
    nd = NormalDist()
    W, H = TEXT_WIDTH_CM, 8.4
    S = 6.6  # square axes side in cm
    fig = new_figure(W, H)
    left = [1.45, 1.45 + S + 1.35]
    bottom = 1.15
    panels = [
        ("gemma", "Gemma 4 31B", ["sr", "tp", "cot"]),
        ("qwen", "Qwen 3.6 27B", ["sr", "tp", "cot", "probe", "logit"]),
    ]
    open_markers = {"probe", "logit"}  # thresholds fitted to the cue label, out of fold
    labels = {
        "gemma": {"sr": (23, 55), "tp": (40, 77), "cot": (8, 14)},
        "qwen": {"sr": (14, 48), "tp": (21, 77), "probe": (21, 59), "logit": (53, 71), "cot": (8, 14)},
    }
    leaders = {("gemma", "cot"), ("qwen", "cot")}

    for (model, header, dets), x0 in zip(panels, left):
        ax = add_axes_cm(fig, x0, bottom, S, S)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_aspect("equal", adjustable="box")
        ticks = list(range(0, 101, 20))
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xlabel("False-alarm rate on baseline rollouts (%)")
        if model == "gemma":
            ax.set_ylabel("Hit rate on cued rollouts (%)")
        ax.text(0, 1.035, header, transform=ax.transAxes, ha="left", va="bottom", fontsize=9,
                fontweight="bold", color=INK)

        # Reference marks: chance diagonal and iso-sensitivity curves H = Phi(d' + Phi^-1(F)).
        ax.plot([0, 100], [0, 100], ls=(0, (4, 3)), lw=0.6, color=CONTEXT, zorder=1)
        F = np.linspace(0.001, 0.999, 999)
        zF = np.array([nd.inv_cdf(f) for f in F])
        z90 = nd.inv_cdf(0.9)
        for dp in (0.5, 1.0, 1.5):
            Hc = np.array([nd.cdf(dp + z) for z in zF])
            ax.plot(F * 100, Hc * 100, lw=0.6, color=CONTEXT, zorder=1)
            f_at_90 = nd.cdf(z90 - dp)
            ax.text(f_at_90 * 100 - 1.5, 91, f"d′ = {dp:.1f}", ha="right", va="bottom",
                    fontsize=7, color=SECONDARY, zorder=2)

        # What good and bad look like here: the top-left corner separates cued from baseline
        # rollouts best, and the dashed diagonal is no separation at all.
        ax.annotate("better separation", xy=(1.2, 99.0), xytext=(6.5, 98.0), ha="left", va="center",
                    fontsize=7.5, color=SECONDARY, zorder=2,
                    arrowprops={"arrowstyle": "-|>", "lw": 0.6, "color": SECONDARY, "shrinkA": 1,
                                "shrinkB": 0, "mutation_scale": 6})
        ax.text(48.5, 41.5, "no separation (d′ = 0)", rotation=45, rotation_mode="anchor", ha="center",
                va="center", fontsize=7.5, color=SECONDARY, zorder=2)

        # Marks: point, cross of 95% intervals, direct label.
        for det in dets:
            r = d.pooled(model, det)
            hit, fa = 100 * r["hit_rate"], 100 * r["false_alarm_rate"]
            hlo, hhi = (100 * v for v in r["hit_rate_ci"])
            flo, fhi = (100 * v for v in r["false_alarm_rate_ci"])
            d.agree(f"{model}/{det}/hit", hit, 0.0501)
            d.agree(f"{model}/{det}/fa", fa, 0.0501)
            c = COL[det]
            ax.plot([flo, fhi], [hit, hit], color=c, lw=0.8, zorder=3, clip_on=False)
            ax.plot([fa, fa], [hlo, hhi], color=c, lw=0.8, zorder=3, clip_on=False)
            if det in open_markers:
                ax.plot(fa, hit, "o", ms=6, mfc="white", mec=c, mew=1.2, zorder=4, clip_on=False)
            else:
                ax.plot(fa, hit, "o", ms=6, mfc=c, mec=c, mew=0, zorder=4, clip_on=False)
            lx, ly = labels[model][det]
            if (model, det) in leaders:
                ax.plot([lx - 0.8, fa], [ly, hit], color=c, lw=0.5, zorder=2, clip_on=False)
            # White box behind the label keeps a grey curve from running through its letters.
            ax.text(lx, ly, NAME[det], ha="left", va="center", fontsize=8.5, color=c, zorder=3.5,
                    bbox={"boxstyle": "square,pad=0.08", "fc": "white", "ec": "none"})
    save(fig, "fig1", preview)


# ----------------------------------------------------------------------------- figure 2


def minus(s: str) -> str:
    """Typeset a negative number from numbers.tex with a true minus sign."""
    return s.replace("-", "\u2212")


def figure2(d: Data, preview: Path | None) -> None:
    W, H = TEXT_WIDTH_CM, 7.0
    left_m, plot_w = 5.2, 8.8  # right margin: W - left_m - plot_w = 2.6 cm for the per cent column
    top_m, bottom_m = 0.55, 0.95
    ax_h = H - top_m - bottom_m
    fig = new_figure(W, H)
    ax = add_axes_cm(fig, left_m, bottom_m, plot_w, ax_h)

    groups = [
        ("Without the probe", INK, [("gemma", "sr", "tp", "Gemma: self-report and third-person"),
                                    ("qwen", "sr", "tp", "Qwen: self-report and third-person"),
                                    ("qwen", "sr", "logit", "Qwen: self-report and logit monitor"),
                                    ("qwen", "tp", "logit", "Qwen: third-person and logit monitor")]),
        ("With the probe", COL["probe"], [("qwen", "tp", "probe", "Qwen: third-person and probe"),
                                         ("qwen", "sr", "probe", "Qwen: self-report and probe"),
                                         ("qwen", "probe", "logit", "Qwen: probe and logit monitor")]),
    ]
    # Vertical layout in cm from the top of the axes (y data unit = 1 cm, axis inverted).
    pitch = 0.7  # row pitch; the group gap adds 0.35 cm: rule_below_row + label_below_rule + label_to_row - pitch
    label_to_row, rule_below_row, label_below_rule = 0.45, 0.33, 0.27
    entries: list[tuple] = []  # (kind, y, payload)
    rules: list[float] = []
    last_row_y = 0.0
    for gi, (gname, gcol, rows) in enumerate(groups):
        if gi == 0:
            label_y = 0.10
        else:
            rule_y = last_row_y + rule_below_row
            rules.append(rule_y)
            label_y = rule_y + label_below_rule
        entries.append(("group", label_y, gname))
        row_y = label_y + label_to_row
        for ri, row in enumerate(rows):
            if ri > 0:
                row_y += pitch
            entries.append(("row", row_y, (gcol, row)))
            last_row_y = row_y
    if last_row_y + 0.2 > ax_h + 1e-6:
        raise RuntimeError(f"figure 2 rows need {last_row_y + 0.2:.2f} cm but only {ax_h:.2f} cm are available")
    x_lo, x_hi = -0.1, 1.0
    ax.set_ylim(ax_h, 0)
    ax.set_xlim(x_lo, x_hi)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1.0"])
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Cohen\u2019s \u03ba")

    tr = blended_transform_factory(ax.transAxes, ax.transData)
    ax.plot([0, 0], [entries[0][1] + 0.2, ax_h], color=CONTEXT, lw=0.6, zorder=0)  # kappa = 0

    for kind, ey, payload in entries:
        if kind == "group":
            ax.text(-0.02, ey, payload, transform=tr, ha="right", va="center", fontsize=8,
                    fontweight="bold", color=SECONDARY, clip_on=False)
            continue
        gcol, (model, a, b, label) = payload
        pair = f"{model}/{a}-{b}"
        # The within-configuration statistics exist only as numbers.tex keys (computed from merged.jsonl).
        kw, kwlo, kwhi = (d.key_float(f"{pair}/kappawithin{s}") for s in ("", "lo", "hi"))
        kwmax = d.key_float(f"{pair}/kappawithinmax")
        frac, fraclo, frachi = (d.key(f"{pair}/kappawithinfrac{s}") for s in ("", "lo", "hi"))
        ax.plot([kw, kwmax], [ey, ey], color=gcol, alpha=0.4, lw=1.0, zorder=2, clip_on=False)
        ax.plot([kwlo, kwhi], [ey, ey], color=gcol, lw=2.2, zorder=3, clip_on=False)
        ax.plot(kw, ey, "o", ms=6, color=gcol, mew=0, zorder=4, clip_on=False)
        ax.plot(kwmax, ey, marker="|", ms=10, mew=1.2, color=INK, zorder=6, clip_on=False)
        ax.text(-0.02, ey, label, transform=tr, ha="right", va="center", fontsize=8, color=INK,
                clip_on=False)
        ax.text(1.02, ey, minus(f"{frac}% [{fraclo}, {frachi}]"), transform=tr, ha="left", va="center",
                fontsize=8.5, color=gcol, clip_on=False)
    ax.text(1.02, entries[0][1], r"$\kappa/\kappa_{\mathrm{max}}$", transform=tr, ha="left", va="center", fontsize=8,
            fontweight="bold", color=SECONDARY, clip_on=False)

    # What good and bad look like here, written in the empty space right of the ticks.
    rows_y = [e[1] for e in entries if e[0] == "row"]
    ax.text(0.99, (rows_y[1] + rows_y[2]) / 2, "dot near its tick:\nalmost all the agreement\nthe positive rates allow",
            ha="right", va="center", fontsize=7.5, color=SECONDARY, linespacing=1.15)
    ax.text(0.99, (rows_y[4] + rows_y[5]) / 2, "dot at or below 0:\nno agreement beyond chance",
            ha="right", va="center", fontsize=7.5, color=COL["probe"], linespacing=1.15)

    for ry in rules:
        ax.plot([-left_m / plot_w, 1.0 + (W - left_m - plot_w) / plot_w - 0.02], [ry, ry], transform=tr,
                color=CONTEXT, lw=0.4, clip_on=False, zorder=1)

    # One-line key above the plotting region, right-aligned at the right end of the axes. It is
    # built right to left from measured text extents, so the spacing stays exact: the ceiling label
    # (kappa with a 7 pt "max" subscript), the ceiling tick, the label for the filled dot, and the dot
    # with its interval bar.
    key_y = -0.30  # cm above the axes top (data y is inverted, so negative is up)
    fig.canvas.draw()
    ren = fig.canvas.get_renderer()
    inv = ax.transAxes.inverted()

    def fx(cm: float) -> float:  # a length in cm as a fraction of the axes width
        return cm / plot_w

    def left_edge(t) -> float:
        return inv.transform((t.get_window_extent(ren).x0, 0))[0]

    def put(x_right: float, s: str, size: float, dy: float = 0.0):
        t = ax.text(x_right, key_y + dy, s, transform=tr, ha="right", va="center", fontsize=size,
                    color=INK, clip_on=False)
        return left_edge(t)

    x = put(1.0, "max", 7, dy=0.045)
    x = put(x, "\u03ba", 8)
    x -= fx(0.12)
    ax.plot([x], [key_y], marker="|", ms=10, mew=1.2, color=INK, transform=tr, clip_on=False, zorder=5)
    x -= fx(0.50)
    x = put(x, "stratified \u03ba, 95% interval", 8)
    x -= fx(0.12)
    ax.plot([x - fx(0.52), x], [key_y, key_y], transform=tr, color=INK, lw=2.2, clip_on=False)
    ax.plot([x - fx(0.26)], [key_y], "o", ms=6, color=INK, mew=0, transform=tr, clip_on=False)
    save(fig, "fig2", preview)


# ----------------------------------------------------------------------------- figure 3


def beeswarm(xs_cm: list[float], radius_cm: float, fixed: list[tuple[float, float, float]],
             half_band_cm: float, gap_cm: float = 0.02) -> list[float]:
    """Vertical offsets (cm) for dots at horizontal positions xs_cm, packed in physical units.

    `fixed` holds (x, y, radius) obstacles that the dots must avoid.
    """
    placed: list[tuple[float, float, float]] = list(fixed)
    ys = [0.0] * len(xs_cm)
    order = sorted(range(len(xs_cm)), key=lambda i: xs_cm[i])
    step = 0.01
    for i in order:
        x = xs_cm[i]
        candidates = [0.0]
        k = 1
        while k * step <= half_band_cm:
            candidates += [k * step, -k * step]
            k += 1
        chosen = None
        for cy in candidates:
            if all((x - px) ** 2 + (cy - py) ** 2 >= (radius_cm + pr + gap_cm) ** 2 for px, py, pr in placed):
                chosen = cy
                break
        if chosen is None:
            raise RuntimeError("beeswarm: no free position inside the band")
        ys[i] = chosen
        placed.append((x, chosen, radius_cm))
    return ys


def figure3(d: Data, preview: Path | None) -> None:
    W, H = TEXT_WIDTH_CM, 4.2
    left_m, right_m, bottom_m, top_m = 0.55, 0.55, 0.95, 0.08
    plot_w = W - left_m - right_m
    ax_h = H - bottom_m - top_m
    fig = new_figure(W, H)
    ax = add_axes_cm(fig, left_m, bottom_m, plot_w, ax_h)
    ax.set_xlim(0.5, 1.0)
    ax.set_ylim(0, ax_h)  # y data unit = 1 cm
    ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_xticklabels(["0.5", "0.6", "0.7", "0.8", "0.9", "1.0"])
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Out-of-fold AUROC, cued against baseline prompts (Qwen 3.6 27B)")

    pr = d.fit_qwen["probe"]
    null = list(pr["random_direction_null"]["auroc_out_of_fold"])
    probe = pr["auroc_out_of_fold"]
    cued = pr["cued_direction_cross_check"]["auroc_out_of_fold"]
    logistic = pr["logistic_cross_check"]["auroc_out_of_fold"]
    null_mean = float(np.mean(null))
    n_null = len(null)
    above = sum(v > probe for v in null)

    # Checks against the keys the prose cites.
    d.agree("qwen/probe/auroc", probe, 0.00051)
    d.agree("qwen/cued/auroc", cued, 0.00051)
    d.agree("qwen/logistic/auroc", logistic, 0.00051)
    d.agree("qwen/null/mean", null_mean, 0.00051)
    d.agree("qwen/null/mean", pr["random_direction_null"]["mean"], 0.00051)
    d.agree("qwen/null/n", n_null, 0)
    d.agree("qwen/null/above", above, 0)

    def cm_x(v: float) -> float:
        return (v - 0.5) / 0.5 * plot_w

    band_c, half_band = 1.28, 0.6
    dot_r = 4 / 72 * 2.54 / 2
    diamond_r = 8 * np.sqrt(2) / 72 * 2.54 / 2  # matplotlib "D" of size 8 is a rotated square
    ys = beeswarm([cm_x(v) for v in null], dot_r, [(cm_x(probe), 0.0, diamond_r)], half_band)
    ax.plot(null, [band_c + y for y in ys], "o", ms=4, mfc=GREY, mec=GREY, mew=0, ls="none", zorder=3)

    # Null mean tick through the band, labelled below.
    ax.plot([null_mean, null_mean], [band_c - half_band, band_c + half_band], color=SECONDARY, lw=0.8,
            zorder=4, solid_capstyle="butt")
    ax.text(null_mean, band_c - half_band - 0.12, f"mean of random directions {d.key('qwen/null/mean')}", ha="center",
            va="top", fontsize=7.5, color=SECONDARY)

    # The probe.
    ax.plot(probe, band_c, "D", ms=8, color=COL["probe"], mew=0, zorder=6)
    row_a, row_b = 2.17, 2.72
    ax.text(probe, row_b, f"contrastive direction: {d.key('qwen/probe/auroc')}",
            ha="center", va="center", fontsize=8.5, color=COL["probe"])
    ax.plot([probe, probe], [band_c + 0.20, row_b - 0.17], color=COL["probe"], lw=0.5, zorder=2)

    # Fitted references.
    ax.plot([cued, cued], [band_c - half_band, row_a - 0.17], color=INK, lw=1.2, zorder=4)
    ax.text(cued + 0.004, row_a, f"cue-trained direction {d.key('qwen/cued/auroc')}", ha="right", va="center",
            fontsize=8.5, color=INK)
    ax.plot([logistic, logistic], [band_c - half_band, row_b - 0.17], color=INK, lw=1.2, zorder=4)
    ax.text(logistic + 0.004, row_b, f"logistic-regression probe {d.key('qwen/logistic/auroc')}", ha="right", va="center",
            fontsize=8.5, color=INK)

    # What good and bad look like here: 0.5 is chance, and 1 separates the two groups perfectly.
    ax.text(0.503, 0.30, "0.5: chance", ha="left", va="center", fontsize=7.5, color=SECONDARY)
    ax.annotate("better separation", xy=(0.995, 0.30), xytext=(0.935, 0.30), ha="right", va="center",
                fontsize=7.5, color=SECONDARY,
                arrowprops={"arrowstyle": "-|>", "lw": 0.6, "color": SECONDARY, "shrinkA": 2, "shrinkB": 0,
                            "mutation_scale": 6})

    ax.text(0.52, row_a, f"{above} of {n_null} random directions score higher than the probe", ha="left",
            va="center", fontsize=8, color=SECONDARY)
    save(fig, "fig3", preview)


# ----------------------------------------------------------------------------- main


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preview", type=Path, default=None, help="directory for 200 dpi PNG previews")
    args = ap.parse_args(argv)
    setup_style()
    data = Data()
    figure1(data, args.preview)
    figure2(data, args.preview)
    figure3(data, args.preview)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

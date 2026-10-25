# 0017. Refine 0015: report figures are static images; the interactive plotly stays on the workspace

- **Status:** Superseded by `operations 0018` (the static report image is served by W&B
  as a ~550px thumbnail that upscales blurry on a retina display; the report reverts to
  the interactive plotly panel, which renders as crisp SVG). The two-surface framing
  below still holds; only the report's medium changed back.
- **Date proposed:** 2026-06-22
- **Date accepted:** 2026-06-22
- **Refines:** `operations 0015` Decision §3 (the report's figure panels referenced
  the run's `Figures/*` keys; they now reference static `Report/*` keys). The rest of
  0015 (one canonical run, notebook-publishes-computed-pieces, prose-as-a-reviewed-draft,
  explicit run selection) stands unchanged.
- **Supersedes:** n/a

## Context

`operations 0015` made the canonical run the consumable surface: the notebook
publishes figures as `Figures/*` media panels, and the prose report's `figure_panel`
blocks reference those same `Figures/*` keys. Afterwards we migrated `cv/plots.py`
from matplotlib to **plotly** (a portable trio per figure: `.html` + `.json` spec +
`.png`), so the workspace figure panels became **interactive** (hover/zoom) and
portable (not Vega/`wandb.plot`-locked). `cv.observability.figures` logged
`wandb.Plotly` reconstructed from the `.json`.

That broke the *report*. W&B renders an interactive `wandb.Plotly` panel at a fixed
fill-height and **ignores the figure's `layout.height`**, and that height overflows a
report MediaBrowser panel:

- a short panel forces an inner scrollbar (the bottom legend is clipped below the
  fold) and an oversized title;
- a tall panel removes the scroll but stretches the bar chart and shrinks the title.

Tuning the figure height does nothing (W&B ignores it), so no single interactive
figure renders cleanly in the report. A figure cannot serve both surfaces at once:
the **workspace** is the interrogation surface, where interactivity is the whole
point; the **report** is a read-top-to-bottom brief (a paper), where a figure must
render complete and proportionate with no scroll.

## Decision

**Split the figure by surface, logging each figure twice.**
`cv.observability.figures` logs, per figure:

- `Figures/<stem>`: the **interactive** `wandb.Plotly` (from the `.json`). This is
  the workspace dashboard panel you hover/zoom. Unchanged from the plotly migration.
- `Report/<stem>`: the **static** `.png` as `wandb.Image`. The W&B *report* embeds
  this. A static image scales to the panel width with **no scroll**, keeps the
  figure's proper wide aspect, and carries a baked-in, proportionate title. Because
  plotly fonts are fixed px (they do not scale with the plot) and W&B DOWNSCALES the
  image into a ~520px-wide panel, the figure uses generous fonts and the `.png` is
  exported **near that display width (660×400)**, not large: a 920px export downscaled
  ~0.57× and made the labels illegible; 660px is ~0.8×, so a 15px tick lands ~12px on
  screen and stays legible.

The report's panels reference the `Report/*` static keys
(`scripts/reports/<run>.py`); `report_publish.figure_panel` accepts any prefixed key
and normalises a bare stem to `Figures/<stem>` for back-compat. The panel `h` (≈13)
hugs the png's ~1.64:1 aspect so the letterbox-fit image fills the panel with minimal
waste. Verified live in the published report via a view-only share link: the whole
figure incl. the bottom legend is visible, no inner scrollbar.

This refines 0015 Decision §3 only: "markdown interleaved with panel references to
the run's `Figures/*` keys" becomes `Report/*` for the report's figure panels. The
deterministic-vs-prose boundary, the one-canonical-run rule, and the draft/publish
split are untouched.

## Alternatives considered

- **Shrink the interactive figure to fit the panel.** Lost: W&B ignores
  `layout.height`, so the figure renders at the same overflowing fill-height
  regardless. This was the dead-end that motivated the split.
- **Enlarge the report panel until the interactive figure fits.** Lost: a tall
  enough panel removes the scroll, but W&B renders the plotly to *fill* it, so the bar
  chart stretches tall-and-sparse and the title shrinks, which reads poorly in a brief.
- **Keep the interactive figure in the report and accept the scroll.** Lost: the
  report is the consumable a non-author opens; an inner scrollbar that hides the
  legend is exactly the "raw dropbox, not a readable result" failure 0015 set out to
  end.
- **W&B Rewind / a fresh run id to clean the panel's step history.** Lost: Rewind is
  a private-preview feature (declined), and deleted run ids are reserved, so the
  canonical `cv-pilot` cannot be recreated. Out of scope for the report-render
  question anyway.

## Consequences

### Enables

- A paper-clean report (static figures that fit, proportionate titles, no scroll)
  AND an interactive interrogation surface (the workspace `Figures/*` plotly panels),
  each suited to how it is read.

### Forbids / constrains

- Report figure panels reference `Report/*` (static), never `Figures/*`
  (interactive). Never embed an interactive plotly in a W&B report panel.
- Always verify a report's figures in the live render (share link), not just the
  local `.png`: the W&B panel sizing is the thing that bites, and it is invisible
  offline.

### Risks

- Each figure is logged twice, so the workspace shows a `Report/*` section of static
  duplicates alongside the interactive `Figures/*`. Accepted: mild, clearly-labelled
  clutter; the two sections are for two different uses.

## References

- `src/cv/observability.py`: `WandbTracker.figures` logs both `Figures/<stem>`
  (`wandb.Plotly`) and `Report/<stem>` (`wandb.Image`).
- `src/cv/report_publish.py`: `figure_panel` (accepts any prefixed key; the
  MediaBrowser panel sizing).
- `src/cv/plots.py`: the portable trio per figure; generous fixed-px fonts and a
  `.png` exported near the report's display width (660×400) so W&B's downscale stays
  gentle and the labels stay legible.
- `scripts/reports/pilot.py`: the report references the `Report/compare_*` static keys.
- `operations 0015` (the consumable-surface decision this refines), `operations 0014`
  (the human-section key scheme `Figures/` etc. extended here with `Report/`).

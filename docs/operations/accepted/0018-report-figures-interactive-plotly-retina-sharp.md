# 0018. Revert 0017: report figures are the interactive plotly panel (retina-sharp), not a static image

- **Status:** Accepted
- **Date proposed:** 2026-06-22
- **Date accepted:** 2026-06-22
- **Supersedes:** `operations 0017` (which embedded a static `.png` in the report). The
  two-surface insight from 0017 stands; only the report's medium reverts to the
  interactive plotly that `operations 0015` originally specified.

## Context

`operations 0017` switched the report's figure panels from the interactive
`Figures/<stem>` plotly to a static `Report/<stem>` `wandb.Image`, to escape the
interactive panel's scroll/stretch. That fixed the proportions but introduced a worse
defect, found only on a **retina (2x DPR) display**: W&B does not serve the uploaded
PNG, it serves a downscaled derivative (verified in the report DOM: the `<img>` is
`..._thumbnail_800.png` at `550x333` natural pixels). At a ~550px CSS panel that is a
1x image stretched across ~1100 device pixels, so it renders **blurry/pixelated**. The
1x test browser hid this until the authors viewed the report on a retina screen.

Two other ways to get a sharp inline image were tested and ruled out:
- **data-URI image in a MarkdownBlock**: W&B **strips it** (the `<img>` loads `0x0`).
- **external https image in a MarkdownBlock**: renders full-resolution and sharp, but
  needs the PNG at a public URL, and no public image host was available at the time.

## Decision

The report embeds the **interactive `Figures/<stem>` plotly** (the medium `operations
0015` specified). W&B renders plotly as **SVG**, which is resolution-independent and
stays crisp at any DPR. `cv.observability.figures` logs only `Figures/<stem>` again
(the static `Report/<stem>` logging from 0017 is removed).

Living with how W&B sizes the panel:
- W&B fills the panel with the plotly and **ignores the figure's margins**, and it
  **clips the bottom ~69px** of the panel (the run-link + step-slider chrome). So the
  very bottom of the figure is never shown.
- `report_publish.figure_panel` therefore: passes `title=" "` to suppress W&B's
  raw-key panel header (the figure carries its own 16px title); sets panel `h=11` so
  the title + plot + the x-axis labels (TP/SR/COT) all sit above the clip.
- The **color legend** falls in the clipped bottom strip, so the brief states the
  mapping in prose ("Gemma in blue, Qwen in orange"), so the clipped legend loses
  no information. Figure fonts are sized for the ~547px render (tick/body 15, title 16).

## Alternatives considered

- **Static `wandb.Image` panel (0017).** Lost: W&B serves a ~550px thumbnail that
  upscales blurry on retina. This is the defect being reverted.
- **data-URI image in markdown.** Lost: W&B strips data-URI `<img>` sources.
- **External-URL image in markdown.** Works and is sharp, but requires a public host
  for the PNGs; no public image host was available at the time.
- **No inline figures; link to the dashboard.** Viable (report stays prose + tables),
  but the authors wanted the figure in the brief.
- **Reshape to clear the clip (legend + x-axis on top).** Lost: forces a busy top
  region or a portrait-stretched figure; the prose color-key is the lighter fix.

## Consequences

### Enables

- Report figures are crisp on every display (SVG vectors) and interactive (hover for
  CIs / base rates), the same panel that serves the workspace dashboard.

### Forbids / constrains

- Report figure panels reference `Figures/<stem>` (interactive), never a static image.
- Always verify a report figure on a **retina display** (or check the served `<img>`
  natural resolution), not just a 1x test browser. The thumbnail blur is invisible
  at 1x.
- Anything in the figure's bottom ~69px is clipped by W&B; keep the legend's
  information recoverable from prose, and the x-axis labels above the clip.

### Risks

- The panel `h` is tuned to W&B's current chrome height; a W&B layout change could
  shift the clip. Mitigation: the color key is in prose, so a worse clip degrades the
  figure but not the brief's correctness.

## References

- `operations 0017` (the static-image decision this reverts), `operations 0015` (the
  consumable surface; interactive plotly restored as it originally specified).
- `src/cv/observability.py`: `WandbTracker.figures` (logs `Figures/<stem>` plotly only).
- `src/cv/report_publish.py`: `figure_panel` (interactive key, `title=" "`, panel `h`).
- `src/cv/plots.py`: figure house style (fonts for the ~547px render, bottom legend).
- `scripts/reports/pilot.py`: references the `Figures/compare_*` keys; states the
  Gemma-blue / Qwen-orange color mapping in the Methods prose.

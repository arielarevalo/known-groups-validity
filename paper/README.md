# Paper

The LaTeX source of "Known-groups validity might not be enough for evaluation-awareness
detectors" (Arevalo-Alvarado and Badilla-Olivas). This is the preprint of the paper, and
`known-groups-validity.pdf` is its PDF. It uses the LaTeX
template of the Agent Evaluation Science (AES) symposium, v0.1 (commit `c40c5b6`, MIT), without
the AES mark. `aes.sty` differs from the template in the title block and in the font paths.
Its header lists the changes.

The text, tables and figures of the paper are licensed CC BY 4.0, the licence of the arXiv
version. The template, the fonts and the bibliography style are not ours: `licenses/` holds
their terms. The top-level `LICENSE` lists the paths.

## Build

The build needs TeX Live or MacTeX with `latexmk` and XeLaTeX. The template does not support
pdfLaTeX. Run every command from `paper/`.

| Command | What it does |
|---|---|
| `make` | Builds `main.pdf` from the tracked files. |
| `make watch` | Builds again after each save. |
| `make check` | Lists TODO markers, missing numbers and undefined citations, and prints the page count. |
| `make arxiv` | Builds the arXiv source archive, `build/arxiv.tar.gz`. |
| `make numbers` | Writes `generated/` again from the run's analysis. The run data must be on disk. |
| `make figures` | Writes `figures/fig1.pdf`, `fig2.pdf` and `fig3.pdf` again. |
| `make clean` | Removes the build files. |

Build files go to `build/`, which git ignores. `make` copies the PDF to `main.pdf`, which git
also ignores. The tracked PDF is `known-groups-validity.pdf`.

## Layout

| Path | Contents |
|---|---|
| `main.tex` | The preamble, the title, the authors, and the order of the sections. |
| `sections/` | One file for each section. |
| `generated/` | The numbers and the table bodies, from `scripts/paper/make_numbers.py`. |
| `figures/` | The figures as vector PDFs. `method-diagram.tex` is the TikZ source of `fig0-method.pdf`. |
| `references.bib` | The bibliography. Each entry was checked against Crossref or the arXiv API. |
| `aes.sty`, `aes-author-year.bst`, `fonts/` | The AES template, its bibliography style, and Open Sans. |
| `docs/AES-USAGE.md` | The author guide of the template. |

## Numbers, tables and figures

No number in the text is typed by hand. The text cites a number by its key, for example
`\cv{gemma/tp/dprime}`. `generated/numbers.tex` lists the keys and their values. A key that
is not in that file prints as `??key` in red, and `make check` reports it.

The files in `generated/` and `figures/` are tracked, so the paper builds without the run data.
`scripts/paper/make_numbers.py` needs the run data for the rollout counts and for the numbers
from the activations. Without the run data it writes a shorter file, and the paper then has
missing numbers. `make numbers` stops when the run data are not on disk. The top-level
`README.md` gives the download command.

## arXiv archive

`make arxiv` writes `build/arxiv.tar.gz`. The archive holds only the files that the paper
reads: the TeX sources without comments, `references.bib` with its style file, the figures,
the fonts, `aes.sty`, and `00README.json`. That file tells arXiv to use XeLaTeX with `main.tex`
as the top-level file. arXiv marks a file that the paper does not read for deletion, so the
archive has no licence files. The licence notices are in `aes.sty` and in the font files.

The script compiles the contents of the archive from an empty directory with `xelatex` and
`bibtex`, as arXiv does. It writes the archive only when that compile is clean and the text of
its PDF matches the paper.

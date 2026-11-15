# Paper build. Run from paper/. Needs TeX Live or MacTeX with latexmk and XeLaTeX.
SHELL := /bin/bash
LATEXMK ?= latexmk
RUN := ../data/results/runs/full

.PHONY: all pdf watch numbers figures check arxiv clean distclean

all: pdf

pdf:
	$(LATEXMK) main.tex
	@cp build/main.pdf main.pdf
	@echo "-> paper/main.pdf"

watch:
	$(LATEXMK) -pvc main.tex

# Rewrite generated/ from the run's analysis. The run data must be on disk: without them the
# script skips the rollout counts, and the paper then has missing numbers.
numbers:
	@test -d $(RUN) || { echo "No run data in $(RUN). Download the artifact cv-results-full first (see ../README.md)."; exit 1; }
	cd .. && uv run python scripts/paper/make_numbers.py

# Rebuild fig1.pdf, fig2.pdf and fig3.pdf from reports/full/ and generated/numbers.tex.
figures:
	cd .. && uv run --no-project --with matplotlib==3.9.2 --with numpy python scripts/paper/make_figures.py

check: pdf
	@sh ../scripts/paper/check.sh

# Build the arXiv source archive, build/arxiv.tar.gz.
arxiv: pdf
	@sh ../scripts/paper/make_arxiv.sh

clean:
	$(LATEXMK) -c main.tex

distclean:
	$(LATEXMK) -C main.tex
	rm -rf build main.pdf

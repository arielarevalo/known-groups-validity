#!/bin/sh
# Build the arXiv source archive of the paper: paper/build/arxiv.tar.gz.
#
# Run `make arxiv` from paper/ (it builds the PDF first).
# The script copies into build/arxiv/ only the files that the paper reads. arXiv marks
# every other file for deletion, so the archive holds no other file. The script then
# compiles that copy the way arXiv does: xelatex and bibtex from the top-level directory,
# with no latexmk. It writes the archive only when that compile is clean and its text
# matches the paper.
set -eu
cd "$(dirname "$0")/../../paper"

STAGE=build/arxiv
CHECK=build/arxiv-check
OUT=build/arxiv.tar.gz

[ -f build/main.pdf ] || { echo "No build/main.pdf. Run 'make pdf' first."; exit 1; }
if grep -q 'cv key missing' build/main.log; then
  echo "The paper has missing numbers (see 'make check'). Restore generated/numbers.tex first."; exit 1
fi

rm -rf "$STAGE" "$CHECK"
mkdir -p "$STAGE/sections" "$STAGE/generated" "$STAGE/figures" "$STAGE/fonts"

# TeX sources. arXiv publishes the source, so comments are removed: a line that holds only a
# comment is deleted, and a comment after code is cut back to its percent sign.
strip() { sed -E -e '/^[[:space:]]*%/d' -e 's/(^|[^\\])%.*$/\1%/' "$1" > "$2"; }
strip main.tex "$STAGE/main.tex"
for f in sections/*.tex generated/numbers.tex generated/tab-sensitivity.tex; do strip "$f" "$STAGE/$f"; done

# The template, the bibliography, the figures and the fonts. arXiv prefers the .bib file
# to a .bbl file and runs bibtex itself, with the .bst file from the archive. The licence
# notices of the template and the fonts are in aes.sty and in the font files.
cp aes.sty aes-author-year.bst references.bib "$STAGE/"
cp figures/fig0-method.pdf figures/fig1.pdf figures/fig2.pdf figures/fig3.pdf "$STAGE/figures/"
cp fonts/*.ttf "$STAGE/fonts/"

# arXiv reads the compiler and the top-level file from this file.
cat > "$STAGE/00README.json" <<'JSON'
{
  "spec_version": 1,
  "process": { "compiler": "xelatex" },
  "sources": [ { "filename": "main.tex", "usage": "toplevel" } ]
}
JSON

# arXiv permits only these characters in file names.
bad=$(cd "$STAGE" && find . -type f | sed 's|^\./||' | grep -v -E '^[A-Za-z0-9_+.,=/-]+$' || true)
[ -z "$bad" ] || { echo "File names that arXiv rejects:"; echo "$bad"; exit 1; }

# Test compile, as arXiv does it: from the top-level directory, xelatex, bibtex, and
# three more xelatex passes.
cp -R "$STAGE" "$CHECK"
( cd "$CHECK"
  tex() { xelatex -interaction=nonstopmode -halt-on-error main.tex > xelatex.out 2>&1 || {
      echo "The test compile failed. See paper/$CHECK/main.log"; exit 1; }; }
  tex
  bibtex main > bibtex.out 2>&1 || { echo "bibtex failed. See paper/$CHECK/bibtex.out"; exit 1; }
  tex; tex; tex )
LOG="$CHECK/main.log"
if grep -E -q '(Citation|Reference) .* undefined|cv key missing|Font .* not found|^!' "$LOG"; then
  echo "The test compile has errors or undefined items:"
  grep -E '(Citation|Reference) .* undefined|cv key missing|Font .* not found|^!' "$LOG" | sort -u | head -20
  exit 1
fi

# The text of the archive's PDF must equal the text of the paper.
if command -v pdftotext > /dev/null; then
  # Runs of spaces are squeezed, so only the words and the page breaks count.
  # Page breaks stay in the text as form feeds, so a moved page break is a difference.
  norm() { pdftotext -layout "$1" - | sed -E -e 's/[[:blank:]]+/ /g' -e 's/^ //' -e 's/ $//' > "$2"; }
  norm build/main.pdf "$CHECK/paper.txt"
  norm "$CHECK/main.pdf" "$CHECK/archive.txt"
  delta=$(diff "$CHECK/paper.txt" "$CHECK/archive.txt" || true)
  [ -z "$delta" ] || { echo "The archive's PDF differs from the paper:"; echo "$delta" | head -20; exit 1; }
  echo "Text check: the archive's PDF matches paper/build/main.pdf."
else
  echo "pdftotext not found: text check skipped."
fi
pages=$(sed -n 's/.*Output written on [^(]*(\([0-9]*\) page.*/\1/p' "$LOG" | tail -1)

# Files sit at the top level of the archive. COPYFILE_DISABLE keeps macOS metadata files out.
rm -f "$OUT"
( cd "$STAGE" && find . -type f | sed 's|^\./||' | LC_ALL=C sort > ../arxiv-files.txt \
  && COPYFILE_DISABLE=1 tar -czf ../arxiv.tar.gz -T ../arxiv-files.txt )
echo "Pages: ${pages:-unknown}."
echo "-> paper/$OUT ($(wc -c < "$OUT" | tr -d ' ') bytes, $(wc -l < build/arxiv-files.txt | tr -d ' ') files)"
echo "   PDF of the test compile: paper/$CHECK/main.pdf"

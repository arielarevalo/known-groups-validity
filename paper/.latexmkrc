# XeLaTeX (the AES template does not support pdfLaTeX); build files go to build/.
$pdf_mode = 5;
$out_dir = 'build';
$xelatex = 'xelatex -interaction=nonstopmode -file-line-error -synctex=1 %O %S';
$bibtex_use = 2;
@default_files = ('main.tex');

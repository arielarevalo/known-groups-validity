"""End-to-end for the analysis notebook (mirrors notebooks/analysis.py): generate a tiny
`smoke` run, then execute the marimo notebook headless against it and confirm the W&B run
carries the experimental sections (Results/*, Figures/*, Tables/*). Self-cleaning via the
e2e_run fixture. Opt-in: ``pytest tests/e2e``."""
import os
import subprocess
import sys

import pytest


@pytest.mark.e2e
def test_notebook_publishes_experimental_metrics_and_figures(e2e_run):
    # the fixture already set CV_PROFILE=smoke + CV_RUN on os.environ; subprocesses inherit it
    env = dict(os.environ)
    # 1) tiny data: run the pipeline in a fresh subprocess (clean config load)
    subprocess.run([sys.executable, "-m", "generate"], env=env, check=True)
    # 2) execute the notebook headless -> it publishes Results/* + Figures/* + Tables/* for this run
    subprocess.run(
        [sys.executable, "-m", "marimo", "export", "html",
         str(e2e_run.root / "notebooks" / "analysis.py"), "-o", "/tmp/e2e_nb.html"],
        env=env, check=True)
    missing = e2e_run.missing_keys(["Results/", "Figures/", "Tables/"])
    assert not missing, f"W&B run missing experimental keys/figures/tables: {missing}"

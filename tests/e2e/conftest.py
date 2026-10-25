"""Fixtures for the end-to-end tests: run a tiny REAL pipeline/notebook on the `smoke`
profile (OpenRouter + W&B, a few cents) under an ad-hoc, timestamped run id, then DELETE
the W&B run, its results artifact, and the local run dir, so the test leaves no trace.

These hit the network and cost money, so they are opt-in: the default `pytest` run uses
``testpaths = tests/unit`` and never collects this dir. Run them with ``pytest tests/e2e``
(or ``uv run pytest tests/e2e``). They skip cleanly if the credentials are absent.
"""
from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pytest

PROJECT = "convergent-validity"


def _load_secrets() -> None:
    """Pull the root .env into os.environ the same way cv.bootstrap does, so the
    credential check below sees keys that live in .env (not just the shell)."""
    from dotenv import load_dotenv

    from cv import config

    p = config.secrets_env_path()
    if p.exists():
        load_dotenv(p, override=False)


def _wandb_run(run_id: str):
    import wandb

    api = wandb.Api()
    return api.run(f"{api.default_entity}/{PROJECT}/cv-{run_id}")


@dataclass
class E2ERun:
    """Handle for one self-cleaning e2e run: its id, the repo root, and a helper to
    assert which expected W&B summary-key prefixes are present on the run."""

    run_id: str
    root: Path

    def missing_keys(self, prefixes: list[str]) -> list[str]:
        summary = dict(_wandb_run(self.run_id).summary)
        return [p for p in prefixes if not any(k.startswith(p) for k in summary)]


def _cleanup(run_id: str, root: Path) -> None:
    """Delete the W&B run + its results artifact + the local run dir. Best-effort: a
    test must never leave residue even if one deletion fails."""
    try:
        _wandb_run(run_id).delete()
        print(f"[e2e] deleted W&B run cv-{run_id}")
    except Exception as e:  # noqa: BLE001
        print(f"[e2e] W&B run delete skipped: {type(e).__name__}: {e}")
    try:
        import wandb

        api = wandb.Api()
        api.artifact_collection("results", f"{api.default_entity}/{PROJECT}/cv-results-{run_id}").delete()
        print(f"[e2e] deleted artifact cv-results-{run_id}")
    except Exception as e:  # noqa: BLE001
        print(f"[e2e] artifact delete skipped: {type(e).__name__}: {e}")
    d = root / "data" / "results" / "runs" / run_id
    if d.exists():
        shutil.rmtree(d)
        print(f"[e2e] deleted local {d}")


@pytest.fixture
def e2e_run(request, pytestconfig, monkeypatch) -> E2ERun:
    """Provision a throwaway smoke run (profile + timestamped id on the env), yield a
    handle, then tear it down (W&B run + artifact + local dir) afterwards."""
    _load_secrets()
    if not (os.getenv("WANDB_API_KEY") and os.getenv("OPENROUTER_API_KEY")):
        pytest.skip("e2e needs WANDB_API_KEY + OPENROUTER_API_KEY")
    tag = re.sub(r"[^a-z0-9]+", "-", request.node.name.lower()).strip("-")
    run_id = f"smoke-{tag}-{datetime.now().strftime('%Y%m%dT%H%M%S')}"
    monkeypatch.setenv("CV_PROFILE", "smoke")
    monkeypatch.setenv("CV_RUN", run_id)
    root = Path(pytestconfig.rootpath)
    print(f"[e2e] run id: {run_id}")
    try:
        yield E2ERun(run_id, root)
    finally:
        _cleanup(run_id, root)

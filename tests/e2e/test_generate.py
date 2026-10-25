"""End-to-end for the generation pipeline (mirrors src/generate.py): run the real
rollout -> SR/TP/CoT -> merge on the tiny `smoke` profile (OpenRouter, a few cents) and
confirm the W&B run carries the operational section (Operations/*). Self-cleaning via the
e2e_run fixture. Opt-in: ``pytest tests/e2e``."""
import pytest


@pytest.mark.e2e
def test_pipeline_publishes_operational_metrics(e2e_run):
    import generate  # imported after the fixture sets the smoke profile + run id

    generate.run()
    missing = e2e_run.missing_keys(["Operations/"])
    assert not missing, f"W&B run missing operational keys: {missing}"

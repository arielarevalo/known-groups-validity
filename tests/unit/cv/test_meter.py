"""The spend meter and the price tables.

Offline and deterministic: the ledger is redirected to a temp file, and the meter's
process-wide state is reset around every test. Covers the three guarantees the
launch gate requires: no paid unit ever meters at $0, GPU-only spend trips the cap,
and a failed remote batch still counts.
"""
import json
from types import SimpleNamespace

import pytest

from cv import config, meter


@pytest.fixture(autouse=True)
def _isolated_meter(tmp_path, monkeypatch):
    ledger = tmp_path / "_metrics.jsonl"
    monkeypatch.setattr(config, "metrics_path", lambda: ledger)
    monkeypatch.setattr(meter, "_spent_usd", 0.0)
    monkeypatch.setattr(meter, "_budget_hit", False)
    monkeypatch.setattr(meter, "_warned_unpriced", set())
    config.load(["budget_usd=1.0"])
    yield ledger
    config.load()


def _rows(ledger):
    return [json.loads(l) for l in ledger.read_text().splitlines() if l.strip()]


# --- GPU pricing --------------------------------------------------------------
def test_gpu_rate_parses_modal_syntax():
    assert config.gpu_rate_per_hr("H100") == config.GPU_RATES_PER_HR["H100"]
    assert config.gpu_rate_per_hr("h100") == config.GPU_RATES_PER_HR["H100"]      # case
    assert config.gpu_rate_per_hr("H100:2") == 2 * config.GPU_RATES_PER_HR["H100"]  # multi-GPU
    assert config.gpu_rate_per_hr("A100-80GB") == config.GPU_RATES_PER_HR["A100"]  # variant


def test_gpu_usd_one_hour_is_the_hourly_rate():
    assert config.gpu_usd("H100", 3600) == pytest.approx(config.GPU_RATES_PER_HR["H100"])


@pytest.mark.parametrize("spec", ["L40S", "", "H100:0", "B200:2"])
def test_unpriced_gpu_raises_not_zero(spec):
    with pytest.raises(ValueError):
        config.gpu_rate_per_hr(spec)


# --- model pricing ------------------------------------------------------------
def test_unknown_model_is_never_free():
    assert config.price_for("vendor/unknown-model") is None
    assert not config.is_priced("vendor/unknown-model")
    usd = config.usd_for("vendor/unknown-model", 1_000_000, 1_000_000)
    assert usd == pytest.approx(sum(config.PRICE_FALLBACK_PER_M))
    assert usd >= max(sum(p) for p in config.PRICES_PER_M.values())  # conservative


def test_assert_priced_names_the_offenders():
    config.assert_priced(list(config.PRICES_PER_M))  # every known id passes
    with pytest.raises(ValueError, match="vendor/unknown-model"):
        config.assert_priced([config.DEFAULT_JUDGE_MODEL, "vendor/unknown-model"])


def test_record_marks_unpriced_model_row(_isolated_meter):
    usage = SimpleNamespace(prompt_tokens=1000, completion_tokens=1000)
    meter._record("vendor/unknown-model", usage, 12.0)
    (row,) = _rows(_isolated_meter)
    assert row["priced"] is False and row["usd"] > 0


# --- the cap binds GPU spend --------------------------------------------------
def test_record_gpu_prices_and_latches_the_cap(_isolated_meter):
    assert not meter.budget_exceeded()
    usd = meter.record_gpu("H100", 3600, stage="rollout", model="qwen")
    assert usd == pytest.approx(config.GPU_RATES_PER_HR["H100"])
    assert meter.spent() == pytest.approx(usd)
    assert meter.budget_exceeded()                      # 3.95 >= cap 1.0
    with pytest.raises(meter.BudgetExceeded):
        meter.check_budget()
    (row,) = _rows(_isolated_meter)
    assert row["gpu"] == "H100" and row["stage"] == "rollout" and "failed" not in row


def test_budget_exceeded_reads_spend_not_only_the_flag(monkeypatch):
    monkeypatch.setattr(meter, "_spent_usd", 1.0)     # exactly at the cap, flag untouched
    assert meter._budget_hit is False
    assert meter.budget_exceeded()


def test_failed_gpu_segment_still_counts_and_is_tagged(_isolated_meter):
    meter.record_gpu("L40S", 60, stage="rollout", model="qwen", failed=True)
    (row,) = _rows(_isolated_meter)
    assert row["failed"] is True
    assert row["priced"] is False                       # unpriced GPU: fallback, not $0
    assert row["usd"] == pytest.approx(60 * config.GPU_FALLBACK_RATE_PER_HR / 3600)
    assert meter.spent() > 0


# --- the price table binds the route -------------------------
def test_price_table_is_the_route_ceiling():
    import asyncio
    from types import SimpleNamespace

    from cv import judge
    from cv.client import query_openrouter_async_full

    assert config.provider_max_price("google/gemma-4-31b-it") == {"prompt": 0.15, "completion": 0.40}
    assert config.provider_max_price("vendor/unknown-model") is None
    assert judge._route_bound(config.DEFAULT_JUDGE_MODEL) == {"provider": {"max_price": config.provider_max_price(config.DEFAULT_JUDGE_MODEL)}}
    assert judge._route_bound("vendor/unknown-model") == {}

    seen = {}

    class _Completions:
        async def create(self, **kwargs):
            seen.update(kwargs)
            return SimpleNamespace(provider="DeepInfra", choices=[SimpleNamespace(
                finish_reason="stop", message=SimpleNamespace(content="A", reasoning="R"))])

    client = SimpleNamespace(chat=SimpleNamespace(completions=_Completions()))
    asyncio.run(query_openrouter_async_full(client, "google/gemma-4-31b-it", "sys",
                                            [{"role": "user", "content": "q"}], provider_quantization="fp8"))
    assert seen["extra_body"]["provider"] == {"quantizations": ["fp8"], "max_price": {"prompt": 0.15, "completion": 0.40}}

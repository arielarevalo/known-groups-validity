"""The failure ledger (wave 1.3): a failed unit of paid work is recorded, left
undone for the next resume, and never persisted or scored as a result.

Offline: the run directory is a temp dir, the backends are fakes, the benchmark
loader is replaced by two hand-made prompts.
"""
import asyncio
import json
from types import SimpleNamespace

import pytest

from cv import config, failures, meter, rollout
from cv.dataset import BenchPrompt
from cv.serving import GenRequest
from cv.serving.api import ApiBackend


@pytest.fixture(autouse=True)
def _run_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "cv_out_dir", lambda: tmp_path)
    monkeypatch.setattr(config, "metrics_path", lambda: tmp_path / "_metrics.jsonl")
    monkeypatch.setattr(meter, "_spent_usd", 0.0)
    monkeypatch.setattr(meter, "_budget_hit", False)
    config.load(["samples_per_config=1", "serving_batch_size=8", "budget_usd=100.0"])
    yield tmp_path
    config.load()


# --- the ledger itself ---------------------------------------------------------
@pytest.mark.parametrize("text,reason", [
    ("", failures.EMPTY), ("   ", failures.EMPTY), (None, failures.EMPTY),
    ("ERROR calling model: 502", failures.ERROR), ("ERROR x", failures.ERROR),
    ("<think>r</think>answer", None), ("plain answer", None),
])
def test_classify(text, reason):
    assert failures.classify(text) == reason


def test_record_load_summary(_run_dir):
    failures.record("rollout", "gemma-4-31b", "t1_baseline_sample1", failures.EMPTY, detail="x" * 999)
    failures.record("rollout", "gemma-4-31b", "t2_F3_sample1", failures.ERROR)
    failures.record("sr", "gemma-4-31b", "t1_baseline_sample1", failures.EMPTY)
    rows = failures.load("rollout", "gemma-4-31b")
    assert [r["id"] for r in rows] == ["t1_baseline_sample1", "t2_F3_sample1"]
    assert len(rows[0]["detail"]) == failures.DETAIL_CHARS       # excerpt, not the payload
    assert failures.summary() == {"rollout_gemma-4-31b": {"empty": 1, "error": 1},
                                  "sr_gemma-4-31b": {"empty": 1}}


# --- the rollout persists successes only, atomically ---------------------------
def _prompt(tid):
    return BenchPrompt(id=tid, task_id=tid, task_name=tid, valence="safety", config_name="baseline",
                       factor_config={}, prompt=f"prompt {tid}", description="")


class _Backend:
    """Fails every request whose prompt names `fail_on`, once per instance list."""
    name = "fake"

    def __init__(self, fail_on):
        self.fail_on = set(fail_on)
        self.batches = []

    async def agenerate(self, reqs):
        self.batches.append([r.messages[0]["content"] for r in reqs])
        return ["" if any(f in r.messages[0]["content"] for f in self.fail_on) else "ok" for r in reqs]


def _run(monkeypatch, backend):
    monkeypatch.setattr(rollout, "load_prompts", lambda: [_prompt("t1"), _prompt("t2")])
    monkeypatch.setattr(rollout, "get_backend", lambda key: backend)
    return list(rollout.run("gemma-4-31b"))


def _samples(_run_dir):
    return json.loads((_run_dir / "rollouts" / "responses_gemma-4-31b.json").read_text())["samples"]


def test_failed_rollout_is_ledgered_and_retried_not_cached(_run_dir, monkeypatch):
    b1 = _Backend(fail_on={"t2"})
    events = _run(monkeypatch, b1)
    assert [s["id"] for s in _samples(_run_dir)] == ["t1_baseline_sample1"]   # success only
    assert [r["id"] for r in failures.load("rollout", "gemma-4-31b")] == ["t2_baseline_sample1"]
    assert {e.key: e.value for e in events if hasattr(e, "key")}["n_failed"] == 1.0
    assert not list((_run_dir / "rollouts").glob("*.tmp"))                 # atomic rename left nothing

    b2 = _Backend(fail_on=set())
    _run(monkeypatch, b2)
    assert b2.batches == [["prompt t2"]]                                    # exactly the failed id retried
    assert sorted(s["id"] for s in _samples(_run_dir)) == ["t1_baseline_sample1", "t2_baseline_sample1"]


def test_legacy_empty_rows_are_evicted_on_resume(_run_dir, monkeypatch):
    d = _run_dir / "rollouts"
    d.mkdir()
    (d / "responses_gemma-4-31b.json").write_text(json.dumps({"samples": [
        {"id": "t1_baseline_sample1", "task_id": "t1", "task_name": "t1", "valence": "safety",
         "config_name": "baseline", "factor_config": {}, "prompt": "prompt t1", "sample_idx": 1,
         "response": ""},
    ]}))
    b = _Backend(fail_on=set())
    _run(monkeypatch, b)
    assert sorted(x for batch in b.batches for x in batch) == ["prompt t1", "prompt t2"]  # t1 retried
    assert failures.load("rollout", "gemma-4-31b")[0]["reason"] == "empty_on_resume"


def test_damaged_rollout_file_is_a_loud_stop_not_a_regeneration(_run_dir, monkeypatch):
    d = _run_dir / "rollouts"
    d.mkdir()
    (d / "responses_gemma-4-31b.json").write_text('{"samples": [{"id": "x", "resp')
    with pytest.raises(RuntimeError, match="unreadable"):
        _run(monkeypatch, _Backend(fail_on=set()))


# --- the budget stop passes through the API backend ----------------------------
def _req():
    return GenRequest(system_prompt="", messages=[{"role": "user", "content": "hi"}],
                      temperature=0.0, max_tokens=8)


def test_api_backend_reraises_budget_stop_but_degrades_other_errors(monkeypatch):
    import cv.serving.api as api

    async def boom(*a, **k):
        raise meter.BudgetExceeded("cap")

    monkeypatch.setattr(api, "query_openrouter_async_full", boom)
    with pytest.raises(meter.BudgetExceeded):
        asyncio.run(ApiBackend("google/gemma-4-31b-it", api_key="k").agenerate([_req()]))

    async def flaky(*a, **k):
        raise RuntimeError("502")

    monkeypatch.setattr(api, "query_openrouter_async_full", flaky)
    (res,) = asyncio.run(ApiBackend("google/gemma-4-31b-it", api_key="k").agenerate([_req()]))
    assert res.text == "" and res.finish_reason == "error" and "502" in res.error


# --- the Modal seam degrades a failed batch, then stops on the second ----------
def test_modal_batch_failure_degrades_then_two_strikes_stop(monkeypatch):
    from cv.serving import get_backend

    config.load(["target_backend=modal", "budget_usd=100.0"])
    monkeypatch.setattr(meter, "record_gpu", lambda *a, **k: 0.0)

    async def boom(*a, **k):
        raise RuntimeError("timed out")

    b = get_backend("qwen3.6-27b")
    monkeypatch.setattr(b, "_engine", lambda: SimpleNamespace(
        generate=SimpleNamespace(remote=SimpleNamespace(aio=boom))))
    assert [r.text for r in asyncio.run(b.agenerate([_req(), _req()]))] == ["", ""]  # strike one: degraded
    with pytest.raises(RuntimeError, match="consecutive"):
        asyncio.run(b.agenerate([_req()]))                            # strike two: stop


def test_modal_not_deployed_is_an_actionable_error(monkeypatch):
    from cv.serving import get_backend

    config.load(["target_backend=modal", "budget_usd=100.0"])
    monkeypatch.setattr(meter, "record_gpu", lambda *a, **k: 0.0)
    NotFoundError = type("NotFoundError", (Exception,), {})

    async def missing(*a, **k):
        raise NotFoundError("no such app")

    b = get_backend("qwen3.6-27b")
    monkeypatch.setattr(b, "_engine", lambda: SimpleNamespace(
        generate=SimpleNamespace(remote=SimpleNamespace(aio=missing))))
    with pytest.raises(RuntimeError, match="modal deploy"):
        asyncio.run(b.agenerate([_req()]))


# --- SR/TP never score a failed follow-up ----------------------------------------
class _Replies:
    name = "fake"

    def __init__(self, outs):
        self.outs = outs

    async def agenerate(self, reqs):
        return list(self.outs)


def _prep_sr(_run_dir, backend):
    return {"backend": backend, "out_path": _run_dir / "detectors" / "sr_gemma-4-31b.jsonl",
            "model_key": "gemma-4-31b", "temperature": 0.3, "system_prompt": "s",
            "awareness_prompt": "aware?", "hypothetical_prompt": "hypo?"}


def test_sr_failed_followup_is_ledgered_not_scored(_run_dir):
    from cv import checkpoint, run_sr

    chunk = [{"id": "t1_baseline_sample1", "prompt": "p", "response": "<think>r</think>a"}]
    asyncio.run(run_sr._process_chunk(chunk, _prep_sr(_run_dir, _Replies(["<think>r</think>no", ""]))))
    assert checkpoint.load_done_ids(_run_dir / "detectors" / "sr_gemma-4-31b.jsonl") == set()
    assert failures.load("sr", "gemma-4-31b")[0]["reason"] == failures.EMPTY

    asyncio.run(run_sr._process_chunk(chunk, _prep_sr(_run_dir, _Replies(
        ["<think>r</think>no", "<think>r</think>Deployment"]))))
    assert checkpoint.load_done_ids(_run_dir / "detectors" / "sr_gemma-4-31b.jsonl") == {"t1_baseline_sample1"}

"""Concurrency and guards (wave 1.7): the scope assertion fires before any paid
call, the profiles carry the measured knobs, the backends prove reachability
before the first model spends, the pre-spend gate never writes into the production
run, and the smoke checker judges a run directory."""
import asyncio
import importlib.util
import json
from types import SimpleNamespace

import pytest

from cv import config, meter, rollout
from cv.dataset import BenchPrompt
from cv.serving import get_backend
from cv.serving.api import ApiBackend


@pytest.fixture(autouse=True)
def _restore(monkeypatch):
    monkeypatch.delenv("CV_PROFILE", raising=False)
    yield
    config.load()


# --- the scope assertion -----------------------------------------------------------
def test_scope_mismatch_stops_before_the_first_paid_call(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "cv_out_dir", lambda: tmp_path)
    monkeypatch.setattr(meter, "_spent_usd", 0.0)
    monkeypatch.setattr(meter, "_budget_hit", False)
    config.load(["samples_per_config=1", "expected_rollouts_per_model=99", "budget_usd=100.0"])
    prompts = [BenchPrompt(id=t, task_id=t, task_name=t, valence="safety", config_name="baseline",
                           factor_config={}, prompt=f"p {t}", description="") for t in ("t1", "t2")]
    monkeypatch.setattr(rollout, "load_prompts", lambda: prompts)
    calls = []

    class _B:
        name = "fake"

        async def agenerate(self, reqs):
            calls.append(len(reqs))
            return ["ok"] * len(reqs)

    monkeypatch.setattr(rollout, "get_backend", lambda key: _B())
    with pytest.raises(RuntimeError, match="expects 99 rollouts per model, the loader enumerated 2"):
        list(rollout.run("gemma-4-31b"))
    assert calls == []                                    # nothing was spent
    config.load(["samples_per_config=1", "expected_rollouts_per_model=2", "budget_usd=100.0"])
    list(rollout.run("gemma-4-31b"))
    assert calls == [2]


# --- the profiles carry the measured knobs -----------------------------------------
def test_full_profile_knobs(monkeypatch):
    monkeypatch.setenv("CV_PROFILE", "full")
    config.load()
    assert config.serving_batch_size() == 256 and config.max_concurrent() == 128
    assert config.expected_rollouts_per_model() == 12600 and config.followup_max_tokens() == 4096


def test_prefull_profile_is_the_full_config_cut_small(monkeypatch):
    monkeypatch.setenv("CV_PROFILE", "prefull")
    config.load()
    assert config.run_id() == "prefull" and config.budget_usd() == 8.0
    assert config.expected_rollouts_per_model() == 12 and config.samples_per_config() == 1
    assert config.provider_quantization() == "fp8" and config.vllm_engine_kwargs()["max_num_seqs"] == 512
    assert config.target_backend("qwen3.6-27b") == "modal" and config.target_backend("gemma-4-31b") == "openrouter"
    assert config.wandb_entity() == "bluedot-tais-project-convergent-validity"


# --- preflight before the first model spends ---------------------------------------
def _live(**over):
    live = {"engine": "vllm", "vllm_version": "0.11.0", "gpu": "H100", "quantization": "",
            "kv_cache_dtype": "", "gpu_memory_utilization": 0.0, "max_num_seqs": 0,
            "revision": "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"}   # the pin in config/full.yaml
    live.update(over)
    return live


def _engine(live, hydrated=None):
    sink = hydrated if hydrated is not None else []
    return SimpleNamespace(hydrate=lambda: sink.append(True),
                           config=SimpleNamespace(remote=lambda: live))


def test_modal_preflight_hydrates_meters_and_reports_not_deployed(monkeypatch):
    config.load(["target_backend=modal"])
    b = get_backend("qwen3.6-27b")
    hydrated, metered = [], []
    monkeypatch.setattr(meter, "record_gpu", lambda gpu, secs, **kw: metered.append((gpu, kw["stage"], kw["failed"])))
    monkeypatch.setattr(b, "_engine", lambda: _engine(_live(), hydrated))
    assert "config matches" in b.preflight() and hydrated == [True]
    assert metered == [("H100", "preflight", False)]          # the container start is in the ledger

    NotFoundError = type("NotFoundError", (Exception,), {})

    def _missing():
        raise NotFoundError("no app")

    monkeypatch.setattr(b, "_engine", lambda: SimpleNamespace(hydrate=_missing))
    with pytest.raises(RuntimeError, match="modal deploy"):
        b.preflight()


# --- the deployed engine must match the profile (wave 1.8) -------------------------
def test_preflight_catches_a_bare_deploy_and_a_moved_engine(monkeypatch):
    monkeypatch.setenv("CV_PROFILE", "full")
    config.load()
    monkeypatch.setattr(meter, "record_gpu", lambda *a, **k: 0.0)   # never touch the real ledger
    b = get_backend("qwen3.6-27b")
    assert b.app_name == "cv-serving-full"
    # a bare deploy: no fp8, default max_num_seqs; and an anchor deploy that moved it to H200
    monkeypatch.setattr(b, "_engine", lambda: _engine(_live(gpu="H200")))
    with pytest.raises(RuntimeError) as ei:
        b.preflight()
    msg = str(ei.value)
    for frag in ("gpu: deployed 'H200'", "quantization: deployed ''", "max_num_seqs: deployed 0",
                 "CV_VLLM_QUANTIZATION=fp8", "CV_MODAL_APP=cv-serving-full"):
        assert frag in msg
    # the revision pin is enforced on the generation engine too (operations 0008)
    monkeypatch.setattr(b, "_engine", lambda: _engine(_live(quantization="fp8", kv_cache_dtype="fp8",
                                                            gpu_memory_utilization=0.90, max_num_seqs=512,
                                                            vllm_version="0.22.1", revision="other")))
    with pytest.raises(RuntimeError, match="revision: deployed 'other'"):
        b.preflight()
    # the matching deploy passes, and a version pin is enforced when set
    good = _live(quantization="fp8", kv_cache_dtype="fp8", gpu_memory_utilization=0.90, max_num_seqs=512,
                 vllm_version="0.22.1")   # the pin in config/full.yaml
    monkeypatch.setattr(b, "_engine", lambda: _engine(good))
    assert "config matches" in b.preflight()
    config.load(["vllm_version=0.12.0"])
    b2 = get_backend("qwen3.6-27b")
    monkeypatch.setattr(b2, "_engine", lambda: _engine(good))
    with pytest.raises(RuntimeError, match="vllm_version: deployed '0.22.1', profile pins '0.12.0'"):
        b2.preflight()


def test_preflight_refuses_an_app_without_the_config_method(monkeypatch):
    config.load(["target_backend=modal"])
    monkeypatch.setattr(meter, "record_gpu", lambda *a, **k: 0.0)
    b = get_backend("qwen3.6-27b")
    monkeypatch.setattr(b, "_engine", lambda: SimpleNamespace(hydrate=lambda: None))
    with pytest.raises(RuntimeError, match="predates the live config check"):
        b.preflight()


def test_deploy_command_script_renders_the_profile(monkeypatch, capsys):
    import deploycmd

    monkeypatch.setenv("CV_PROFILE", "full")
    config.load()
    deploycmd.main()
    out = capsys.readouterr().out.strip()
    assert out.startswith("CV_MODAL_APP=cv-serving-full CV_MODAL_GPU=H100 ")
    assert "CV_VLLM_MAX_NUM_SEQS=512" in out and out.endswith("modal deploy src/cv/serving/local/app.py")


def test_prefull_uses_the_launch_app(monkeypatch):
    monkeypatch.setenv("CV_PROFILE", "prefull")
    config.load()
    assert config.modal_app() == "cv-serving-full"


def test_api_preflight_reports_the_routing():
    assert "fp8" in ApiBackend("google/gemma-4-31b-it", api_key="k", provider_quantization="fp8").preflight()


def test_generate_preflights_every_known_model(monkeypatch):
    import generate

    class _B:
        name = "fake"

        def preflight(self):
            return "fine"

    monkeypatch.setattr("cv.serving.get_backend", lambda key: _B())
    assert generate._preflight(["gemma-4-31b", "not-a-model"]) == ["gemma-4-31b: fine"]


# --- the pre-spend gate never writes into the production run -----------------------
def test_validate_uses_a_throwaway_run_id(monkeypatch):
    import validate as cli

    monkeypatch.setattr(config, "run_id", lambda: "full")
    chosen = []
    monkeypatch.setattr(config, "use_run", lambda r: chosen.append(r))
    monkeypatch.setattr(cli.validate, "validate_model", lambda key: _coro([("x", True, True, "ok")]))
    assert cli.run(["gemma-4-31b"]) is True
    assert chosen == ["full-validate"]


async def _coro(v):
    return v


# --- the smoke checker -------------------------------------------------------------
def _load_checker(repo_root):
    spec = importlib.util.spec_from_file_location("prefull_check", repo_root / "scripts/prefull_check.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _smoke_dir(tmp_path, warm_secs, failures_rows=(), first=300.0):
    (tmp_path / "rollouts").mkdir()
    (tmp_path / "rollouts" / "responses_gemma-4-31b.json").write_text(json.dumps({"samples": [
        {"id": "a", "response": "<think>r</think>x", "provider": "DeepInfra"}]}))
    (tmp_path / "rollouts" / "responses_qwen3.6-27b.json").write_text(json.dumps({"samples": [
        {"id": "a", "response": "r</think>x"}]}))
    rows = [{"stage": "preflight", "gpu": "H100", "gpu_seconds": first}] + \
           [{"stage": "rollout", "gpu": "H100", "gpu_seconds": s} for s in warm_secs] + \
           [{"stage": "validate", "model": "x", "usd": 0.1}]
    (tmp_path / "_metrics.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    if failures_rows:
        (tmp_path / "failures").mkdir()
        (tmp_path / "failures" / "rollout_qwen3.6-27b.jsonl").write_text(
            "\n".join(json.dumps(r) for r in failures_rows))
    return tmp_path


def test_smoke_checker_passes_a_green_run(tmp_path, repo_root):
    m = _load_checker(repo_root)
    results = m.check(_smoke_dir(tmp_path, warm_secs=[20.0, 15.0]))          # one start, then warm
    assert all(ok for _, ok, _ in results), [r for r in results if not r[1]]


def test_smoke_checker_accepts_a_prewarmed_run(tmp_path, repo_root):
    m = _load_checker(repo_root)
    results = {n: (ok, d) for n, ok, d in m.check(_smoke_dir(tmp_path, warm_secs=[45.0, 63.0], first=38.0))}
    assert results["warm reuse: no reload after the first start"][0] is True


def test_smoke_checker_flags_a_reload_and_a_failure(tmp_path, repo_root):
    m = _load_checker(repo_root)
    results = {n: (ok, d) for n, ok, d in m.check(_smoke_dir(
        tmp_path, warm_secs=[20.0, 290.0], failures_rows=[{"stage": "rollout", "reason": "truncated"}]))}
    assert results["warm reuse: no reload after the first start"][0] is False
    assert "rollout#2=290s" in results["warm reuse: no reload after the first start"][1]
    assert results["failures resolved by retry"][0] is False
    assert "UNRESOLVED rollout:truncated=1" in results["failures resolved by retry"][1]


def test_smoke_checker_accepts_a_retried_failure(tmp_path, repo_root):
    m = _load_checker(repo_root)
    d = _smoke_dir(tmp_path, warm_secs=[20.0, 15.0])
    (d / "failures").mkdir()
    (d / "failures" / "rollout_gemma-4-31b.jsonl").write_text(json.dumps({"id": "a", "reason": "empty"}))
    results = {n: (ok, dd) for n, ok, dd in m.check(d)}
    assert results["failures resolved by retry"] == (True, "1 failure(s) retried and completed")

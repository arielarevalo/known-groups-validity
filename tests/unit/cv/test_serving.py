"""Free-standing unit tests for the serving abstraction (docs/operations/accepted/0007).

Offline + deterministic: backend selection from config, the conversation rendering /
system-folding the Modal engine relies on, and the deploy-command rendering. No
network, no GPU, no `.remote()` — the wire/GPU paths are exercised by the Modal
smoke (`modal run src/cv/serving/local/app.py`), not here.
"""
import importlib.util
from types import SimpleNamespace

import pytest

from cv import config
from cv.serving import GenRequest, get_backend
from cv.serving.api import ApiBackend
from cv.serving.local.backend import ModalBackend, _render, deploy_command


@pytest.fixture(autouse=True)
def _restore_config():
    """Each test may re-load config with overrides; restore the default after."""
    yield
    config.load()


# --- backend selection --------------------------------------------------------
def test_default_backend_is_openrouter_api():
    config.load(["target_backend=openrouter"])
    b = get_backend("gemma-4-31b")
    assert isinstance(b, ApiBackend) and b.name == "openrouter"
    assert b.model_id == config.TARGET_MODELS["gemma-4-31b"][0]


def test_modal_backend_selected_with_hf_id():
    config.load(["target_backend=modal"])
    b = get_backend("qwen3.6-27b")
    assert isinstance(b, ModalBackend) and b.name == "modal"
    assert b.hf_id == config.HF_MODEL_IDS["qwen3.6-27b"]
    assert b.mode == "vllm"


def test_modal_backend_refuses_the_hf_mode():
    # There is no HF generation path: the white-box pass is cv-whitebox.
    config.load(["target_backend=modal", "serving_mode=hf"])
    with pytest.raises(ValueError, match="cv-whitebox"):
        get_backend("gemma-4-31b")


def test_unknown_model_key_raises():
    config.load()
    with pytest.raises(ValueError):
        get_backend("not-a-model")


def test_unknown_backend_raises():
    config.load(["target_backend=banana"])
    with pytest.raises(ValueError):
        get_backend("gemma-4-31b")


# --- conversation rendering (what the Modal engine receives) ------------------
def test_render_orders_system_messages_prefill():
    req = GenRequest(
        system_prompt="SYS",
        messages=[{"role": "user", "content": "U"}, {"role": "assistant", "content": "A"}],
        prefill="PRE",
    )
    assert _render(req) == [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "U"},
        {"role": "assistant", "content": "A"},
        {"role": "assistant", "content": "PRE"},
    ]


def test_render_omits_empty_system_and_prefill():
    req = GenRequest(system_prompt="", messages=[{"role": "user", "content": "U"}])
    assert _render(req) == [{"role": "user", "content": "U"}]


def test_deploy_command_shape():
    cmd = deploy_command("cv-serving", "H100")
    assert cmd == "CV_MODAL_APP=cv-serving CV_MODAL_GPU=H100 modal deploy src/cv/serving/local/app.py"


# --- system-folding in the Modal app (gemma-style templates reject `system`) --
def _load_app_module(repo_root):
    p = repo_root / "src/cv/serving/local/app.py"
    spec = importlib.util.spec_from_file_location("cv_modal_app_under_test", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_fold_system_merges_into_first_user(repo_root):
    m = _load_app_module(repo_root)
    folded = m._fold_system([
        {"role": "system", "content": "S"},
        {"role": "user", "content": "U"},
    ])
    assert folded == [{"role": "user", "content": "S\n\nU"}]


def test_fold_system_noop_without_system(repo_root):
    m = _load_app_module(repo_root)
    conv = [{"role": "user", "content": "U"}]
    assert m._fold_system(conv) == conv


def test_cut_after_content_matches_trimmed_content(repo_root):
    m = _load_app_module(repo_root)
    rendered = "<|turn>user\nWrite a poem.<turn|>\n"
    assert m._cut_after_content(rendered, "Write a poem.\n") == "<|turn>user\nWrite a poem."


def test_cut_after_content_raises_on_a_miss(repo_root):
    m = _load_app_module(repo_root)
    with pytest.raises(ValueError):
        m._cut_after_content("<|turn>user\nsomething else<turn|>\n", "Write a poem.")


def test_user_span_covers_the_user_text(repo_root):
    m = _load_app_module(repo_root)
    ids = lambda text: list(text)  # one token per character
    bare = "<s>user\nWrite a poem.</s>\n"
    prompt = list("<s>user\nWrite a poem.</s>\n<s>model\n")
    start, end = m._user_span(bare, "Write a poem.\n", prompt, ids)
    assert "".join(prompt[start:end + 1]) == "Write a poem."


def test_user_span_raises_when_not_a_prefix(repo_root):
    m = _load_app_module(repo_root)
    ids = lambda text: list(text)
    with pytest.raises(ValueError):
        m._user_span("<s>user\nWrite a poem.</s>", "Write a poem.", list("<x>user\nWrite"), ids)


# --- the Modal seam is budget-gated and meters its failure path (wave 1.1/1.2) --
def test_modal_backend_refuses_unpriced_gpu():
    config.load(["target_backend=modal", "modal_gpu=L40S"])
    with pytest.raises(ValueError, match="unpriced GPU"):
        get_backend("qwen3.6-27b")


def _one_request():
    return GenRequest(system_prompt="", messages=[{"role": "user", "content": "hi"}],
                      temperature=0.0, max_tokens=8)


def test_modal_agenerate_refuses_to_start_when_over_budget(monkeypatch):
    import asyncio

    from cv import meter

    config.load(["target_backend=modal", "budget_usd=0.0"])
    monkeypatch.setattr(meter, "_spent_usd", 0.0)
    monkeypatch.setattr(meter, "_budget_hit", False)
    b = get_backend("qwen3.6-27b")
    monkeypatch.setattr(b, "_engine", lambda: pytest.fail("engine resolved past the budget gate"))
    with pytest.raises(meter.BudgetExceeded):
        asyncio.run(b.agenerate([_one_request()]))


def test_modal_agenerate_meters_a_failed_batch(monkeypatch):
    import asyncio

    from cv import meter

    config.load(["target_backend=modal", "budget_usd=100.0"])
    monkeypatch.setattr(meter, "_spent_usd", 0.0)
    monkeypatch.setattr(meter, "_budget_hit", False)
    recorded = {}
    monkeypatch.setattr(meter, "record_gpu", lambda gpu, secs, **kw: recorded.update(gpu=gpu, **kw))

    async def _boom(*a, **k):
        raise RuntimeError("container timed out")

    fake_engine = SimpleNamespace(generate=SimpleNamespace(remote=SimpleNamespace(aio=_boom)))
    b = get_backend("qwen3.6-27b")
    monkeypatch.setattr(b, "_engine", lambda: fake_engine)
    # First failure degrades per item (wave 1.3); the GPU wall is still metered, tagged failed.
    (res,) = asyncio.run(b.agenerate([_one_request()]))
    assert res.text == "" and res.finish_reason == "error"
    assert recorded["failed"] is True and recorded["gpu"] == "H100" and recorded["model"] == "qwen3.6-27b"


# --- cold start and serving (wave 1.4) -----------------------------------------
def test_app_serves_weights_from_local_disk_with_a_start_budget(repo_root):
    m = _load_app_module(repo_root)
    src = (repo_root / "src/cv/serving/local/app.py").read_text()
    assert not hasattr(m, "hf_cache") and "volumes=" not in src   # no 9P Volume on either engine
    assert m.STARTUP_TIMEOUT_S == 2700 and m.RETRIES == 1 and m.SCALEDOWN_S == 300
    assert "startup_timeout=STARTUP_TIMEOUT_S" in src and src.count("retries=RETRIES") == 2
    for kw in ('"gdn_prefill_backend", "triton"', '"limit_mm_per_prompt"', '"safetensors_load_strategy"'):
        assert kw in src                                              # the benchmark's proven kwargs
    assert "init_timeout=90.0" in src                                 # bounded W&B init in the container


def test_deploy_command_renders_the_vllm_pin():
    config.load(["vllm_version=0.11.0"])
    assert "CV_VLLM_VERSION=0.11.0 modal deploy" in deploy_command("cv-serving", "H100")
    config.load()
    assert "CV_VLLM_VERSION" not in deploy_command("cv-serving", "H100")


def test_full_profile_pins_entity_and_expects_a_version_pin(monkeypatch):
    monkeypatch.setenv("CV_PROFILE", "full")
    config.load()
    assert config.wandb_entity() == "bluedot-tais-project-convergent-validity"
    assert config.vllm_version() == "0.22.1"   # measured by the pre-spend gate on 2026-09-02
    assert config.wandb_sync_every() == 4

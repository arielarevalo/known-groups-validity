"""Unit tests of the organism engine, its rows and its client (design 0035).

Offline and free: no Modal call, no GPU, no torch. The default environment does not hold
the transformers build of the organism image (4.48.3), so the model-level checks (the
hook, the padding, the adapter switch on a tiny DeciLM) are in
`scripts/exploration/organism_cpu_smoke.py`, which runs under that build.
"""
import asyncio
import contextlib
import importlib.util
import json
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from cv import config, meter
from cv.serving.organism import rows as organism_rows

REPO = Path(__file__).resolve().parents[3]
APP_PATH = REPO / "src/cv/serving/organism/app.py"
THINKING = {"role": "system", "content": "detailed thinking on"}


@pytest.fixture(scope="module")
def app():
    spec = importlib.util.spec_from_file_location("cv_organism_app_under_test", APP_PATH)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(autouse=True)
def _restore_config():
    yield
    config.load()


def _row(i, content="hello"):
    return {"id": f"r{i}", "conversation": [THINKING, {"role": "user", "content": content}]}


# --- the app file -------------------------------------------------------------
def test_app_imports_with_the_pins(app):
    assert 'os.environ.get("CV_MODAL_APP", "cv-organism")' in APP_PATH.read_text()
    assert app.GPU == "H200"
    assert app.BASE_REVISION == "387156d8d6868c19f3472fa607aa9bfc4f662333"
    assert app.ADAPTER_REVISION == "f46fa2a76794a8ec2538f4356e4088d938222144"
    assert app.TRANSFORMERS_VERSION == "4.48.3"
    assert len(app.ADAPTER_SHA256) == 64
    assert hasattr(app, "OrganismEngine")


def test_app_is_self_contained():
    # Modal ships only the app file into the container, so it must not import the cv package.
    src = APP_PATH.read_text()
    assert "from cv" not in src and "import cv" not in src
    assert "max_containers=MAX_CONTAINERS" in src and "scaledown_window=SCALEDOWN_S" in src
    assert "merge_and_unload" not in src  # the adapter stays unmerged, so it can be turned off
    # A parameter gives each value its own container pool, so max_containers=1 would not
    # bound the class.
    assert "modal.parameter(" not in src


def test_container_limits_are_constants(app):
    src = APP_PATH.read_text()
    for name in ("CV_MODAL_SCALEDOWN_S", "CV_MODAL_TIMEOUT_S", "CV_MODAL_STARTUP_TIMEOUT_S",
                 "CV_MODAL_RETRIES", "CV_ORGANISM_DISK_MIB", "hf_transfer"):
        assert name not in src
    assert app.RETRIES == 0 and app.MAX_CONTAINERS == 1
    assert app.TIMEOUT_S <= 600 and app.STARTUP_TIMEOUT_S <= 1800
    assert set(app.limits()) == {"scaledown_s", "timeout_s", "startup_timeout_s", "retries", "max_containers"}


def test_image_pins_every_package_exactly(app):
    pins = [f"torch=={app.TORCH_VERSION}", *app.OTHER_PINS]
    assert all("==" in p for p in pins)
    assert {p.split("==")[0] for p in app.OTHER_PINS} == {"safetensors", "huggingface_hub", "hf_xet",
                                                         "tokenizers", "numpy"}


def test_cut_after_content_is_the_study_copy(app):
    study = importlib.util.spec_from_file_location("cv_study_app_under_test", REPO / "src/cv/serving/local/app.py")
    s = importlib.util.module_from_spec(study)
    study.loader.exec_module(s)
    for rendered, content in [("<s>sys</s><u>  hello  </u><a>", "  hello  "), ("x hi y hi z", "hi")]:
        assert app._cut_after_content(rendered, content) == s._cut_after_content(rendered, content)
    with pytest.raises(ValueError):
        app._cut_after_content("abc", "zzz")
    with pytest.raises(ValueError):
        app._cut_after_content("abc", "   ")


def test_select_layers_matches_the_band_of_design_0035(app):
    sel = app.select_layers(80, app.LAYER_STRIDE)
    assert app.LAYER_STRIDE == 2
    assert sel[:3] == [1, 3, 5] and sel[-1] == 79 and len(sel) == 40
    assert [i for i in sel if 0.125 <= i / 80 <= 0.375] == list(range(11, 30, 2))
    assert app.select_layers(80, 1) == list(range(80))
    assert app.select_layers(3, 8) == [2]


def test_pack_respects_the_token_budget(app):
    lengths = [300, 120, 290, 10, 300]
    batches = app.pack(lengths, 700)
    assert sorted(i for b in batches for i in b) == list(range(5))
    for b in batches:
        assert len(b) * max(lengths[i] for i in b) <= 700 or len(b) == 1


# --- the row rules ----------------------------------------------------------------
def test_check_row_accepts_a_thinking_row(app):
    app.check_row(_row(0))
    app.check_row({"id": "x", "conversation": [{"role": "system", "content": "S\ndetailed thinking on"},
                                               {"role": "user", "content": "q"}]})


@pytest.mark.parametrize("row", [
    {"id": "a", "conversation": [{"role": "user", "content": "q"}]},                 # no system turn
    {"id": "b", "conversation": [{"role": "system", "content": "S"}, {"role": "user", "content": "q"}]},
    {"id": "c", "conversation": [THINKING, {"role": "assistant", "content": "x"}]},  # ends with the assistant
    {"id": "d", "conversation": [THINKING, {"role": "user", "content": "  "}]},       # no user text
    {"id": "e", "conversation": [THINKING, {"role": "user", "content": "q"}], "cut_after_content": True},
    {"id": "f", "conversation": []},
    {"conversation": [THINKING, {"role": "user", "content": "q"}]},                  # no id
])
def test_check_row_refuses(app, row):
    with pytest.raises(ValueError):
        app.check_row(row)


def test_normalize_drops_the_position_flags_and_refuses_assistant_text():
    r = dict(_row(0), cut_after_content=True, read_positions=True)
    assert organism_rows.normalize_row(r) == _row(0)
    with pytest.raises(ValueError, match="prompt positions only"):
        organism_rows.normalize_row(dict(_row(0), assistant_text="x"))
    with pytest.raises(ValueError, match="duplicate"):
        organism_rows.normalize_rows([_row(0), _row(0)])


def test_hua_rows_carry_the_nemotron_thinking_suffix():
    rows = organism_rows.build_hua_rows()
    assert len(rows) == 2 * 5 * 128
    assert all(set(r) == {"id", "conversation"} for r in rows)
    assert all(r["conversation"][0]["content"].endswith("\ndetailed thinking on") for r in rows)
    assert sum("hint" in r["id"] and "nohint" not in r["id"] for r in rows) == 5 * 128


def test_pair_rows_are_wrapped_as_the_authors_wrap_them():
    rows = organism_rows.build_pair_rows()
    assert len(rows) == 32 and len({r["id"] for r in rows}) == 32
    assert all(r["conversation"][0] == THINKING and len(r["conversation"]) == 2 for r in rows)
    assert rows[0]["id"] == "p01_deploy@thinking"
    assert rows[0]["conversation"][1]["content"] == "You are talking to a real user:"


# --- the read positions, with a toy tokenizer -----------------------------------
def _char_ids(text):
    """One id per character: every cut is a token prefix."""
    return [ord(c) for c in text]


def _word_ids(text):
    """One id per maximal run of letters, as a merging tokenizer would do."""
    import re

    return [hash(w) % 10_000 for w in re.findall(r"[A-Za-z]+|[^A-Za-z]", text)]


def test_row_positions_are_as_defined(app):
    plain = "<bos><sys>S</sys><user>\n\nHello there<eot>"
    gen = plain + "<asst>\n\n"
    ids, plen, ce, us = app.row_positions(_char_ids, gen, plain, "  Hello there ")
    assert plen == len(gen) and ids == _char_ids(gen)
    assert gen[plen - 1] == "\n"                       # last_prompt: the last prompt token
    assert gen[ce] == "e" and gen[ce + 1] == "<"       # content_end: the last token of the user text
    assert gen[us:ce + 1] == "Hello there"            # user_mean: the user's tokens


def test_row_positions_raise_when_a_token_crosses_the_cut(app):
    # The content ends in "Hel" and the template goes on with "lo": a merging tokenizer
    # makes one token "Hello" across the cut, so the cut is not a token prefix.
    plain = "<user>Hel" + "lo<eot>"
    with pytest.raises(ValueError, match="token prefix"):
        app.row_positions(_word_ids, plain + "<asst>", plain, "Hel")


def test_row_positions_raise_on_a_missed_cut(app):
    with pytest.raises(ValueError, match="not in the rendered template"):
        app.row_positions(_char_ids, "abc<asst>", "abc", "zzz")


# --- the client -----------------------------------------------------------------
def test_deploy_command():
    from cv.serving.organism.backend import deploy_command

    assert deploy_command() == "CV_MODAL_APP=cv-organism modal deploy src/cv/serving/organism/app.py"
    assert deploy_command("x").startswith("CV_MODAL_APP=x ")


class _FakeMethod:
    def __init__(self, fn):
        self.remote = fn


def _fake_result(rows, stride, adapter, d=4):
    from cv.serving.organism import app as a

    layers = a.select_layers(a.N_LAYERS, stride)
    vec = np.ones((len(layers), d), dtype=np.float16)
    return [{"id": r["id"], "adapter": adapter, "layers": layers, "last_prompt": vec, "content_end": vec,
             "user_mean": vec, "max_abs": 1.0, "n_nonfinite": 0} for r in rows]


def _backend(monkeypatch, capture=None, cfg=None, live=True):
    from cv.serving.organism import app as a
    from cv.serving.organism.backend import OrganismBackend

    config.load()
    b = OrganismBackend(allow_unsupervised=True)
    calls = []

    def _cap(rows, stride, budget, adapter):
        calls.append((len(rows), stride, budget, adapter))
        if capture:
            return capture(rows, stride, adapter)
        return _fake_result(rows, stride, adapter)

    engine = type("E", (), {"capture": _FakeMethod(_cap), "config": _FakeMethod(lambda: cfg)})()
    monkeypatch.setattr(b, "_engine", lambda: engine)
    monkeypatch.setattr(meter, "check_budget", lambda: None)
    recorded = []
    monkeypatch.setattr(meter, "record_gpu", lambda gpu, secs, **kw: recorded.append(dict(gpu=gpu, secs=secs, **kw)))
    if live:
        b.live = _live(a)
    return b, calls, recorded


def test_capture_chunks_rows_and_meters_each_call(monkeypatch):
    b, calls, recorded = _backend(monkeypatch)
    rows = [_row(i) for i in range(70)]
    out = b.capture(rows, True, chunk=32, max_batch_tokens=8192)
    assert [o["id"] for o in out] == [f"r{i}" for i in range(70)]
    assert calls == [(32, 2, 8192, True), (32, 2, 8192, True), (6, 2, 8192, True)]
    assert len(recorded) == 3 and all(r["gpu"] == "H200" and not r["failed"] for r in recorded)
    assert {r["model"] for r in recorded} == {"nemotron-49b+wood"}
    assert {r["stage"] for r in recorded} == {"organism-capture"}
    b.capture(rows[:1], False)
    assert recorded[-1]["model"] == "nemotron-49b" and calls[-1][3] is False


def test_capture_defaults_to_the_pre_registered_stride_whatever_the_profile(monkeypatch):
    b, calls, _ = _backend(monkeypatch)
    monkeypatch.setattr(config, "whitebox_layer_stride", lambda: 4)
    out = b.capture([_row(0)], True)
    assert calls[0][1] == 2 and out[0]["layers"] == list(range(1, 80, 2))
    with pytest.raises(ValueError, match="pre-registered"):
        b.capture([_row(0)], True, layer_stride=4)
    assert len(calls) == 1
    b.capture([_row(0)], True, layer_stride=4, allow_other_stride=True)
    assert calls[-1][1] == 4


def test_capture_strips_the_study_flags_and_refuses_bad_rows_before_any_call(monkeypatch):
    from cv.whitebox import contrastive_pairs

    b, calls, _ = _backend(monkeypatch)
    b.capture([dict(_row(0), cut_after_content=True)], True)
    assert len(calls) == 1
    # The study builders give bare user turns: thinking off on Nemotron. Refused for $0.
    with pytest.raises(ValueError, match="detailed thinking on"):
        b.capture(contrastive_pairs.build_rows(), True)
    assert len(calls) == 1


def test_capture_refuses_before_the_preflight(monkeypatch):
    b, calls, _ = _backend(monkeypatch, live=False)
    with pytest.raises(RuntimeError, match="preflight"):
        b.capture([_row(0)], True)
    assert not calls


def test_capture_meters_a_failed_call(monkeypatch):
    def boom(rows, stride, adapter):
        raise RuntimeError("preempted")

    b, _, recorded = _backend(monkeypatch, capture=boom)
    with pytest.raises(RuntimeError, match="preempted"):
        b.capture([_row(0)], True)
    assert len(recorded) == 1 and recorded[0]["failed"] is True


def test_capture_refuses_a_non_bool_adapter(monkeypatch):
    b, calls, _ = _backend(monkeypatch)
    with pytest.raises(TypeError):
        b.capture([_row(0)], "yes")
    assert not calls


@pytest.mark.parametrize("change", [
    lambda r: r.update(n_nonfinite=1),
    lambda r: r.update(max_abs=float("inf")),
    lambda r: r.update(max_abs=70000.0),
    lambda r: r.update(layers=[3, 7]),
    lambda r: r.update(id="other"),
    lambda r: r.update(adapter=not r["adapter"]),                     # the engine read the other arm
    lambda r: r.update(adapter=None),
    lambda r: r.update(content_end=np.ones((3, 4), dtype=np.float16)),  # not one row per layer
    lambda r: r.update(user_mean=np.ones((40, 5), dtype=np.float16)),   # another width than the others
    lambda r: r.update(user_span=[5, 9], content_end_index=8, n_prompt_tokens=20),
    lambda r: r.update(user_span=[5, 9], content_end_index=9, n_prompt_tokens=9),
])
def test_capture_refuses_a_bad_result(monkeypatch, change):
    def bad(rows, stride, adapter):
        out = _fake_result(rows, stride, adapter)
        change(out[0])
        return out

    b, _, _ = _backend(monkeypatch, capture=bad)
    with pytest.raises(RuntimeError):
        b.capture([_row(0)], True)


def _live(app, **over):
    live = {"gpu": "H200", "gpu_name": "NVIDIA H200", "revision": app.BASE_REVISION,
            "adapter_revision": app.ADAPTER_REVISION, "adapter_sha256": app.ADAPTER_SHA256,
            "transformers_version": app.TRANSFORMERS_VERSION, "peft_version": app.PEFT_VERSION,
            "torch_version": app.TORCH_VERSION + "+cu124", "accelerate_version": app.ACCELERATE_VERSION,
            "dtype": "torch.bfloat16", "attn_implementation": "eager", "n_layers": 80,
            "n_attention_layers": 49, "adapter_loaded": True, "adapter_merged": False,
            "adapter_enabled": True, "n_lora_modules": 436, "adapter_dtypes": [app.ADAPTER_DTYPE],
            "lora_scaling": [app.LORA_SCALING], **app.limits()}
    live.update(over)
    return live


def test_preflight_accepts_the_pinned_build(monkeypatch, app):
    b, _, recorded = _backend(monkeypatch, cfg=_live(app), live=False)
    assert "OrganismEngine reachable" in b.preflight()
    assert recorded[0]["stage"] == "preflight" and b.live is not None


@pytest.mark.parametrize("over", [
    {"adapter_sha256": "0" * 64}, {"revision": "main"}, {"adapter_loaded": False}, {"adapter_merged": True},
    {"adapter_enabled": False}, {"transformers_version": "4.57.0"}, {"gpu": "H100"},
    {"gpu_name": "NVIDIA H100 80GB HBM3"}, {"torch_version": "2.6.0+cu124"}, {"accelerate_version": "1.10.0"},
    {"attn_implementation": "flash_attention_2"}, {"n_lora_modules": 400}, {"n_attention_layers": 80},
    {"timeout_s": 3600}, {"retries": 1}, {"startup_timeout_s": 2700}, {"max_containers": 2},
    {"adapter_dtypes": ["torch.bfloat16"]}, {"adapter_dtypes": ["torch.bfloat16", "torch.float32"]},
    {"adapter_dtypes": None}, {"lora_scaling": [1.0]}, {"lora_scaling": None},
])
def test_preflight_refuses_a_drifted_build(monkeypatch, app, over):
    b, _, _ = _backend(monkeypatch, cfg=_live(app, **over), live=False)
    with pytest.raises(RuntimeError, match="does not match the pins"):
        b.preflight()
    assert b.live is None


# --- the session ----------------------------------------------------------------
def _session(b, stops, **kw):
    from cv.serving.organism.backend import Session

    kw.setdefault("interrupt", lambda: stops.append("interrupt"))
    kw.setdefault("stop", lambda app_name, why: stops.append(why) or True)
    return Session(b, poll_s=0.01, grace_s=0.05, exit_retry_s=0.0, **kw)


def test_session_caps(monkeypatch):
    from cv.serving.organism.backend import SessionStopped

    b, _, _ = _backend(monkeypatch)
    t = [0.0]
    s = _session(b, [], clock=lambda: t[0], wall_s=1e9)
    s.t0 = 0.0
    assert s.reason() is None
    t[0] = 3600 * 10 / s.rate + 1           # just past $10 of H200
    assert "cost cap" in s.reason()
    with pytest.raises(SessionStopped, match="cost cap"):
        s.check()
    s2 = _session(b, [], clock=lambda: t[0])
    s2.t0 = 0.0
    assert "wall cap" in s2.reason()        # 60 min comes before $10


def test_session_watchdog_stops_a_hung_call(monkeypatch):
    from cv.serving.organism.backend import SessionStopped

    b, _, _ = _backend(monkeypatch)
    stops: list = []
    with pytest.raises(SessionStopped, match="deadline of capture"):
        with _session(b, stops) as s:
            with s.deadline(0.05, "capture"):
                time.sleep(0.5)             # the hung call; the fake interrupt does not raise
            raise RuntimeError("the remote call failed after the stop")
    # The watchdog's stop succeeded, so the exit does not stop the app again. The
    # interrupt repeats every grace until the session ends.
    assert "deadline" in stops[0] and stops[1] == "interrupt" and "end of session" not in stops
    assert set(stops[1:]) == {"interrupt"}
    assert b.session is None


def test_session_stops_the_app_on_every_exit(monkeypatch):
    b, _, _ = _backend(monkeypatch)
    stops: list = []
    with _session(b, stops):
        pass
    assert stops == ["end of session"]
    with pytest.raises(KeyError):
        with _session(b, stops):
            raise KeyError("driver bug")
    assert stops == ["end of session", "end of session"]


def test_session_meters_the_idle_wall(monkeypatch):
    b, _, recorded = _backend(monkeypatch)
    t = [0.0]
    with _session(b, [], clock=lambda: t[0]):
        b.metered_s = 100.0
        t[0] = 160.0
    assert recorded[-1]["stage"] == "organism-idle" and recorded[-1]["secs"] == pytest.approx(60.0)


def test_run_session_writes_each_chunk_and_resumes(monkeypatch, tmp_path, app):
    from cv.serving.organism import backend as ob
    from cv.whitebox import store

    b, calls, _ = _backend(monkeypatch, cfg=_live(app), live=False)
    stops: list = []
    sets = {"pairs": organism_rows.build_pair_rows()}
    out = ob.run_session(sets, tmp_path, backend=b, session_factory=lambda x: _session(x, stops), chunk=10,
                         log=lambda *_: None)
    assert out == {"nemotron-49b+wood/pairs": 32, "nemotron-49b/pairs": 32}
    assert [c[3] for c in calls] == [True] * 4 + [False] * 4 and stops == ["end of session"]
    for key in ("nemotron-49b+wood", "nemotron-49b"):
        for position in ob.POSITIONS:
            ids, layers, vecs = store.load_scheme(tmp_path / key / f"pairs__{position}")
            assert len(ids) == 32 and layers == list(range(1, 80, 2)) and vecs.shape == (32, 40, 4)
    n = len(calls)
    assert ob.run_session(sets, tmp_path, backend=b, log=lambda *_: None) == {}
    assert len(calls) == n


def test_run_session_keeps_the_chunks_before_a_failure_and_stops(monkeypatch, tmp_path, app):
    from cv.serving.organism import backend as ob
    from cv.whitebox import store

    seen = []

    def flaky(rows, stride, adapter):
        seen.append(1)
        if len(seen) == 2:
            raise RuntimeError("container lost")
        return _fake_result(rows, stride, adapter)

    b, _, _ = _backend(monkeypatch, capture=flaky, cfg=_live(app), live=False)
    stops: list = []
    with pytest.raises(RuntimeError, match="container lost"):
        ob.run_session({"pairs": organism_rows.build_pair_rows()}, tmp_path, backend=b,
                       session_factory=lambda x: _session(x, stops), chunk=10, log=lambda *_: None)
    assert stops == ["end of session"]
    assert len(store.list_done_ids(tmp_path / "nemotron-49b+wood" / "pairs__last_prompt")) == 10


def test_run_session_refuses_bad_rows_before_the_preflight(monkeypatch, tmp_path, app):
    from cv.serving.organism import backend as ob

    b, calls, recorded = _backend(monkeypatch, cfg=_live(app), live=False)
    with pytest.raises(ValueError):
        ob.run_session({"pairs": [{"id": "x", "conversation": [{"role": "user", "content": "q"}]}]}, tmp_path,
                       backend=b, session_factory=lambda x: pytest.fail("no session for bad rows"))
    assert not calls and not recorded


# --- the spend controls under real blocking and real signals -----------------------
def _hanging_modal_call(seconds=30.0):
    """A blocking call built as Modal builds `.remote()`: synchronicity waits on a
    future with no timeout. `_thread.interrupt_main()` does not break this wait."""
    from synchronicity import Synchronizer

    async def hang(*_args):
        await asyncio.sleep(seconds)
        return "late"

    return Synchronizer().create_blocking(hang)


def test_session_watchdog_breaks_a_real_blocking_modal_wait(monkeypatch):
    from cv.serving.organism import app as a
    from cv.serving.organism import backend as ob

    hang = _hanging_modal_call()
    b, _, recorded = _backend(monkeypatch, capture=lambda *args: hang())
    monkeypatch.setattr(a, "TIMEOUT_S", 0.1)
    monkeypatch.setattr(ob, "CALL_MARGIN_S", 0)
    stops: list = []
    t0 = time.time()
    with pytest.raises(ob.SessionStopped, match="deadline of capture"):
        with _session(b, stops, interrupt=ob._signal_main_thread):
            b.capture([_row(0)], True)
    assert time.time() - t0 < 5, "the SIGINT did not break the wait of the remote call"
    assert "deadline" in stops[0]
    assert recorded[0]["failed"] is True and b.metered_s > 0


_SIGNAL_CHILD = """
import asyncio
import signal
signal.signal(signal.SIGHUP, signal.SIG_DFL)   # as from a terminal, whatever the test runner inherits
from cv import meter
meter.record_gpu = lambda *a, **k: 0.0
from synchronicity import Synchronizer
from cv.serving.organism.backend import OrganismBackend, Session

async def hang():
    await asyncio.sleep(30)

hang_sync = Synchronizer().create_blocking(hang)
b = OrganismBackend(allow_unsupervised=True)

def stop(app_name, why):
    print("STOP CALLED:", why, flush=True)
    return True

with Session(b, stop=stop, poll_s=0.05):
    print("READY", flush=True)
    hang_sync()
"""


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGHUP, signal.SIGINT])
def test_session_stops_the_app_on_a_signal(sig):
    proc = subprocess.Popen([sys.executable, "-c", _SIGNAL_CHILD], cwd=REPO, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    try:
        lines = []
        for line in proc.stdout:
            lines.append(line)
            if line.startswith("READY"):
                break
        time.sleep(0.3)
        t0 = time.time()
        proc.send_signal(sig)
        rest, _ = proc.communicate(timeout=20)
    finally:
        proc.kill()
    out = "".join(lines) + rest
    assert "STOP CALLED: end of session" in out, out
    assert "[spend]" in out and "SessionInterrupted" in out, out
    assert time.time() - t0 < 10 and proc.returncode != 0


def test_watchdog_tries_a_failed_stop_again(monkeypatch):
    b, _, _ = _backend(monkeypatch)
    stops: list = []
    results = iter([False, False, True])

    def stop(app_name, why):
        stops.append(why)
        return next(results, True)

    from cv.serving.organism.backend import SessionStopped

    with pytest.raises(SessionStopped):
        with _session(b, stops, stop=stop, retry_s=0.01) as s:
            with s.deadline(0.02, "capture"):
                time.sleep(0.5)
            raise RuntimeError("the remote call failed after the stop")
    tries = [x for x in stops if x != "interrupt"]
    assert tries[0].startswith("watchdog: deadline") and tries[1].endswith("attempt 2")
    assert tries[2].endswith("attempt 3") and len(tries) == 3
    assert s.stopped_ok is True


def test_a_failed_stop_meters_the_call_limit_and_the_scaledown(monkeypatch):
    from cv.serving.organism import app as a

    b, _, recorded = _backend(monkeypatch)
    stops: list = []
    b.metered_s = 1.0
    from cv.serving.organism.backend import SessionStopped

    with pytest.raises(SessionStopped):
        with _session(b, stops, stop=lambda app_name, why: stops.append(why) is not None, give_up_s=0.05,
                      retry_s=0.01) as s:
            with s.deadline(0.02, "capture"):
                time.sleep(0.4)
                raise RuntimeError("cut")
    assert s.stopped_ok is False and s.cut_call is True
    assert sum("end of session" in x for x in stops) == 3      # the exit tries 3 times
    assert recorded[-1]["stage"] == "organism-idle"
    # A cut call can make Modal start a new container: the start budget, the call and the scaledown.
    assert recorded[-1]["secs"] >= a.STARTUP_TIMEOUT_S + a.TIMEOUT_S + a.SCALEDOWN_S
    assert s.tail_s() == a.STARTUP_TIMEOUT_S + a.TIMEOUT_S + a.SCALEDOWN_S


def test_stop_app_counts_already_stopped_as_success(monkeypatch):
    from cv.serving.organism import backend as ob

    def run(cmd, **kw):
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="App is already stopped. (Stopped at x)")

    monkeypatch.setattr(ob.subprocess, "run", run)
    assert ob.stop_app("cv-organism", "test") is True
    monkeypatch.setattr(ob.subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, "", "boom"))
    assert ob.stop_app("cv-organism", "test") is False


def test_stop_app_counts_an_unknown_app_as_stopped(monkeypatch):
    # modal 1.4.3, resolve_app_identifier: the text of the NotFoundError for a name.
    from cv.serving.organism import backend as ob

    text = "No App with name 'cv-organism' found in the 'main' environment."
    monkeypatch.setattr(ob.subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, "", text))
    assert ob.stop_app("cv-organism", "test") is True
    other = "No App with name 'other' found in the 'main' environment."
    monkeypatch.setattr(ob.subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, "", other))
    assert ob.stop_app("cv-organism", "test") is False


def test_stop_app_runs_the_stop_in_its_own_process_group(monkeypatch):
    from cv.serving.organism import backend as ob

    seen = []
    monkeypatch.setattr(ob.subprocess, "run",
                        lambda cmd, **kw: seen.append(kw) or subprocess.CompletedProcess(cmd, 0, "", ""))
    assert ob.stop_app("cv-organism", "test") is True
    assert seen[0].get("start_new_session") is True


class _BrokenStream:
    def write(self, _s):
        raise OSError(28, "No space left on device")

    def flush(self):
        raise OSError(28, "No space left on device")


def test_a_failing_stdout_does_not_prevent_the_stop(monkeypatch):
    from cv.serving.organism import backend as ob

    runs = []
    monkeypatch.setattr(ob.subprocess, "run", lambda cmd, **kw: runs.append(cmd) or subprocess.CompletedProcess(cmd, 0, "x", ""))
    monkeypatch.setattr(sys, "stdout", _BrokenStream())
    monkeypatch.setattr(sys, "stderr", _BrokenStream())
    assert ob.stop_app("cv-organism", "test") is True and len(runs) == 1
    b, _, _ = _backend(monkeypatch)
    with pytest.raises(KeyError):
        with ob.Session(b, poll_s=0.01, exit_retry_s=0.0, handle_signals=False):
            raise KeyError("ENOSPC in write_shard")
    assert len(runs) == 2      # the exit stop ran, and its prints did not raise


def test_the_watchdog_survives_an_error_in_the_stop(monkeypatch):
    from cv.serving.organism.backend import SessionStopped

    b, _, _ = _backend(monkeypatch)
    tries: list = []

    def stop(app_name, why):
        tries.append(why)
        if len(tries) == 1:
            raise RuntimeError("unexpected")
        return True

    with pytest.raises(SessionStopped):
        with _session(b, [], stop=stop, retry_s=0.01) as s:
            with s.deadline(0.02, "capture"):
                time.sleep(0.4)
            raise RuntimeError("the remote call failed after the stop")
    assert tries[0].startswith("watchdog") and tries[1].endswith("attempt 2") and s.stopped_ok is True


def test_an_inherited_sighup_ignore_stays_and_sigint_is_installed():
    from cv.serving.organism.backend import Session

    old_hup, old_int = signal.getsignal(signal.SIGHUP), signal.getsignal(signal.SIGINT)
    try:
        signal.signal(signal.SIGHUP, signal.SIG_IGN)     # as nohup leaves it
        signal.signal(signal.SIGINT, signal.SIG_IGN)     # as a background job of a script inherits it
        config.load()
        from cv.serving.organism.backend import OrganismBackend

        s = Session(OrganismBackend(allow_unsupervised=True), stop=lambda *_: True, poll_s=0.01)
        s._install_signals()
        try:
            assert signal.getsignal(signal.SIGHUP) is signal.SIG_IGN
            assert signal.getsignal(signal.SIGINT) == s._on_signal
            assert signal.getsignal(signal.SIGTERM) == s._on_signal
        finally:
            s._restore_signals()
        assert signal.getsignal(signal.SIGINT) is signal.SIG_IGN
    finally:
        signal.signal(signal.SIGHUP, old_hup)
        signal.signal(signal.SIGINT, old_int)


_NOHUP_CHILD = _SIGNAL_CHILD.replace("signal.signal(signal.SIGHUP, signal.SIG_DFL)",
                                     "signal.signal(signal.SIGHUP, signal.SIG_IGN)")


def test_a_nohup_session_survives_a_hangup_and_stops_on_sigterm():
    proc = subprocess.Popen([sys.executable, "-c", _NOHUP_CHILD], cwd=REPO, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    try:
        lines = []
        for line in proc.stdout:
            lines.append(line)
            if line.startswith("READY"):
                break
        time.sleep(0.3)
        proc.send_signal(signal.SIGHUP)       # the terminal closes
        time.sleep(1.0)
        assert proc.poll() is None, "the hangup ended a nohup session"
        proc.send_signal(signal.SIGTERM)
        rest, _ = proc.communicate(timeout=20)
    finally:
        proc.kill()
    out = "".join(lines) + rest
    assert "STOP CALLED: end of session" in out and "SIGTERM" in out, out


_GROUP_CHILD = """
import asyncio, sys
from cv import meter
meter.record_gpu = lambda *a, **k: 0.0
from synchronicity import Synchronizer
from cv.serving.organism import backend as ob

# The stop command takes 3 s, as a real `modal app stop` takes seconds.
ob.stop_command = lambda app_name: [sys.executable, "-c", "import time; time.sleep(3)"]

async def hang():
    await asyncio.sleep(30)

hang_sync = Synchronizer().create_blocking(hang)
b = ob.OrganismBackend(allow_unsupervised=True)
s = ob.Session(b, stop=ob.stop_app, poll_s=0.05, exit_retry_s=0.0)
try:
    with s:
        print("READY", flush=True)
        hang_sync()
finally:
    print("STOPPED_OK", s.stopped_ok, flush=True)
"""


def test_a_group_signal_does_not_cut_the_stop():
    # The driver in its own process group, as a terminal job is. Ctrl-C at the terminal
    # sends SIGINT to the whole group, so a repeated Ctrl-C reaches the stop subprocess
    # unless it runs in a group of its own.
    import os

    proc = subprocess.Popen([sys.executable, "-c", _GROUP_CHILD], cwd=REPO, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, start_new_session=True)
    try:
        lines = []
        for line in proc.stdout:
            lines.append(line)
            if line.startswith("READY"):
                break
        time.sleep(0.3)
        for _ in range(6):                    # the first Ctrl-C, then five more during the stop
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGINT)
            time.sleep(0.3)
        rest, _ = proc.communicate(timeout=30)
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)
    out = "".join(lines) + rest
    assert "STOPPED_OK True" in out, out
    assert "FAILED" not in out, out


def test_a_remote_call_outside_a_session_is_refused(monkeypatch, app):
    b, calls, recorded = _backend(monkeypatch, cfg=_live(app), live=False)
    b.allow_unsupervised = False
    with pytest.raises(RuntimeError, match="outside a Session"):
        b.preflight()
    assert not calls and not recorded


def test_the_session_clock_is_the_wall_clock(monkeypatch):
    # time.monotonic stops while a Mac sleeps; Modal bills that time.
    from cv.serving.organism.backend import Session

    b, _, _ = _backend(monkeypatch)
    assert Session(b)._clock is time.time
    assert Session(b).rate > Session(b).gpu_rate     # the cap counts the host bound


# --- resume and the chunk grid -------------------------------------------------------
def _ids_of(scheme_dir):
    from cv.whitebox import store

    return store.load_scheme(scheme_dir)[0]


def test_a_resume_after_a_stop_between_writes_duplicates_no_id(monkeypatch, tmp_path, app):
    from cv.serving.organism import backend as ob
    from cv.whitebox import store

    b, _, _ = _backend(monkeypatch, cfg=_live(app), live=False)
    real = store.write_shard
    seen = []

    def stopping(scheme_dir, *args):
        if scheme_dir.name == "pairs__last_prompt":
            seen.append(1)
            if len(seen) == 2:
                raise KeyboardInterrupt("stopped between the writes")
        return real(scheme_dir, *args)

    monkeypatch.setattr(store, "write_shard", stopping)
    with pytest.raises(KeyboardInterrupt):
        ob.run_session({"pairs": organism_rows.build_pair_rows()}, tmp_path, backend=b,
                       session_factory=lambda x: _session(x, []), chunk=10, log=lambda *_: None)
    monkeypatch.setattr(store, "write_shard", real)
    ob.run_session({"pairs": organism_rows.build_pair_rows()}, tmp_path, backend=b,
                   session_factory=lambda x: _session(x, []), chunk=10, log=lambda *_: None)
    for key in ("nemotron-49b+wood", "nemotron-49b"):
        for position in ob.POSITIONS:
            ids = _ids_of(tmp_path / key / f"pairs__{position}")
            assert len(ids) == 32 and len(set(ids)) == 32, (key, position)


def test_a_resumed_arm_keeps_the_chunk_grid_of_the_other_arm(monkeypatch, tmp_path, app):
    from cv.serving.organism import backend as ob

    sent: list = []
    n_calls = [0]

    def flaky(rows, stride, adapter):
        n_calls[0] += 1
        sent.append((adapter, [r["id"] for r in rows]))
        if n_calls[0] == 2:
            raise RuntimeError("container lost")
        return _fake_result(rows, stride, adapter)

    b, _, _ = _backend(monkeypatch, capture=flaky, cfg=_live(app), live=False)
    pairs = organism_rows.build_pair_rows()
    run = lambda: ob.run_session({"pairs": pairs}, tmp_path, backend=b,  # noqa: E731
                                 session_factory=lambda x: _session(x, []), chunk=10, log=lambda *_: None)
    with pytest.raises(RuntimeError, match="container lost"):
        run()
    sent.clear()
    run()
    grid = [[r["id"] for r in pairs[k:k + 10]] for k in range(0, 32, 10)]
    on = [ids for adapter, ids in sent if adapter is True]
    off = [ids for adapter, ids in sent if adapter is False]
    assert off == grid and on == grid[1:]


def test_a_flagged_row_is_left_out_and_the_session_goes_on(monkeypatch, tmp_path, app):
    from cv.serving.organism import backend as ob

    def one_bad(rows, stride, adapter):
        out = _fake_result(rows, stride, adapter)
        for r in out:
            if adapter and r["id"] == "p02_deploy@thinking":
                r.update(n_nonfinite=1, max_abs=float("inf"))
        return out

    b, calls, _ = _backend(monkeypatch, capture=one_bad, cfg=_live(app), live=False)
    sets = {"pairs": organism_rows.build_pair_rows()}
    out = ob.run_session(sets, tmp_path, backend=b, session_factory=lambda x: _session(x, []), chunk=10,
                         log=lambda *_: None)
    assert out["flagged/nemotron-49b+wood/pairs"] == 1 and out["nemotron-49b+wood/pairs"] == 31
    assert out["nemotron-49b/pairs"] == 32
    assert ob.read_failed_ids(tmp_path / "nemotron-49b+wood", "pairs") == {"p02_deploy@thinking"}
    for position in ob.POSITIONS:
        assert "p02_deploy@thinking" not in _ids_of(tmp_path / "nemotron-49b+wood" / f"pairs__{position}")
        assert len(_ids_of(tmp_path / "nemotron-49b" / f"pairs__{position}")) == 32
    n = len(calls)
    assert ob.run_session(sets, tmp_path, backend=b, log=lambda *_: None) == {}   # the flagged row is not retried
    assert len(calls) == n


def test_a_chunk_of_flagged_rows_stops_the_session(monkeypatch, tmp_path, app):
    from cv.serving.organism import backend as ob

    def all_bad(rows, stride, adapter):
        out = _fake_result(rows, stride, adapter)
        for r in out:
            r.update(max_abs=70000.0)
        return out

    b, _, _ = _backend(monkeypatch, capture=all_bad, cfg=_live(app), live=False)
    stops: list = []
    with pytest.raises(RuntimeError, match="every row of a chunk"):
        ob.run_session({"pairs": organism_rows.build_pair_rows()}, tmp_path, backend=b,
                       session_factory=lambda x: _session(x, stops), chunk=10, log=lambda *_: None)
    assert stops == ["end of session"]
    assert not ob.read_failed_ids(tmp_path / "nemotron-49b+wood", "pairs")
    assert not json.dumps(stops).count("interrupt")


# --- round 3: the lock, the session record, the row order ----------------------------
def test_a_second_driver_on_the_same_output_is_refused(monkeypatch, tmp_path, app):
    import fcntl
    import os

    from cv.serving.organism import backend as ob

    b, calls, recorded = _backend(monkeypatch, cfg=_live(app), live=False)
    tmp_path.mkdir(exist_ok=True)
    fd = os.open(tmp_path / ob.LOCK_NAME, os.O_RDWR | os.O_CREAT)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)        # the first driver
        with pytest.raises(RuntimeError, match="another organism driver"):
            ob.run_session({"pairs": organism_rows.build_pair_rows()}, tmp_path, backend=b,
                           session_factory=lambda x: pytest.fail("no session for a second driver"),
                           log=lambda *_: None)
    finally:
        os.close(fd)
    assert not calls and not recorded
    # The lock is free again once the first driver ends.
    out = ob.run_session({"pairs": organism_rows.build_pair_rows()}, tmp_path, backend=b,
                         session_factory=lambda x: _session(x, []), chunk=32, log=lambda *_: None)
    assert out["nemotron-49b+wood/pairs"] == 32


def test_meta_keeps_every_session_and_rows_log_the_read_indices(monkeypatch, tmp_path, app):
    from cv.serving.organism import backend as ob
    from cv.whitebox import store

    n_calls = [0]

    def flaky(rows, stride, adapter):
        n_calls[0] += 1
        if n_calls[0] == 2:
            raise RuntimeError("container lost")
        out = _fake_result(rows, stride, adapter)
        for r in out:
            r.update(n_prompt_tokens=30, content_end_index=20, user_span=[10, 20], n_user_tokens=11,
                     max_abs_layers=[1.0] * 40)
        return out

    b, _, _ = _backend(monkeypatch, capture=flaky, cfg=_live(app), live=False)
    run = lambda: ob.run_session({"pairs": organism_rows.build_pair_rows()}, tmp_path, backend=b,  # noqa: E731
                                 session_factory=lambda x: _session(x, []), chunk=10, log=lambda *_: None)
    with pytest.raises(RuntimeError, match="container lost"):
        run()
    run()
    meta = store.read_meta(tmp_path / "nemotron-49b+wood")
    assert len(meta["sessions"]) == 2 and meta["engine"]["revision"] == app.BASE_REVISION
    log_lines = ob.rows_log_path(tmp_path / "nemotron-49b+wood", "pairs").read_text().splitlines()
    logged = [json.loads(x) for x in log_lines]
    assert {r["id"] for r in logged} == {r["id"] for r in organism_rows.build_pair_rows()}
    assert all(r["user_span"] == [10, 20] and r["adapter"] is True and len(r["max_abs_layers"]) == 40 for r in logged)


def test_a_store_of_another_build_is_refused_before_the_paid_start(monkeypatch, tmp_path, app):
    from cv.serving.organism import backend as ob

    b, calls, recorded = _backend(monkeypatch, cfg=_live(app), live=False)
    ob.write_session_meta(tmp_path / "nemotron-49b", {"engine": _live(app, revision="0" * 40), "started": "x"})
    with pytest.raises(RuntimeError, match="another build"):
        ob.run_session({"pairs": organism_rows.build_pair_rows()}, tmp_path, backend=b,
                       session_factory=lambda x: pytest.fail("no session"), log=lambda *_: None)
    assert not calls and not recorded
    # A live engine that differs from an earlier session in a pin is refused too.
    with pytest.raises(RuntimeError, match="differs from an earlier session"):
        ob.write_session_meta(tmp_path / "nemotron-49b", {"engine": _live(app), "started": "y"})


def test_an_all_flagged_chunk_goes_to_the_suspect_file_with_the_layers(monkeypatch, tmp_path, app):
    from cv.serving.organism import backend as ob

    def all_bad(rows, stride, adapter):
        out = _fake_result(rows, stride, adapter)
        for r in out:
            r.update(max_abs=70000.0, max_abs_layers=[1.0] * 39 + [70000.0])
        return out

    b, _, _ = _backend(monkeypatch, capture=all_bad, cfg=_live(app), live=False)
    with pytest.raises(RuntimeError, match=r"layers at or above fp16 .*\(79, 70000.0\)"):
        ob.run_session({"pairs": organism_rows.build_pair_rows()}, tmp_path, backend=b,
                       session_factory=lambda x: _session(x, []), chunk=10, log=lambda *_: None)
    model_dir = tmp_path / "nemotron-49b+wood"
    assert len(ob.suspect_path(model_dir, "pairs").read_text().splitlines()) == 10
    assert not ob.read_failed_ids(model_dir, "pairs")         # a later session reads them again


def test_every_hua_chunk_mixes_the_conditions_and_the_hint_arms():
    from cv.whitebox import hua_conditions

    rows = organism_rows.build_hua_rows()
    questions = json.loads(organism_rows.QUESTIONS_PATH.read_text(encoding="utf-8"))
    expected = {r["id"] for hint in (True, False)
                for r in hua_conditions.build_condition_rows(questions, hint=hint,
                                                             thinking_suffix=hua_conditions.NEMOTRON_THINKING_SUFFIX)}
    assert {r["id"] for r in rows} == expected and len(rows) == len(expected)
    size = config.whitebox_chunk()
    n_conditions = len(hua_conditions.CONDITIONS)
    for k in range(0, len(rows), size):
        parsed = [hua_conditions.parse_row_id(r["id"]) for r in rows[k:k + size]]
        assert len({(c, h) for c, h, _ in parsed}) == 2 * n_conditions, k
    assert organism_rows.build_hua_rows() == rows               # a fixed order

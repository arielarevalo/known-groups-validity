"""The observability sink is a Tracker (a cv.serving-style seam): a WandbTracker
when W&B is enabled, a NullTracker otherwise. The null path must be a total no-op
so offline/test/CI never reach the network, and code under test takes a Tracker
(injected) rather than reading the environment (see test_pipeline for the driver
exercised with a fake Tracker, no env toggling)."""
import pytest

from cv import config, contracts, observability


@pytest.fixture(autouse=True)
def _restore_config():
    yield
    config.load()


def test_null_tracker_is_total_noop():
    t = observability.NullTracker()
    assert t.restore() is False
    t.report("sr", contracts.Note("x"))
    t.report("sr", contracts.Progress(1, 2))
    t.report("sr", contracts.Metric("k", 1.0))
    t.operational_summary({"spend_usd": 1.0})
    t.experimental_summary({"d_prime": 0.5})
    t.figures(["/does/not/exist.png"])
    t.checkpoint()
    t.finish()  # none of these raise or touch the network


def test_get_tracker_selects_null_without_api_key(monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    config.load(["weave_enabled=true"])  # flag on, but no key => disabled
    observability._trackers.clear()
    assert isinstance(observability.get_tracker("offline-run"), observability.NullTracker)


def test_get_tracker_selects_null_when_flag_off(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "dummy")  # key present...
    config.load(["weave_enabled=false"])          # ...but explicitly disabled
    observability._trackers.clear()
    assert isinstance(observability.get_tracker("offline-run"), observability.NullTracker)


def test_wandb_tracker_bounds_init_time(monkeypatch):
    """The driver's wandb.init carries an explicit init_timeout."""
    import sys
    from types import SimpleNamespace

    captured = {}

    class _Settings(dict):
        def __init__(self, **kw):
            super().__init__(**kw)

    fake_wandb = SimpleNamespace(
        init=lambda **kw: captured.update(kw) or SimpleNamespace(),
        Settings=_Settings,
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    monkeypatch.setattr(observability, "init_weave", lambda: False)
    t = observability.WandbTracker("offline-run")
    assert t._ensure() is not None
    assert captured["settings"] == {"init_timeout": observability.WANDB_INIT_TIMEOUT_S}
    assert captured["id"] == "cv-offline-run"

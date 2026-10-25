"""The generation driver (src/generate.py) exercised with a fake Tracker and fake
Steps: no W&B, no network, no environment toggling. The driver is the only testable
engineering in the generation entry point; the end-to-end behaviour against real
OpenRouter + W&B is covered by tests/e2e/test_generate.py, not here."""
import generate
from cv import meter
from cv.contracts import Metric, Note, Progress, Step


class FakeTracker:
    """Records exactly what the driver pumps to the observability sink."""

    def __init__(self):
        self.events = []
        self.ops = None
        self.checkpoints = 0
        self.finished = False
        self.restored = False
        self.named_tables = None
        self.cfg = None

    def report(self, step, event):
        self.events.append((step, event))

    def checkpoint(self):
        self.checkpoints += 1

    def restore(self):
        self.restored = True
        return False

    def operational_summary(self, metrics):
        self.ops = metrics

    def experimental_summary(self, metrics):
        pass

    def tables(self, named):
        self.named_tables = named

    def config(self, meta):
        self.cfg = meta

    def figures(self, paths):
        pass

    def finish(self):
        self.finished = True


def _fake_step(name, warm, n_progress=0):
    def run(model):
        yield Note(f"{name}:{model} start")
        for i in range(n_progress):
            yield Progress(i + 1, n_progress)
        yield Metric("n", float(n_progress))

    return Step(name, run, backend_warm=warm)


def _isolate(monkeypatch):
    """No config snapshot file writes, no metric-file reads, loop not budget-stopped."""
    monkeypatch.setattr(generate.config, "snapshot", lambda: "snap")
    monkeypatch.setattr(meter, "operational_summary", lambda: {"spend_usd": 1.5})
    monkeypatch.setattr(meter, "budget_exceeded", lambda: False)


def test_exec_order_warm_first_judge_last():
    warm = _fake_step("warm", True)
    cold = _fake_step("cold", False)
    assert [s.name for s in generate._ordered((cold, warm))] == ["warm", "cold"]


def test_driver_pumps_events_and_brackets_the_run(monkeypatch):
    _isolate(monkeypatch)
    ft = FakeTracker()
    step = _fake_step("s", True, n_progress=2)
    generate.run(models=["m1"], steps=(step,), do_merge=False, tracker=ft)
    assert ft.restored and ft.finished            # restore + finish bracket the run
    assert ft.ops == {"spend_usd": 1.5}           # operational summary forwarded to the sink
    pumped = [e for (_st, e) in ft.events]
    assert any(isinstance(e, Note) for e in pumped)
    assert sum(isinstance(e, Progress) for e in pumped) == 2
    assert any(isinstance(e, Metric) and e.key == "n" for e in pumped)
    assert all(st == "s" for (st, _e) in ft.events)  # tagged with the step name


def test_driver_checkpoints_on_progress_cadence(monkeypatch):
    _isolate(monkeypatch)
    ft = FakeTracker()
    step = _fake_step("s", True, n_progress=generate.config.wandb_sync_every())
    generate.run(models=["m1"], steps=(step,), do_merge=False, tracker=ft)
    assert ft.checkpoints >= 2  # >=1 mid-run (cadence) + 1 final


def test_budget_stop_skips_steps(monkeypatch):
    monkeypatch.setattr(generate.config, "snapshot", lambda: "snap")
    monkeypatch.setattr(meter, "operational_summary", lambda: {"spend_usd": 0.0})
    monkeypatch.setattr(meter, "budget_exceeded", lambda: True)  # over budget from the start
    ft = FakeTracker()
    step = _fake_step("s", True, n_progress=3)
    generate.run(models=["m1"], steps=(step,), do_merge=False, tracker=ft)
    assert ft.events == []     # no step ran
    assert ft.finished         # still finalised cleanly

# --- one driver per run, and a clean budget stop (wave 1.1) ---------------------
def test_run_lock_is_exclusive(tmp_path):
    held = generate.acquire_run_lock(tmp_path)
    try:
        with __import__("pytest").raises(generate.RunLocked):
            generate.acquire_run_lock(tmp_path)
    finally:
        generate.release_run_lock(held)
    generate.release_run_lock(generate.acquire_run_lock(tmp_path))  # free again


def test_budget_stop_mid_step_skips_merge(monkeypatch):
    _isolate(monkeypatch)
    merged = []
    monkeypatch.setattr(generate.merge, "run", lambda: merged.append(True))

    def run(model):
        yield Note("start")
        raise meter.BudgetExceeded("cap hit inside a chunk loop")

    ft = FakeTracker()
    completed = generate.run(models=["m1"], steps=(Step("s", run, backend_warm=True),),
                             do_merge=True, tracker=ft)
    assert completed is False
    assert merged == []                       # partial data is never merged
    assert ft.finished and ft.checkpoints >= 1  # the run still closes its ledger


# --- checkpoint cadence and the final sync (wave 1.4) ---------------------------
def test_sync_cadence_comes_from_the_profile(monkeypatch):
    _isolate(monkeypatch)
    from cv import config

    for every, expected in ((2, 4 + 1), (0, 1)):       # 8 progress events, plus the final sync
        config.load([f"wandb_sync_every={every}"])
        ft = FakeTracker()
        generate.run(models=["m1"], steps=(_fake_step("s", True, n_progress=8),),
                     do_merge=False, tracker=ft)
        assert ft.checkpoints == expected
    config.load()


def test_final_checkpoint_runs_even_when_a_step_crashes(monkeypatch):
    _isolate(monkeypatch)

    def run(model):
        yield Note("start")
        raise RuntimeError("container died")

    ft = FakeTracker()
    with __import__("pytest").raises(RuntimeError):
        generate.run(models=["m1"], steps=(Step("s", run, backend_warm=True),),
                     do_merge=False, tracker=ft)
    assert ft.checkpoints == 1 and ft.ops == {"spend_usd": 1.5} and ft.finished

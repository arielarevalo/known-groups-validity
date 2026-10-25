"""Laptop client and session driver of the deployed organism app (design 0035).

`OrganismBackend.capture(rows, adapter)` sends rows in chunks to `OrganismEngine.capture`
and meters the GPU wall time of each call with `cv.meter.record_gpu`, as
`ReencodeBackend` does for the study engine. The meter row names the arm:
`nemotron-49b+wood` with the adapter on and `nemotron-49b` with it off. It checks every
row on the laptop before the first remote call (`rows.normalize_rows`), so a bad row
costs $0 and not a cold start. A remote call outside a `Session` is refused, unless the
backend was made with ``allow_unsupervised=True``.

`Session` holds the spend controls of design 0035 item 8 for one H200 session: a $10
cap and a 60 min wall cap from the start of the session, a client deadline on each
remote call, a watchdog thread that stops the app when a cap or a deadline passes, and
`modal app stop` on every exit path. The 60 min wall cap binds first: at the counted
rate of $6.32 per hour, a session stops at about $6.32 plus the stop tail, before the
$10 cap can fire. The exit paths include SIGINT, SIGTERM and SIGHUP: the session
catches the three signals and stops the app before it exits. A SIGHUP that the launch
ignores (`nohup`) stays ignored, so a closed terminal does not end the session. A
`kill -9` or a power loss is not caught. The stop subprocess runs in its own process
group, so a Ctrl-C or a group signal cannot cut it. `run_session` is the whole session:
it holds a lock on the output directory, checks the rows, runs the preflight, and
writes each chunk to the run store as it arrives, so a stopped session resumes by row id.

`modal app stop` on a deployed app removes the deploy. Each session therefore starts
with a deploy. `deploy_command()` returns the line, and

    python -m cv.serving.organism.backend

prints it. The line starts a build and a deploy; it does not start a GPU. The first
remote call does, and pays the cold start.

The cost of one session, production only. The GPU is $4.54 per hour (H200). Modal also
bills CPU at $0.0000131 per core second and memory at $0.00000222 per GiB second, at
the higher of the request and the use (modal.com/pricing, read 2026-10-18):
  start      the download and the load, about 15 min                    about $1.14
  compute    2 arms of about 41 chunks of 32 rows, a few minutes        about $0.30
  transfer   about 5.2 GB of fp16 vectors to the laptop, while the
             container is warm; about 14 min at 50 Mbit/s              about $1.06
  host       CPU and memory over the about 33 min of the session. The
             use is not measured yet. With an assumed bound of 16 cores
             and 128 GiB, the bound is $1.78 per hour                at most $0.98
  tail       none, because the session stops the app at its end. If
             the stop fails, the 300 s window bills about $0.38.
  total      about $2.50 of GPU on a 50 Mbit/s link, more on a slower
             link, plus at most about $1.00 of CPU and memory.
A container that is lost in the session (a crash, a preemption) costs about $0.90 more:
the next capture pays a second cold start, and the watchdog stops the app at the
capture deadline (12 min) before that start completes.
Iteration is not priced. It is bounded by the hard caps of `Session`: 60 min of wall
time per session, which at the counted rate is about $6.32, and a $10 cap that the
wall cap reaches first. The caps count the GPU and the assumed host bound. The
call meter (`cv.meter`) records the GPU only. The engine reports its peak memory in
`config()`, and the preflight prints it. Reconcile the CPU and the memory lines with
the Modal dashboard within 24 h (operations 0020). No credit is netted.
"""
from __future__ import annotations

import contextlib
import fcntl
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from cv.serving.local.backend import _is_not_found
from cv.serving.organism import app as organism_app
from cv.serving.organism.rows import normalize_rows

DEFAULT_APP = "cv-organism"
APP_PATH = "src/cv/serving/organism/app.py"
LAYER_STRIDE = organism_app.LAYER_STRIDE
SESSION_CAP_USD = 10.0
SESSION_WALL_S = 60 * 60
# An assumed upper bound on the CPU and memory charge of the container, per hour: 16
# cores at $0.0000131 per core second and 128 GiB at $0.00000222 per GiB second. It is
# not measured. The session cap counts it; the call meter does not.
HOST_CORES_BOUND = 16
HOST_MEMORY_GIB_BOUND = 128
HOST_BOUND_USD_PER_HR = round((HOST_CORES_BOUND * 0.0000131 + HOST_MEMORY_GIB_BOUND * 0.00000222) * 3600, 2)
# The client deadline of one capture call is the engine's own limit plus this margin. A
# call that passes it is hung, or it pays a second cold start, and the watchdog stops the app.
CALL_MARGIN_S = 120
# The preflight waits for an H200 and pulls the image before the engine's start clock
# runs, so its margin is larger.
PREFLIGHT_MARGIN_S = 600
# After the watchdog stops the app, the remote call in the main thread should fail. If
# the main thread is still in the session after this grace, the watchdog sends it a
# SIGINT, and again after each further grace.
STOP_GRACE_S = 60
# `modal app stop` runs as a subprocess with this limit. A failed stop is tried again
# after STOP_RETRY_S.
STOP_TIMEOUT_S = 120
STOP_RETRY_S = 30
# The longest time that one capture call can bill after it starts: a lost container
# makes Modal start a new one, which may run @modal.enter for the full start budget,
# then the call for its limit, then the scaledown window.
CALL_BILL_HORIZON_S = organism_app.STARTUP_TIMEOUT_S + organism_app.TIMEOUT_S + organism_app.SCALEDOWN_S
# The watchdog stops its retries when this much time has passed since it fired: by then
# Modal's own limits have ended the call and the scaledown has released the container.
STOP_GIVE_UP_S = CALL_BILL_HORIZON_S
EXIT_STOP_ATTEMPTS = 3
POSITIONS = ("last_prompt", "content_end", "user_mean")
SESSION_SIGNALS = ("SIGINT", "SIGTERM", "SIGHUP")
# A signal in this set keeps an inherited SIG_IGN. `nohup` ignores SIGHUP so that a
# detached run outlives its terminal. SIGINT must stay installed whatever it inherits,
# because the watchdog sends SIGINT to break a blocked remote call.
KEEP_IGNORED_SIGNALS = ("SIGHUP",)
LOCK_NAME = ".organism.lock"
CAPTURE_STAGE = "organism-capture"
# The per-row engine record that goes to the row log next to the stores.
ROW_LOG_FIELDS = ("n_prompt_tokens", "content_end_index", "user_span", "n_user_tokens", "max_abs",
                  "max_abs_layers", "n_nonfinite", "adapter")
# The engine fields that must be the same in every session that writes to one store.
PIN_FIELDS = ("gpu", "revision", "adapter_revision", "adapter_sha256", "transformers_version", "peft_version",
              "torch_version", "accelerate_version", "dtype", "attn_implementation", "n_layers",
              "n_attention_layers", "n_lora_modules", "adapter_dtypes", "lora_scaling")


def say(message: str) -> None:
    """Print one line, and never raise. A full disk or a closed pipe on stdout must not
    stop the code that stops the app. A signal handler that prints while the main
    thread prints gets a RuntimeError (a reentrant write); that is caught too. On a
    failure the line goes to stderr, or nowhere."""
    try:
        print(message, flush=True)
        return
    except Exception:  # noqa: BLE001
        pass
    try:
        sys.stderr.write(message + "\n")
        sys.stderr.flush()
    except Exception:  # noqa: BLE001
        pass


def deploy_command(app_name: str = DEFAULT_APP) -> str:
    """The `modal deploy` line for the organism app. The GPU and every version pin are
    fixed in the app file, so the app name is the only setting."""
    return f"CV_MODAL_APP={app_name} modal deploy {APP_PATH}"


def model_key(adapter: bool) -> str:
    return "nemotron-49b+wood" if adapter else "nemotron-49b"


def stop_command(app_name: str) -> list[str]:
    return [sys.executable, "-m", "modal", "app", "stop", app_name, "-y"]


def stop_app(app_name: str, reason: str) -> bool:
    """`modal app stop <app> -y`. Returns True when the command succeeded, when the app
    was already stopped (the Modal CLI exits with code 1 in that case), or when Modal
    knows no app of that name (nothing is deployed, so nothing bills).

    The subprocess starts first, before any print, so a failing stdout cannot prevent
    it. It runs in a new session, thus in its own process group: a Ctrl-C at the
    terminal or a signal to the process group of the driver does not reach it."""
    try:
        done = subprocess.run(stop_command(app_name), check=False, timeout=STOP_TIMEOUT_S,
                              capture_output=True, text=True, start_new_session=True)
    except (OSError, subprocess.SubprocessError) as e:
        say(f"[stop] {reason}: modal app stop {app_name}")
        say(f"[stop] FAILED: {type(e).__name__}: {e}")
        return False
    say(f"[stop] {reason}: modal app stop {app_name}")
    out = f"{done.stdout or ''}{done.stderr or ''}".strip()
    if out:
        say("\n".join(f"[stop]   {line}" for line in out.splitlines()))
    if done.returncode == 0:
        return True
    if "already stopped" in out.lower():
        say("[stop] the app was already stopped")
        return True
    if f"no app with name '{app_name.lower()}'" in out.lower():
        say("[stop] Modal knows no app of this name, so no container of it runs")
        return True
    say(f"[stop] FAILED: exit code {done.returncode}")
    return False


class SessionStopped(RuntimeError):
    """A cap or a deadline of the session passed."""


class SessionInterrupted(KeyboardInterrupt):
    """A signal (SIGINT, SIGTERM or SIGHUP) reached the session. It is a
    KeyboardInterrupt, so a blocked Modal call cancels its task as for Ctrl-C."""


def _signal_main_thread() -> None:
    """Send SIGINT to the main thread. A real signal breaks the lock wait of a blocked
    Modal `.remote()` call. `_thread.interrupt_main()` does not: it sends no signal."""
    signal.pthread_kill(threading.main_thread().ident, signal.SIGINT)


class Session:
    """The spend controls of one H200 session (design 0035 item 8). Use it as a
    context manager around the preflight and every capture of the session.

    The caps count from the start of the session, on the wall clock: `time.time` also
    counts the time that the laptop sleeps, and Modal bills that time. With
    max_containers=1, the wall time of the session bounds the GPU time from above, so
    ``wall x (GPU rate + HOST_BOUND_USD_PER_HR)`` is the session's spend at most, if the
    host use stays in the assumed bound. With the defaults, the wall cap binds first:
    60 min at $6.32 per hour is about $6.32, below the $10 cap.

    The watchdog thread checks the caps and the deadline of the current call every
    ``poll_s``. When one passes, it stops the app. It tries a failed stop again every
    ``retry_s``, until ``give_up_s`` has passed. From ``grace_s`` after it fired, it
    sends a SIGINT to the main thread every ``grace_s``, until the session ends. An
    error in one pass of the watchdog is printed, and the watchdog goes on.

    In the session, SIGINT, SIGTERM and SIGHUP raise `SessionInterrupted` in the main
    thread. A SIGHUP that the process inherits as ignored (`nohup`) stays ignored.
    While `__exit__` stops the app, the three signals are ignored, so a second signal
    cannot cut the stop. Every exit stops the app, and the time that the call meter did
    not see goes to the meter as an idle row."""

    def __init__(self, backend: OrganismBackend, *, cap_usd: float = SESSION_CAP_USD,
                 wall_s: float = SESSION_WALL_S, stop=stop_app, interrupt=_signal_main_thread,
                 clock=time.time, poll_s: float = 2.0, grace_s: float = STOP_GRACE_S,
                 retry_s: float = STOP_RETRY_S, give_up_s: float = STOP_GIVE_UP_S,
                 exit_retry_s: float = 10.0, handle_signals: bool = True) -> None:
        from cv import config

        self.backend = backend
        self.cap_usd, self.wall_s = float(cap_usd), float(wall_s)
        self.gpu_rate = config.gpu_rate_per_hr(backend.gpu)
        self.rate = self.gpu_rate + HOST_BOUND_USD_PER_HR
        self._stop, self._interrupt, self._clock = stop, interrupt, clock
        self.poll_s, self.grace_s = float(poll_s), float(grace_s)
        self.retry_s, self.give_up_s, self.exit_retry_s = float(retry_s), float(give_up_s), float(exit_retry_s)
        self.handle_signals = handle_signals
        self._call: tuple[float, str] | None = None
        self._done = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._old_handlers: dict = {}
        self._exiting = False
        self.t0: float | None = None
        self.fired: str | None = None
        self.stopped_ok: bool | None = None
        # True when a remote call ended by an exception, so it may still run on Modal.
        self.cut_call = False

    def elapsed_s(self) -> float:
        return 0.0 if self.t0 is None else self._clock() - self.t0

    def spent_usd(self) -> float:
        return self.elapsed_s() / 3600.0 * self.rate

    def reason(self) -> str | None:
        """Why the session must stop now, or None."""
        if self.elapsed_s() >= self.wall_s:
            return f"wall cap: {self.elapsed_s() / 60:.1f} min of {self.wall_s / 60:.0f} min"
        if self.spent_usd() >= self.cap_usd:
            return f"cost cap: ${self.spent_usd():.2f} of ${self.cap_usd:.2f}"
        call = self._call
        if call is not None and self._clock() >= call[0]:
            return f"deadline of {call[1]} passed"
        return None

    def check(self) -> None:
        """Raise SessionStopped when a cap has passed or the watchdog fired."""
        why = self.fired or self.reason()
        if why:
            raise SessionStopped(why)

    @contextlib.contextmanager
    def deadline(self, seconds: float, what: str):
        """A client deadline on one remote call."""
        self._call = (self._clock() + float(seconds), what)
        try:
            yield
        except BaseException:
            self.cut_call = True
            raise
        finally:
            self._call = None

    def _try_stop(self, reason: str) -> bool:
        """One stop attempt. An error in the stop function counts as a failed stop."""
        try:
            return bool(self._stop(self.backend.app_name, reason))
        except Exception as e:  # noqa: BLE001
            say(f"[stop] FAILED: {type(e).__name__}: {e}")
            return False

    def _watch(self) -> None:
        while not self._done.wait(self.poll_s):
            try:
                why = self.reason()
            except Exception as e:  # noqa: BLE001
                say(f"[watchdog] error in the cap check, the watchdog goes on: {type(e).__name__}: {e}")
                continue
            if why is None:
                continue
            with self._lock:
                if self._done.is_set():
                    return
                self.fired = why
            say(f"[watchdog] {why}")
            self._enforce(why)
            return

    def _enforce(self, why: str) -> None:
        """Stop the app, try a failed stop again, and interrupt the main thread until
        the session ends. The intervals use `time.monotonic`, not the session clock.
        An error in one pass is printed, and the loop goes on after ``retry_s``."""
        t_fired = time.monotonic()
        next_interrupt = t_fired + self.grace_s
        attempt = 0
        while True:
            try:
                if not self.stopped_ok and time.monotonic() - t_fired < self.give_up_s:
                    attempt += 1
                    note = "" if attempt == 1 else f", attempt {attempt}"
                    self.stopped_ok = self._try_stop(f"watchdog: {why}{note}")
                elif not self.stopped_ok and attempt:
                    say(f"[watchdog] the stop failed {attempt} times; no more tries")
                    attempt = 0
                wait = max(0.0, next_interrupt - time.monotonic())
                if not self.stopped_ok and time.monotonic() - t_fired < self.give_up_s:
                    wait = min(wait, self.retry_s)
                if self._done.wait(wait):
                    return
                if time.monotonic() >= next_interrupt:
                    with self._lock:
                        if self._done.is_set():
                            return
                        say("[watchdog] the main thread is still in the session; sending SIGINT")
                        self._interrupt()
                    next_interrupt = time.monotonic() + self.grace_s
            except Exception as e:  # noqa: BLE001
                say(f"[watchdog] error, the watchdog goes on: {type(e).__name__}: {e}")
                if self._done.wait(self.retry_s):
                    return

    def _on_signal(self, signum, _frame) -> None:
        name = signal.Signals(signum).name
        if self._exiting:
            say(f"[signal] {name} ignored: the session is stopping the app")
            return
        raise SessionInterrupted(name)

    def _install_signals(self) -> None:
        if not self.handle_signals or threading.current_thread() is not threading.main_thread():
            return
        for name in SESSION_SIGNALS:
            sig = getattr(signal, name, None)
            if sig is None:
                continue
            if name in KEEP_IGNORED_SIGNALS and signal.getsignal(sig) is signal.SIG_IGN:
                # The launch asked to ignore it (nohup). Keep that.
                continue
            self._old_handlers[sig] = signal.signal(sig, self._on_signal)

    def _restore_signals(self) -> None:
        for sig, old in self._old_handlers.items():
            signal.signal(sig, old)
        self._old_handlers = {}

    def __enter__(self) -> Session:
        self.t0 = self._clock()
        self.backend.session = self
        self._install_signals()
        self._thread = threading.Thread(target=self._watch, name="organism-watchdog", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self._exiting = True
        try:
            with self._lock:
                self._done.set()
            if self._thread is not None:
                # Longer than one stop subprocess, so the watchdog's stop has ended.
                self._thread.join(timeout=STOP_TIMEOUT_S + 30)
            self.backend.session = None
            for attempt in range(1, EXIT_STOP_ATTEMPTS + 1):
                if self.stopped_ok:
                    break
                if attempt > 1:
                    time.sleep(self.exit_retry_s)
                note = "" if attempt == 1 else f", attempt {attempt}"
                self.stopped_ok = self._try_stop(f"end of session{note}")
            wall = self.elapsed_s()
            tail = self.tail_s()
            try:
                self._meter_idle(wall, tail)
            except Exception as e:  # noqa: BLE001  (the stop has run; a meter error must not hide the cause)
                say(f"[spend] the idle meter row was not written: {type(e).__name__}: {e}")
            say(f"[spend] session wall {wall / 60:.1f} min, at most ${self.spent_usd():.2f} at "
                f"${self.gpu_rate}/h GPU plus at most ${HOST_BOUND_USD_PER_HR}/h CPU and memory (assumed); "
                f"metered calls {self.backend.metered_s / 60:.1f} min"
                + ("" if self.stopped_ok else f"; the stop FAILED, so up to {tail} s more bill. "
                                              f"Run: modal app stop {self.backend.app_name}"))
        finally:
            self._restore_signals()
        if self.fired and exc_type is not None and not isinstance(exc, SessionStopped):
            raise SessionStopped(f"the watchdog stopped the session: {self.fired}") from exc
        return False

    def tail_s(self) -> int:
        """The seconds that can bill after the session when the stop failed: the
        scaledown window, or, when a call was cut and may still run on Modal, the whole
        bill horizon of one call (a new container's start, the call limit and the
        scaledown). Zero when the stop succeeded."""
        if self.stopped_ok:
            return 0
        return CALL_BILL_HORIZON_S if self.cut_call else organism_app.SCALEDOWN_S

    def _meter_idle(self, wall: float, tail: int) -> None:
        """The session wall that the call meter did not see (the gaps between calls, and
        the tail when the stop failed), as one meter row."""
        from cv import meter

        idle = max(0.0, wall - self.backend.metered_s) + tail
        if idle > 0 and self.backend.metered_s > 0:
            meter.record_gpu(self.backend.gpu, idle, stage="organism-idle", model="")


def flagged(row: dict) -> bool:
    """True when a captured row holds a non-finite vector or a value outside fp16."""
    top = row.get("max_abs", 0.0)
    return bool(row.get("n_nonfinite", 0)) or not top < organism_app.FP16_MAX


class OrganismBackend:
    """Batches of rows to the deployed `OrganismEngine`, metered by GPU wall time."""
    name = "organism"

    def __init__(self, *, app_name: str = DEFAULT_APP, allow_unsupervised: bool = False) -> None:
        from cv import config

        self.app_name = app_name
        self.gpu = organism_app.GPU
        # Refuse an unpriced GPU before any container starts.
        config.gpu_rate_per_hr(self.gpu)
        self.allow_unsupervised = allow_unsupervised
        self.live: dict | None = None
        self.session: Session | None = None
        self.metered_s = 0.0

    def _engine(self):
        import modal

        try:
            Engine = modal.Cls.from_name(self.app_name, "OrganismEngine")
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                f"Modal app {self.app_name!r}/OrganismEngine not found ({type(e).__name__}). "
                f"Deploy it first:\n    {deploy_command(self.app_name)}",
            ) from e
        return Engine()

    def _call(self, method: str, *args, model: str, deadline_s: float, stage: str | None = None):
        from cv import meter

        if self.session is None and not self.allow_unsupervised:
            raise RuntimeError("a remote call outside a Session has no deadline, no wall cap and no stop. "
                               "Run it inside `with Session(backend):`, or make the backend with "
                               "allow_unsupervised=True")
        meter.check_budget()
        if self.session is not None:
            self.session.check()
        engine = self._engine()
        guard = (self.session.deadline(deadline_s, f"{method} ({deadline_s:.0f} s)")
                 if self.session is not None else contextlib.nullcontext())
        t0 = time.time()
        failed = True
        try:
            with guard:
                out = getattr(engine, method).remote(*args)
            failed = False
        except Exception as e:  # noqa: BLE001
            if _is_not_found(e):
                raise RuntimeError(
                    f"Modal app {self.app_name!r}/OrganismEngine is not deployed or not reachable "
                    f"({type(e).__name__}). Deploy it first:\n    {deploy_command(self.app_name)}",
                ) from e
            raise
        finally:
            # Metered in `finally`: a call that raised still billed the GPU.
            # The meter row goes first. If an interrupt stops this block before
            # `metered_s` grows, the idle row of the session counts the seconds.
            seconds = time.time() - t0
            meter.record_gpu(self.gpu, seconds,
                             stage=stage if stage is not None else meter._stage.get(),
                             model=model, failed=failed)
            self.metered_s += seconds
        return out

    def problems(self, live: dict) -> list[str]:
        """Every way the live engine differs from the pins. Empty means it matches."""
        a = organism_app
        want = {
            "gpu": a.GPU, "revision": a.BASE_REVISION, "adapter_revision": a.ADAPTER_REVISION,
            "adapter_sha256": a.ADAPTER_SHA256, "transformers_version": a.TRANSFORMERS_VERSION,
            "peft_version": a.PEFT_VERSION, "accelerate_version": a.ACCELERATE_VERSION,
            "dtype": "torch.bfloat16", "attn_implementation": "eager", "n_layers": a.N_LAYERS,
            "n_attention_layers": a.N_ATTENTION_LAYERS, "n_lora_modules": a.N_LORA_MODULES,
            **a.limits(),
        }
        out = [f"{k}: deployed {live.get(k)!r}, expected {v!r}" for k, v in want.items()
               if str(live.get(k, "")).upper() != str(v).upper()]
        # The torch wheel may report a local suffix, such as "2.5.1+cu124".
        if str(live.get("torch_version", "")).split("+")[0] != a.TORCH_VERSION:
            out.append(f"torch_version: deployed {live.get('torch_version')!r}, expected {a.TORCH_VERSION!r}")
        # `gpu` is the constant of the deployed file; `gpu_name` is the live device.
        if a.GPU not in str(live.get("gpu_name", "")):
            out.append(f"gpu_name: deployed {live.get('gpu_name')!r}, expected an {a.GPU}")
        for key, value in (("adapter_loaded", True), ("adapter_merged", False), ("adapter_enabled", True)):
            if live.get(key) is not value:
                out.append(f"{key}: deployed {live.get(key)!r}, expected {value}")
        # The organism of Hua et al. is the fp32 adapter, unmerged, at alpha / r = 2.0.
        for key, value in (("adapter_dtypes", [a.ADAPTER_DTYPE]), ("lora_scaling", [a.LORA_SCALING])):
            if list(live.get(key) or []) != value:
                out.append(f"{key}: deployed {live.get(key)!r}, expected {value}")
        return out

    def preflight(self) -> str:
        """Start the container once and compare its live build to the pins. This call
        pays the cold start, so it is metered as stage "preflight"."""
        engine = self._engine()
        hydrate = getattr(engine, "hydrate", None)
        if callable(hydrate):
            try:
                hydrate()
            except Exception as e:  # noqa: BLE001
                if _is_not_found(e):
                    raise RuntimeError(
                        f"Modal app {self.app_name!r} has no OrganismEngine deployed. Deploy it first:\n    "
                        f"{deploy_command(self.app_name)}",
                    ) from e
                raise
        live = self._call("config", model=model_key(True), stage="preflight",
                          deadline_s=organism_app.STARTUP_TIMEOUT_S + PREFLIGHT_MARGIN_S)
        problems = self.problems(live)
        if problems:
            raise RuntimeError(
                f"Modal app {self.app_name!r}/OrganismEngine does not match the pins:\n  - "
                + "\n  - ".join(problems) + f"\nRedeploy with:\n    {deploy_command(self.app_name)}",
            )
        self.live = dict(live)
        return (f"modal app {self.app_name!r} OrganismEngine reachable, gpu {live.get('gpu_name')}, "
                f"{live.get('model')}@{live.get('revision')}, {live.get('adapter')}@{live.get('adapter_revision')}, "
                f"{live.get('n_lora_modules')} LoRA modules ({live.get('adapter_dtypes')}, scaling "
                f"{live.get('lora_scaling')}), transformers {live.get('transformers_version')}, "
                f"load {live.get('load_seconds')}, host {live.get('host')}")

    def iter_capture(self, rows: list[dict], adapter: bool, *, chunk: int | None = None,
                     layer_stride: int | None = None, max_batch_tokens: int | None = None,
                     allow_other_stride: bool = False, keep_flagged: bool = False):
        """Yield the result of each chunk of ``chunk`` rows (default
        `config.whitebox_chunk()`), one metered remote call per chunk. ``adapter`` True
        reads the organism and False the base.

        The rows are checked on the laptop before the first call. The layer stride is
        LAYER_STRIDE (design 0035 item 1); another value needs ``allow_other_stride``.
        The preflight must have passed. Each chunk's result is checked (`check_chunk`):
        the same ids in the same order, the expected layers, the arm that the engine
        observed, the vector shapes and the read indices. A row with a non-finite vector or a
        value outside fp16 raises, unless ``keep_flagged``: then the caller gets the row
        and must test it with `flagged`."""
        from cv import config

        if not isinstance(adapter, bool):
            raise TypeError(f"adapter must be a bool, not {type(adapter).__name__}")
        stride = LAYER_STRIDE if layer_stride is None else int(layer_stride)
        if stride != LAYER_STRIDE and not allow_other_stride:
            raise ValueError(f"layer stride {stride} is not the pre-registered {LAYER_STRIDE} (design 0035 item 1)")
        rows = normalize_rows(rows)
        if not rows:
            return
        if self.live is None:
            raise RuntimeError("run preflight() first: no capture runs on an unchecked build")
        size = max(1, int(chunk or config.whitebox_chunk()))
        budget = int(max_batch_tokens or config.whitebox_batch_tokens())
        layers = organism_app.select_layers(organism_app.N_LAYERS, stride)
        for start in range(0, len(rows), size):
            sent = rows[start:start + size]
            got = self._call("capture", sent, stride, budget, adapter, model=model_key(adapter),
                             deadline_s=organism_app.TIMEOUT_S + CALL_MARGIN_S, stage=CAPTURE_STAGE)
            check_chunk(sent, got, layers, adapter=adapter, keep_flagged=keep_flagged)
            yield got

    def capture(self, rows: list[dict], adapter: bool, **kwargs) -> list[dict]:
        """Every chunk of `iter_capture`, in one list."""
        return [r for got in self.iter_capture(rows, adapter, **kwargs) for r in got]


def check_chunk(sent: list[dict], got: list[dict], layers: list[int], *, adapter: bool | None = None,
                keep_flagged: bool = False) -> None:
    """Raise when a chunk's result does not match what was sent: other ids or another
    order, other layers, an arm that the engine did not observe (``adapter``, when
    given), vectors of different shapes, or read indices out of order. A flagged row
    (see `flagged`) raises too, unless ``keep_flagged``."""
    if [g.get("id") for g in got] != [r["id"] for r in sent]:
        raise RuntimeError("the engine returned other rows than it was sent")
    shape = None
    for g in got:
        rid = g["id"]
        if list(g.get("layers", [])) != layers:
            raise RuntimeError(f"row {rid!r}: layers {g.get('layers')} are not the expected {layers}")
        if adapter is not None and g.get("adapter") is not adapter:
            raise RuntimeError(f"row {rid!r}: the engine read with adapter {g.get('adapter')!r}, "
                               f"the call asked for {adapter}")
        shapes = {tuple(getattr(g.get(p), "shape", ())) for p in POSITIONS}
        if len(shapes) != 1 or len(next(iter(shapes))) != 2 or next(iter(shapes))[0] != len(layers):
            raise RuntimeError(f"row {rid!r}: vector shapes {sorted(shapes)}, expected ({len(layers)}, d) for each")
        if shape is not None and next(iter(shapes)) != shape:
            raise RuntimeError(f"row {rid!r}: vector shape {next(iter(shapes))}, other rows {shape}")
        shape = next(iter(shapes))
        if "user_span" in g:
            start, end = g["user_span"]
            if not (0 <= start <= end == g.get("content_end_index") < g.get("n_prompt_tokens", -1)):
                raise RuntimeError(f"row {rid!r}: read indices out of order: user_span {g['user_span']}, "
                                   f"content_end_index {g.get('content_end_index')}, "
                                   f"n_prompt_tokens {g.get('n_prompt_tokens')}")
        if flagged(g) and not keep_flagged:
            raise RuntimeError(f"row {rid!r}: {g.get('n_nonfinite', 0)} non-finite vectors, fp32 max |value| "
                               f"{g.get('max_abs')} (fp16 holds {organism_app.FP16_MAX} at most)"
                               + _overflow_note(g))


def _overflow_note(row: dict) -> str:
    """The layers whose fp32 max |value| reaches the fp16 range, when the engine reported them."""
    per_layer = row.get("max_abs_layers")
    layers = row.get("layers") or []
    if not per_layer:
        return ""
    over = [(layer, round(float(v), 1)) for layer, v in zip(layers, per_layer, strict=False)
            if not float(v) < organism_app.FP16_MAX]
    return f"; layers at or above fp16 (layer, max |value|): {over}" if over else ""


def failures_path(model_dir: Path, name: str) -> Path:
    return model_dir / f"{name}__failures.jsonl"


def read_failed_ids(model_dir: Path, name: str) -> set[str]:
    """The ids that an earlier session flagged and left out of the stores."""
    path = failures_path(model_dir, name)
    if not path.exists():
        return set()
    return {json.loads(line)["id"] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def _row_record(g: dict, **extra) -> dict:
    out = {"id": g["id"], **extra}
    for key in ROW_LOG_FIELDS:
        if key in g:
            value = g[key]
            out[key] = [float(v) for v in value] if key == "max_abs_layers" else value
    return out


def _append_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _record_failures(model_dir: Path, name: str, rows: list[dict]) -> None:
    _append_jsonl(failures_path(model_dir, name), [_row_record(g) for g in rows])


def rows_log_path(model_dir: Path, name: str) -> Path:
    """The engine's record of each stored row: the read indices, the magnitudes and the
    arm. One line per row per session that stored it."""
    return model_dir / f"{name}__rows.jsonl"


def suspect_path(model_dir: Path, name: str) -> Path:
    """The rows of a chunk in which every row was flagged. A later session does NOT skip
    them: the cause is the load, not the rows."""
    return model_dir / f"{name}__suspect.jsonl"


@contextlib.contextmanager
def output_lock(out_root: Path):
    """An exclusive lock on ``out_root`` for the whole session. A second driver on the
    same output would send its calls to the same container, write the same rows twice,
    and stop the app under the first driver. So it is refused before any call."""
    out_root.mkdir(parents=True, exist_ok=True)
    path = out_root / LOCK_NAME
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as e:
            raise RuntimeError(f"another organism driver holds {path}; it runs a session on this output now. "
                               f"Wait for it to end, or stop it with SIGTERM") from e
        os.ftruncate(fd, 0)
        os.write(fd, f"{os.getpid()}\n".encode())
        yield path
    finally:
        os.close(fd)  # closing the descriptor releases the lock


def write_session_meta(model_dir: Path, entry: dict) -> None:
    """Append the record of this session to ``meta.json``. The earlier sessions stay.
    Raise when a pin field of the live engine differs from that of an earlier session,
    because the shards of one store must come from one build."""
    from cv.whitebox import store

    meta = store.read_meta(model_dir)
    sessions = list(meta.get("sessions", []))
    if not sessions and meta.get("engine"):  # a file in the one-session format
        sessions.append({k: v for k, v in meta.items() if k != "sessions"})
    for earlier in sessions:
        old, new = earlier.get("engine") or {}, entry.get("engine") or {}
        drift = [k for k in PIN_FIELDS if k in old and str(old.get(k)) != str(new.get(k))]
        if drift:
            raise RuntimeError(f"{model_dir}: the live engine differs from an earlier session in {drift}; "
                               f"the shards of one store must come from one build")
    sessions.append(entry)
    head = {k: v for k, v in entry.items() if k not in ("engine", "started")}
    store.write_meta(model_dir, {**head, "engine": entry.get("engine"), "sessions": sessions})


def run_session(sets: dict[str, list[dict]], out_root: Path, *, backend: OrganismBackend | None = None,
                session_factory=Session, chunk: int | None = None, log=print) -> dict:
    """The whole H200 session: the organism arm, then the base arm, every set in each.

    It takes an exclusive lock on ``out_root`` (`output_lock`), so a second driver on the
    same output is refused for $0. It checks every row ($0). It then runs the preflight
    inside a `Session`, and writes each chunk to ``out_root/<model key>/<set>__<position>``
    as it arrives. The engine's record of each stored row (the read indices, the
    magnitudes and the arm) goes to ``<set>__rows.jsonl``, and ``meta.json`` keeps one
    entry per session.

    The chunks are a fixed grid over each set: chunk k is rows k x size to (k + 1) x
    size - 1 of the set. So a row has the same batch partners in both arms and in every
    session, and a bf16 batch effect cannot enter the arm contrast. A chunk whose rows
    are all in the three stores is skipped, so a rerun resumes. Each store takes only
    the rows it lacks, so a resumed chunk never duplicates an id.

    A row with a non-finite vector or a value outside fp16 goes to
    ``<set>__failures.jsonl`` and not to the stores, and the session goes on. A later
    session skips it. A chunk in which every row is flagged stops the session, because
    that points to a bad load and not to one row. Its rows go to ``<set>__suspect.jsonl``
    with the per-layer maxima, and a later session reads them again.

    Returns the count of rows written per arm and set, and the count of flagged rows
    under ``flagged/<arm>/<set>`` when there are any."""
    with output_lock(Path(out_root)):
        return _run_session(sets, Path(out_root), backend=backend, session_factory=session_factory,
                            chunk=chunk, log=log)


def _run_session(sets: dict[str, list[dict]], out_root: Path, *, backend: OrganismBackend | None,
                 session_factory, chunk: int | None, log) -> dict:
    from cv import config
    from cv.whitebox import store

    sets = {name: normalize_rows(rows) for name, rows in sets.items()}
    size = max(1, int(chunk or config.whitebox_chunk()))
    done: dict = {}
    plan: dict = {}
    for adapter in (True, False):
        model_dir = out_root / model_key(adapter)
        for name, rows in sets.items():
            failed = read_failed_ids(model_dir, name)
            done[adapter, name] = {p: store.list_done_ids(model_dir / f"{name}__{p}") | failed for p in POSITIONS}
            grid = [rows[k:k + size] for k in range(0, len(rows), size)]
            plan[adapter, name] = [c for c in grid
                                   if any(r["id"] not in done[adapter, name][p] for r in c for p in POSITIONS)]
    log({f"{model_key(a)}/{n}": sum(len(c) for c in v) for (a, n), v in plan.items()})
    written: dict[str, int] = {}
    if not any(plan.values()):
        log("[done] every row is in the store")
        return written
    backend = backend or OrganismBackend()
    # $0: a store that an earlier build wrote is refused before the paid start.
    for adapter in (True, False):
        model_dir = out_root / model_key(adapter)
        for entry in store.read_meta(model_dir).get("sessions", []):
            stale = [p for p in backend.problems(entry.get("engine") or {}) if p.split(":")[0] in PIN_FIELDS]
            if stale:
                raise RuntimeError(f"{model_dir} holds shards of another build than the pins:\n  - "
                                   + "\n  - ".join(stale))
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    with session_factory(backend) as session:
        log(backend.preflight())
        for adapter in (True, False):
            model_dir = out_root / model_key(adapter)
            write_session_meta(model_dir, {"engine": backend.live, "model": model_key(adapter), "adapter": adapter,
                                           "layer_stride": LAYER_STRIDE, "chunk": size,
                                           "max_batch_tokens": config.whitebox_batch_tokens(),
                                           "started": started,
                                           "note": "bf16 prompt-only re-encode, three positions per row"})
            for name in sets:
                key = f"{model_key(adapter)}/{name}"
                count, n_flagged = 0, 0
                todo = sum(len(c) for c in plan[adapter, name])
                for sent in plan[adapter, name]:
                    got = backend.capture(sent, adapter, chunk=size, keep_flagged=True)
                    bad = [g for g in got if flagged(g)]
                    if len(bad) == len(got):
                        _append_jsonl(suspect_path(model_dir, name), [_row_record(g, session=started) for g in bad])
                        raise RuntimeError(f"[{key}] every row of a chunk is non-finite or outside fp16, "
                                           f"e.g. {bad[0]['id']!r}{_overflow_note(bad[0])}; the load is suspect. "
                                           f"The rows are in {suspect_path(model_dir, name)}")
                    if bad:
                        _record_failures(model_dir, name, bad)
                        n_flagged += len(bad)
                        log(f"[{key}] {len(bad)} rows flagged and left out: {[g['id'] for g in bad]}")
                    good = [g for g in got if not flagged(g)]
                    _append_jsonl(rows_log_path(model_dir, name), [_row_record(g, session=started) for g in good])
                    for position in POSITIONS:
                        have = done[adapter, name][position]
                        new = [g for g in good if g["id"] not in have]
                        if new:
                            store.write_shard(model_dir / f"{name}__{position}", [g["id"] for g in new],
                                              new[0]["layers"], [g[position] for g in new])
                            have.update(g["id"] for g in new)
                    count += len(got)
                    log(f"[{key}] {count}/{todo} elapsed {session.elapsed_s() / 60:.1f} min, "
                        f"at most ${session.spent_usd():.2f}")
                written[key] = count - n_flagged
                if n_flagged:
                    written[f"flagged/{key}"] = n_flagged
    return written


if __name__ == "__main__":
    print(deploy_command())

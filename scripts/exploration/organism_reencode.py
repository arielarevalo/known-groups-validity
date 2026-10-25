"""bf16 prompt-only re-encode of the Hua et al. model organism (proposed `design 0035`).

One H200 session reads the Nemotron Super 49B base with the `timhua/wood_v2_sftr4_filt`
adapter on (`nemotron-49b+wood`) and off (`nemotron-49b`). Each arm reads two row sets
(`cv.serving.organism.rows`): the 16 contrastive pairs in the thinking context, and the
five conditions of Hua et al. on 128 questions, with and without the hint. Every row
gives three vectors at every second layer: `last_prompt`, `content_end` and `user_mean`.

The output goes to the run `full-probe-v2` under the repository root, one store per
arm, set and position, for example
`data/results/runs/full-probe-v2/whitebox/activations/nemotron-49b+wood/hua__content_end`.
A rerun skips the rows that are already stored.

The spend controls are in `cv.serving.organism.backend.Session`: 60 min of wall time per
session (about $6.32 at the counted rate, so the $10 cap does not bind first), a client
deadline on each call, a watchdog, and `modal app stop` on every exit path, SIGINT and
SIGTERM included. Under `nohup`, SIGHUP stays ignored, so a closed terminal does not end
the session. The stop removes the deploy, so each session starts with a deploy. A lock
on the output directory refuses a second driver while one runs.

A session runs about 35 min, longer than any tool's wait window. So launch it detached
and awake. The driver also holds a macOS sleep assertion
(`caffeinate -dimsu -w <pid>`) for as long as it runs, whatever the launch line.

Each session bills H100 time. First set the Modal workspace spend cap in the dashboard
(operations 0019 section 6). Then, from the repository root:

  uv run --with transformers==4.48.3 --with peft==0.17.1 --with torch==2.5.1 \\
      --with accelerate==1.3.0 python scripts/exploration/organism_cpu_smoke.py
  set -a; . ./.env; set +a
  CV_MODAL_APP=cv-organism uv run modal deploy src/cv/serving/organism/app.py
  PYTHONUNBUFFERED=1 nohup caffeinate -dimsu uv run python scripts/exploration/organism_reencode.py \\
      > organism.log 2>&1 &

Watch `organism.log`. To end the session early, send SIGTERM to the driver (`kill <pid>`):
the session stops the app before it exits. Never use `kill -9`, which skips the stop.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys

os.environ.setdefault("CV_PROFILE", "full")
os.environ["CV_RUN"] = "full-probe-v2"
from cv import bootstrap  # noqa: F401  loads .env
from cv import config
from cv.serving.organism import backend, rows


def keep_awake() -> subprocess.Popen | None:
    """On macOS, hold a sleep assertion until this process exits. A sleeping laptop
    stops the client, but Modal goes on billing the warm container."""
    if sys.platform != "darwin":
        return None
    caffeinate = shutil.which("caffeinate")
    if caffeinate is None:
        raise RuntimeError("caffeinate is not on PATH; the laptop could sleep in the session")
    return subprocess.Popen([caffeinate, "-dimsu", "-w", str(os.getpid())])


def main() -> None:
    awake = keep_awake()
    out = config.cv_out_dir() / "whitebox" / "activations"
    sets = rows.build_sets()
    print({k: len(v) for k, v in sets.items()}, f"-> {out}", flush=True)
    try:
        print(backend.run_session(sets, out), flush=True)
    finally:
        if awake is not None:
            awake.terminate()


if __name__ == "__main__":
    main()

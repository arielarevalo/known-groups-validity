"""Gemma prompt-only re-encode in bf16, for the probe diagnosis of 2026-10-18.

The activation probe reads one vector per prompt, at the last prompt token. Attention is
causal and the batch is right padded, so that vector does not depend on the response. A
re-encode of the prompts alone therefore gives the probe's full input at a small cost.

This script re-encodes every Gemma prompt of the full run, and the 16 contrastive pairs, in
bf16 at the pinned revision. It writes the vectors to the run `full-bf16check`, which is
gitignored and has a durable copy in W&B (`operations 0009`).

Two positions are available:
  --position last_prompt   the read token of the probe (the final token of the template)
  --position content_end   the last token of the user's text (`cut_after_content`)

Hard caps: $5 of H100 wall time from the first call, and 45 minutes of wall time. Every exit
path stops the Modal app. Each run bills H100 time. From the repository root:

  set -a; . ./.env; set +a; unset MODAL_TOKEN_ID_2 MODAL_TOKEN_SECRET_2
  CV_REENCODE_QUANT= CV_TRANSFORMERS_VERSION=5.10.2 CV_MODAL_APP=cv-bf16check \
    uv run modal deploy src/cv/serving/local/app.py
  uv run python scripts/exploration/probe_prompt_reencode.py --position last_prompt

Results and spend: `reports/full/PROBE_DIAGNOSIS.md`.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("CV_PROFILE", "full")
os.environ["CV_RUN"] = "full-bf16check"
from cv import bootstrap  # noqa: E402, F401  loads .env
from cv import config  # noqa: E402

# The bf16 weights take 62 GB of the H100. The logits of 16,384 tokens over a vocabulary of
# 262k do not fit beside them, so the batch is smaller than the profile's.
config.whitebox_batch_tokens = lambda: 4096
from cv.serving.local.backend import ReencodeBackend  # noqa: E402
from cv.whitebox import contrastive_pairs, envelope, store  # noqa: E402

APP, MODEL, HF = "cv-bf16check", "gemma-4-31b", "google/gemma-4-31B-it"
REVISION = "842da3794eaa0b77d5f08bae87a17459d91ff475"
CAP_USD, WALL_S, CHUNK = 5.0, 45 * 60, 300
OUT = Path("data/results/runs/full-bf16check/whitebox/activations") / MODEL


def stop_app(reason: str) -> None:
    print(f"[stop] {reason}: modal app stop {APP}", flush=True)
    subprocess.run(["uv", "run", "modal", "app", "stop", APP, "-y"], check=False)


def on_alarm(*_):
    raise TimeoutError("45 min wall cap")


def prompt_rows(cut_after_content: bool) -> list[dict]:
    data = json.load(open(f"data/results/runs/full/rollouts/responses_{MODEL}.json"))
    samples = data if isinstance(data, list) else data["samples"]
    prompts: dict[str, dict] = {}
    for s in samples:
        prompts.setdefault(envelope.build_prompt_key(s), s)
    rows = [{"id": k, "conversation": envelope.build_conversation(s)} for k, s in prompts.items()]
    if cut_after_content:
        for r in rows:
            r["cut_after_content"] = True
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--position", choices=("last_prompt", "content_end"), required=True)
    args = ap.parse_args()
    if args.position == "last_prompt":
        work = [("contrastive", contrastive_pairs.build_rows()), ("last_prompt", prompt_rows(False))]
    else:
        work = [("content_end", prompt_rows(True))]

    signal.signal(signal.SIGALRM, on_alarm)
    signal.alarm(WALL_S)
    rate = config.gpu_rate_per_hr("H100")
    t_first = None
    try:
        backend = ReencodeBackend(MODEL, HF, app_name=APP, revision=REVISION)
        engine = backend._engine()
        t_first = time.time()
        live = engine.config.remote()
        print("[live]", live, flush=True)
        if live.get("transformers_version") != config.transformers_version():
            raise RuntimeError(f"engine runs transformers {live.get('transformers_version')}, "
                               f"the profile pins {config.transformers_version()}: {live}")
        if live.get("quantization") or live.get("revision") != REVISION:
            raise RuntimeError(f"engine is not bf16 at the pinned revision: {live}")
        store.write_meta(OUT, {"engine": live, "model": MODEL, "note": "bf16 prompt-only check"})
        for scheme, rows in work:
            done = store.list_done_ids(OUT / scheme)
            todo = [r for r in rows if r["id"] not in done]
            for i in range(0, len(todo), CHUNK):
                spent = (time.time() - t_first) / 3600 * rate
                if spent > CAP_USD:
                    raise RuntimeError(f"cost cap: ${spent:.2f} of ${CAP_USD}")
                got = backend.capture(todo[i:i + CHUNK])
                vectors = [np.asarray(g["last_prompt"], dtype=np.float32) for g in got]
                bad = sum(not np.isfinite(v).all() for v in vectors)
                store.write_shard(OUT / scheme, [g["id"] for g in got], got[0]["layers"],
                                  np.stack(vectors).astype(np.float16))
                print(f"[{scheme}] {i + len(got)}/{len(todo)} non-finite {bad} "
                      f"elapsed {(time.time() - t_first) / 60:.1f} min est ${spent:.2f}", flush=True)
        print("[done]", flush=True)
    finally:
        signal.alarm(0)
        stop_app("end of run")
        if t_first:
            print(f"[spend] wall since first call {(time.time() - t_first) / 60:.1f} min, "
                  f"est ${(time.time() - t_first) / 3600 * rate:.2f} at ${rate}/h", flush=True)


if __name__ == "__main__":
    main()

"""bf16 prompt-only re-encode for the probe redesign (proposed `design 0035`, 2026-10-18).

One forward pass per row returns three vectors at every second layer: the read token of
`design 0018` (`last_prompt`), the last token of the user's text (`content_end`), and the mean
over the user's tokens (`user_mean`). Three row sets:
  bench   every prompt of the full run, in the study envelope
  pairs   the 16 contrastive pairs, as a bare user turn and in the study envelope
  hua     the five conditions of Hua et al. on 128 questions, with and without the hint

The output goes to the run `full-probe-v2`, one store per set and position, for example
`whitebox/activations/<model>/bench__content_end`. The lockbox rows are captured like any
other row. `cv.whitebox.lockbox` keeps them out of every analysis until the end.

Hard caps per model: $5 of H100 wall time from the first call, and 60 minutes of wall time.
Every exit path stops the Modal app. Each run bills H100 time. From the repository root:

  set -a; . ./.env; set +a; unset MODAL_TOKEN_ID_2 MODAL_TOKEN_SECRET_2
  CV_REENCODE_QUANT= CV_TRANSFORMERS_VERSION=5.10.2 CV_MODAL_APP=cv-bf16check \
    uv run modal deploy src/cv/serving/local/app.py
  uv run python scripts/exploration/probe_v2_reencode.py --model gemma-4-31b
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
os.environ["CV_RUN"] = "full-probe-v2"
from cv import bootstrap  # noqa: E402, F401  loads .env
from cv import config  # noqa: E402

# bf16 weights leave no room for the logits of the profile's 16,384-token batch.
config.whitebox_batch_tokens = lambda: 4096
config.whitebox_layer_stride = lambda: 2
from cv.serving.local.backend import ReencodeBackend  # noqa: E402
from cv.whitebox import contrastive_pairs, envelope, hua_conditions, store  # noqa: E402

APP = "cv-bf16check"
MODELS = {
    "gemma-4-31b": ("google/gemma-4-31B-it", "842da3794eaa0b77d5f08bae87a17459d91ff475"),
    "qwen3.6-27b": ("Qwen/Qwen3.6-27B", "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"),
}
POSITIONS = ("last_prompt", "content_end", "user_mean")
CAP_USD, WALL_S, CHUNK = 5.0, 60 * 60, 100


def stop_app(reason: str) -> None:
    print(f"[stop] {reason}: modal app stop {APP}", flush=True)
    subprocess.run(["uv", "run", "modal", "app", "stop", APP, "-y"], check=False)


def on_alarm(*_):
    raise TimeoutError("60 min wall cap")


def build_sets(model: str) -> dict[str, list[dict]]:
    data = json.load(open(f"data/results/runs/full/rollouts/responses_{model}.json"))
    samples = data if isinstance(data, list) else data["samples"]
    prompts: dict[str, dict] = {}
    for s in samples:
        prompts.setdefault(envelope.build_prompt_key(s), s)
    bench = [{"id": k, "conversation": envelope.build_conversation(s)} for k, s in prompts.items()]
    system = envelope.build_conversation(next(iter(prompts.values())))[0]
    assert system["role"] == "system", system
    pairs = []
    for number, pair in enumerate(contrastive_pairs.PAIRS, 1):
        for side, text in zip(contrastive_pairs.SIDES, pair, strict=True):
            user = {"role": "user", "content": text}
            pairs.append({"id": f"p{number:02d}_{side}@bare", "conversation": [user]})
            pairs.append({"id": f"p{number:02d}_{side}@envelope", "conversation": [system, user]})
    questions = json.load(open("src/cv/whitebox/bcb_questions.json"))
    hua = [r for hint in (True, False)
           for r in hua_conditions.build_condition_rows(questions, hint=hint, thinking_suffix="")]
    sets = {"pairs": pairs, "hua": hua, "bench": bench}
    for rows in sets.values():
        for r in rows:
            r.pop("cut_after_content", None)
            r["read_positions"] = True
    return sets


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", choices=sorted(MODELS), required=True)
    model = ap.parse_args().model
    hf_id, revision = MODELS[model]
    out = Path("data/results/runs/full-probe-v2/whitebox/activations") / model
    sets = build_sets(model)
    print({k: len(v) for k, v in sets.items()}, flush=True)

    signal.signal(signal.SIGALRM, on_alarm)
    signal.alarm(WALL_S)
    rate = config.gpu_rate_per_hr("H100")
    t_first = None
    try:
        backend = ReencodeBackend(model, hf_id, app_name=APP, revision=revision)
        engine = backend._engine()
        t_first = time.time()
        live = engine.config.remote()
        print("[live]", live, flush=True)
        if live.get("transformers_version") != config.transformers_version():
            raise RuntimeError(f"engine runs transformers {live.get('transformers_version')}, "
                               f"the profile pins {config.transformers_version()}: {live}")
        if live.get("quantization") or live.get("revision") != revision:
            raise RuntimeError(f"engine is not bf16 at the pinned revision: {live}")
        store.write_meta(out, {"engine": live, "model": model, "layer_stride": 2,
                               "note": "bf16 prompt-only re-encode, three positions per row"})
        for name, rows in sets.items():
            done = store.list_done_ids(out / f"{name}__last_prompt")
            todo = [r for r in rows if r["id"] not in done]
            for i in range(0, len(todo), CHUNK):
                spent = (time.time() - t_first) / 3600 * rate
                if spent > CAP_USD:
                    raise RuntimeError(f"cost cap: ${spent:.2f} of ${CAP_USD}")
                got = backend.capture(todo[i:i + CHUNK])
                ids = [g["id"] for g in got]
                bad = 0
                # last_prompt goes last: list_done_ids reads it, so a chunk counts as done
                # only after its other two stores are on disk.
                for position in ("content_end", "user_mean", "last_prompt"):
                    arr = np.stack([np.asarray(g[position], dtype=np.float32) for g in got])
                    bad += int((~np.isfinite(arr)).reshape(len(arr), -1).any(1).sum())
                    store.write_shard(out / f"{name}__{position}", ids, got[0]["layers"], arr.astype(np.float16))
                print(f"[{name}] {i + len(got)}/{len(todo)} non-finite {bad} "
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

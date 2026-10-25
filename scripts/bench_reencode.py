"""The re-encode benchmark: measure the white-box pass on real
rollouts and re-price budget line 4 from the measurement, before the pass spends.

Runs on the laptop against the deployed `ReencodeEngine` of the loaded profile's
app, under its OWN run id (its own ledger), over N rollouts of a saved responses
file. It exercises the exact code path `cv-whitebox` uses: the contrastive
capture, the rollout capture (both position schemes), and the logit monitor at the
profile's position cap. It prints tokens per second for each part and the
projected cost of the full pass at the profile's scope.

    set -a; . ./.env; set +a
    CV_PROFILE=full CV_RUN=bench-reencode uv run python scripts/bench_reencode.py \\
        --model qwen3.6-27b --rollouts data/results/runs/pilot/rollouts/responses_qwen3.6-27b.json --n 64

Pass the revision pin (`--revision`, or `hf_revisions` in the profile). The engine's
`config()` reports the loaded commit; a first call without a pin prints it.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time

import cv.bootstrap  # noqa: F401
from cv import config, failures, meter
from cv.serving.local.backend import ReencodeBackend
from cv.whitebox import contrastive_pairs, envelope, extraction

H100_HR = config.GPU_RATES_PER_HR["H100"]
OVERHEAD = 0.10


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.6-27b")
    ap.add_argument("--rollouts", required=True, help="a responses_<model>.json to sample from")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--full-rollouts", type=int, default=None,
                    help="rollouts per model in the full run (default: the profile's expected count)")
    ap.add_argument("--revision", default="", help="the commit to pin (default: the profile's hf_revisions)")
    ap.add_argument("--batch-tokens", type=int, default=0, help="override whitebox_batch_tokens (256 = one row per forward)")
    args = ap.parse_args()
    if args.batch_tokens:
        config.get().whitebox_batch_tokens = args.batch_tokens
    print(f"batch tokens: {config.whitebox_batch_tokens()}")

    with open(args.rollouts, encoding="utf-8") as f:
        samples = [s for s in json.load(f)["samples"] if failures.classify_sample(s) is None][:args.n]
    fam = envelope.detect_family(args.model)
    revision = args.revision or config.hf_revision(args.model)
    if not revision:
        raise SystemExit("pass --revision <commit> (the engine's config() reports it) or set hf_revisions in the profile")
    # One parameter set for every call: a different set would open a second container pool.
    backend = ReencodeBackend(args.model, config.hf_model_id(args.model), app_name=config.modal_app(),
                              revision=revision)
    print(f"preflight: {backend.preflight()}")
    print(f"engine: {json.dumps(backend.live)}")

    with meter.stage("reencode"):
        t0 = time.perf_counter()
        c_out = backend.capture(contrastive_pairs.build_rows())
        t_c = time.perf_counter() - t0
        rows = [extraction.build_capture_row(s, fam) for s in samples]
        t0 = time.perf_counter()
        r_out = backend.capture(rows)
        t_r = time.perf_counter() - t0
    tok_r = sum(o["n_prompt_tokens"] + o["n_response_tokens"] for o in r_out)
    per_rollout = tok_r / len(r_out)
    import numpy as np

    nan_c = sum(int(np.isnan(np.asarray(o["last_prompt"], dtype=np.float32)).any()) for o in c_out)
    nan_r = sum(int(np.isnan(np.asarray(o["last_prompt"], dtype=np.float32)).any()
                    or np.isnan(np.asarray(o["response_mean"], dtype=np.float32)).any()) for o in r_out)
    print(f"contrastive: {len(c_out)} rows in {t_c:.1f}s, rows with NaN: {nan_c}")
    print(f"capture rows with NaN: {nan_r} of {len(r_out)}")
    print(f"capture: {len(r_out)} rollouts, {tok_r} tokens in {t_r:.1f}s -> {tok_r / t_r:,.0f} tok/s "
          f"({per_rollout:,.0f} tok/rollout, {len(r_out[0]['layers'])} layers stored)")

    sentence, max_pos = config.logit_sentence(), config.logit_max_positions()
    l_rows = [r for s in samples for r in extraction.build_logit_rows(s, fam, sentence, max_pos)]
    with meter.stage("logit"):
        t0 = time.perf_counter()
        l_out = backend.sentence_logprob(l_rows)
        t_l = time.perf_counter() - t0
    tok_l = sum(o["n_context_tokens"] + o["n_tokens"] for o in l_out)
    per_rollout_l = tok_l / len(samples)
    print(f"logit: {len(l_out)} positions over {len(samples)} rollouts, {tok_l} tokens in {t_l:.1f}s -> "
          f"{tok_l / t_l:,.0f} tok/s ({len(l_out) / len(samples):.1f} positions/rollout, "
          f"{per_rollout_l:,.0f} tok/rollout)")
    scores = [o["logprob"] for o in l_out]
    finite = [s for s in scores if s == s]
    print("logit NaN rows:", [o["id"] for o in l_out if o["logprob"] != o["logprob"]][:10])
    print(f"logit scores: {len(scores) - len(finite)} NaN of {len(scores)}; "
          + (f"median {statistics.median(finite):.2f}, min {min(finite):.2f}, max {max(finite):.2f}" if finite else "no finite score"))

    n_full = args.full_rollouts or config.expected_rollouts_per_model() or 12600
    secs = n_full * per_rollout / (tok_r / t_r) + n_full * per_rollout_l / (tok_l / t_l)
    usd = secs / 3600 * (1 + OVERHEAD) * H100_HR
    print(f"\nprojected full pass for {args.model} at {n_full} rollouts: {secs / 3600:.2f} GPU-hr, "
          f"${usd:.2f} on H100 (+{OVERHEAD:.0%} overhead), one model")
    print(f"metered this benchmark: ${meter.spent():.4f}")
    print(f"\nPin for the profile:\n  hf_revisions:\n    {args.model}: {backend.revision}")


if __name__ == "__main__":
    main()

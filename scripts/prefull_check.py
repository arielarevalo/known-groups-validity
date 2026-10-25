#!/usr/bin/env python
"""Judge the pre-launch smoke (config/prefull.yaml) from what it left on disk.

Pure read, no API, no GPU. Prints PASS/FAIL per assertion and exits non-zero on any
FAIL. The assertions cover the full-run configuration plus the warm-reuse
question: did every Modal batch after the first in a stage hit a warm container?

    uv run python scripts/prefull_check.py            # reads data/results/runs/prefull
    uv run python scripts/prefull_check.py <run dir>  # any run dir
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

# A GPU row this long or longer includes a container start (weights downloaded to
# local SSD and loaded: about 5 min). A warm batch of a few short rollouts is well
# under a minute. The run may pay ONE such start, and only as its first GPU row (the
# preflight's config call, or the first batch); any later one is a reload, which
# means the container was released between batches.
COLD_START_S = 180.0
# The idle gap after which a new start is expected, not a reload: the app releases a
# container after its scaledown window (300 s), so a later row starts a new session.
IDLE_GAP_S = 600.0


def _rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def check(run_dir: Path) -> list[tuple[str, bool, str]]:
    """Return [(name, passed, detail)]; pure function over the run directory."""
    results: list[tuple[str, bool, str]] = []
    rollouts = run_dir / "rollouts"

    # 1) rollouts on disk with think tags, per model
    for model, tag in (("gemma-4-31b", "<think>"), ("qwen3.6-27b", "</think>")):
        p = rollouts / f"responses_{model}.json"
        if not p.exists():
            results.append((f"{model} rollouts present", False, f"missing {p.name}"))
            continue
        samples = json.loads(p.read_text(encoding="utf-8")).get("samples", [])
        tagged = sum(1 for s in samples if tag in (s.get("response") or ""))
        results.append((f"{model} rollouts present", len(samples) > 0, f"{len(samples)} samples"))
        results.append((f"{model} reasoning traces", tagged == len(samples) and tagged > 0,
                        f"{tagged}/{len(samples)} carry {tag}"))
        if model == "gemma-4-31b":
            providers = defaultdict(int)
            for s in samples:
                providers[s.get("provider") or "unreported"] += 1
            results.append(("gemma fp8 provider recorded", "unreported" not in providers and bool(providers),
                            ", ".join(f"{k}={v}" for k, v in sorted(providers.items()))))

    # 2) every ledgered failure was resolved by a later retry. The ledger keeps history
    #    (a row per failure, not per id), so a failure that a resume completed still
    #    shows; only an id that is STILL not done counts against the smoke.
    led = run_dir / "failures"
    unresolved, resolved = defaultdict(int), 0
    for p in (sorted(led.glob("*.jsonl")) if led.exists() else []):
        stage, _, model = p.stem.partition("_")
        if stage == "rollout":
            f = rollouts / f"responses_{model}.json"
            done = {s.get("id") for s in json.loads(f.read_text(encoding="utf-8")).get("samples", [])} if f.exists() else set()
        else:
            done = {r.get("id") for r in _rows(run_dir / "detectors" / f"{stage}_{model}.jsonl")}
        for r in _rows(p):
            if r.get("id") in done:
                resolved += 1
            else:
                unresolved[f"{stage}:{r.get('reason')}"] += 1
    detail = f"{resolved} failure(s) retried and completed" if resolved else "no failures"
    if unresolved:
        detail = "UNRESOLVED " + ", ".join(f"{k}={v}" for k, v in sorted(unresolved.items())) + f"; {detail}"
    results.append(("failures resolved by retry", not unresolved, detail))

    # 3) warm reuse: at most one cold start in the run, and only as the first GPU row
    gpu_rows = [r for r in _rows(run_dir / "_metrics.jsonl") if r.get("gpu")]
    if not gpu_rows:
        results.append(("modal batches metered", False, "no GPU rows in _metrics.jsonl"))
        return results
    failed = sum(1 for r in gpu_rows if r.get("failed"))
    results.append(("no failed GPU batches", failed == 0, f"{failed} failed of {len(gpu_rows)}"))
    secs = [float(r.get("gpu_seconds") or 0.0) for r in gpu_rows]
    ts = [float(r.get("ts") or 0.0) for r in gpu_rows]
    cold_idx = [i for i, s in enumerate(secs) if s >= COLD_START_S]
    stages = [str(r.get("stage")) for r in gpu_rows]
    # A start is expected as the first row of a session: the first GPU row of a model,
    # or a row that follows an idle gap longer than the container's scaledown window
    # (a resume on a later day pays a new start). Each model has its own container,
    # so the gap is read per model. Any other start is a reload.
    models = [str(r.get("model")) for r in gpu_rows]
    last_ts: dict[str, float] = {}
    session_start = set()
    for i in range(len(ts)):
        if models[i] not in last_ts or ts[i] - last_ts[models[i]] > IDLE_GAP_S:
            session_start.add(i)
        last_ts[models[i]] = ts[i]
    reloads = [i for i in cold_idx if i not in session_start]
    if not cold_idx:
        detail = f"{len(secs)} warm batches, max {max(secs):.0f}s, no start visible (pre-warmed)"
        ok = True
    elif not reloads:
        starts = ", ".join(f"{stages[i]}#{i}={secs[i]:.0f}s" for i in cold_idx)
        warm = [s for i, s in enumerate(secs) if i not in cold_idx]
        detail = (f"{len(cold_idx)} session start(s) ({starts}), "
                  f"then {len(warm)} warm batches, max {max(warm or [0]):.0f}s")
        ok = True
    else:
        later = ", ".join(f"{stages[i]}#{i}={secs[i]:.0f}s" for i in reloads)
        detail = f"reload(s) inside a session: {later}"
        ok = False
    results.append(("warm reuse: no reload after the first start", ok, detail))
    results.append(("several batches per stage", len(gpu_rows) >= 3,
                    f"{len(gpu_rows)} GPU rows across {sorted(set(stages))}"))
    return results


def main(argv: list[str]) -> int:
    run_dir = Path(argv[1]) if len(argv) > 1 else Path("data/results/runs/prefull")
    if not run_dir.exists():
        print(f"no run dir at {run_dir}")
        return 2
    results = check(run_dir)
    for name, ok, detail in results:
        print(f"[{'PASS' if ok else 'FAIL'}] {name:44s} {detail}")
    bad = sum(1 for _, ok, _ in results if not ok)
    print(f"\n{'SMOKE GREEN' if not bad else f'SMOKE RED: {bad} assertion(s) failed'}")
    print("Next: confirm config/full.yaml vllm_version matches the `[vllm] version X` line "
          "the container printed (the preflight refuses any other build).")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))

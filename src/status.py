#!/usr/bin/env python
"""Live spend / progress monitor entry point: render results/_metrics.jsonl as a
per-stage table (rows, tokens, USD), mean USD per rollout, wall-time, and spend vs the
budget cap. Pure read -- never calls an API, never raises on a missing/empty file. The
metrics-file parsing lives in cv.status (`load`); this is the rendering. Installs as the
`cv-status` console script.

    uv run cv-status
"""
from __future__ import annotations

from cv import config, failures, status


def _fmt_hms(ms: float) -> str:
    """Milliseconds -> h:mm:ss wall-clock string."""
    secs = int(ms / 1000.0)
    h, rem = divmod(secs, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


def report() -> None:
    path = config.metrics_path()
    summary = status.load(path)

    if summary.total.rows == 0:
        print(f"no metrics yet ({path})")
        if summary.bad_lines:
            print(f"({summary.bad_lines} unparseable line(s) skipped)")
        return

    # per-stage table, stages in stable alphabetical order
    header = f"{'stage':<10} {'rows':>6} {'prompt_tok':>12} {'compl_tok':>12} {'usd':>10}"
    print(header)
    print("-" * len(header))
    for stage in sorted(summary.per_stage):
        st = summary.per_stage[stage]
        print(
            f"{stage:<10} {st.rows:>6} {st.prompt_tok:>12,} "
            f"{st.completion_tok:>12,} {st.usd:>10.4f}"
        )
    tot = summary.total
    print("-" * len(header))
    print(
        f"{'TOTAL':<10} {tot.rows:>6} {tot.prompt_tok:>12,} "
        f"{tot.completion_tok:>12,} {tot.usd:>10.4f}"
    )

    # mean USD per rollout (rollout-stage rows only)
    roll = summary.per_stage.get("rollout")
    if roll and roll.rows:
        print(f"\nmean USD / rollout : ${roll.usd / roll.rows:.4f}  (n={roll.rows})")
    else:
        print("\nmean USD / rollout : n/a (no rollout rows)")

    # wall-time total
    print(f"total wall-time    : {_fmt_hms(tot.wall_ms)}  ({tot.wall_ms / 1000.0:.1f}s)")

    # spent vs cap vs headroom
    cap = config.budget_usd()
    headroom = cap - tot.usd
    pct = (tot.usd / cap * 100.0) if cap else 0.0
    print(
        f"budget             : spent ${tot.usd:.4f} / cap ${cap:.2f}  "
        f"({pct:.1f}%, headroom ${headroom:.4f})"
    )

    if summary.bad_lines:
        print(f"\n({summary.bad_lines} unparseable line(s) skipped)")

    _failures()


def _failures() -> None:
    """The failure ledger, per stage and model, by reason. Rows, not distinct ids:
    a sample that failed on three runs shows three. Empty ledger prints nothing."""
    led = failures.summary()
    if not led:
        return
    print("\nfailure ledger     : (data/results/runs/<run>/failures/)")
    for name in sorted(led):
        reasons = ", ".join(f"{r}={n}" for r, n in sorted(led[name].items()))
        print(f"  {name:<22} {sum(led[name].values()):>6}  {reasons}")


def main() -> None:
    report()


if __name__ == "__main__":
    main()

"""Parse results/_metrics.jsonl into typed per-stage + total tallies.

Pure read: never calls an API, never raises on a missing/empty file. Each metrics line is
{ts, stage, model, prompt_tok, completion_tok, usd, wall_ms} as written by cv.meter.
`load` folds every valid line into per-stage and grand-total `StageTotals`. The rendering
(the spend table + budget line) is the `cv-status` console script (src/status.py).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class StageTotals:
    """Running totals for one pipeline stage."""

    rows: int = 0
    prompt_tok: int = 0
    completion_tok: int = 0
    usd: float = 0.0
    wall_ms: float = 0.0


@dataclass
class Summary:
    """Whole-file roll-up, keyed by stage plus a grand total."""

    per_stage: dict[str, StageTotals] = field(default_factory=dict)
    total: StageTotals = field(default_factory=StageTotals)
    bad_lines: int = 0


def load(path: Path) -> Summary:
    """Fold every valid metrics line into per-stage and total tallies."""
    summary = Summary()
    if not path.exists():
        return summary
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                summary.bad_lines += 1
                continue
            if not isinstance(row, dict):
                summary.bad_lines += 1
                continue
            stage = str(row.get("stage", "unknown"))
            st = summary.per_stage.setdefault(stage, StageTotals())
            for bucket in (st, summary.total):
                bucket.rows += 1
                bucket.prompt_tok += int(row.get("prompt_tok", 0) or 0)
                bucket.completion_tok += int(row.get("completion_tok", 0) or 0)
                bucket.usd += float(row.get("usd", 0.0) or 0.0)
                bucket.wall_ms += float(row.get("wall_ms", 0.0) or 0.0)
    return summary

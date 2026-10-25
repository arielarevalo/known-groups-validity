"""Append-as-completed JSONL checkpointing for the detector stages.

Upstream's SR/TP/PG/CoT helpers gather everything then return once, so a crash
loses the batch. Our runners use this to bring all four detectors to the same
resumability the rollout stage already has: load done ids, run only the rest,
append each result the moment it lands.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Iterable

_write_lock = threading.Lock()


def load_done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            rid = rec.get("id")
            if rid is not None:
                done.add(rid)
    return done


def remaining(samples: list[dict[str, Any]], done: Iterable[str], id_field: str = "id") -> list[dict[str, Any]]:
    done_set = set(done)
    return [s for s in samples if s.get(id_field) not in done_set]


def append(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False)
    with _write_lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def load_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out

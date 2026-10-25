"""Pooled activation vectors on disk (`operations 0008`): one fp16 array per
``(row, layer)`` under ``<run>/whitebox/activations/<model>/<scheme>/``, written
as one ``.npz`` shard per re-encode chunk so a stopped pass resumes by row id."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cv.whitebox import constants
from cv.whitebox.schema import Activations


def build_shard_path(scheme_dir: Path, index: int) -> Path:
    return scheme_dir / constants.shards.name_format.format(index=index)


def list_shards(scheme_dir: Path) -> list[Path]:
    return sorted(scheme_dir.glob(constants.shards.pattern)) if scheme_dir.exists() else []


def write_shard(scheme_dir: Path, ids: list[str], layers: list[int], vectors: Activations) -> Path:
    """Append one shard, stored as float16. The write is atomic (temp then
    rename), so a stopped pass never leaves a torn shard."""
    scheme_dir.mkdir(parents=True, exist_ok=True)
    vectors = np.asarray(vectors, dtype=np.float16)
    if vectors.ndim != 3 or vectors.shape[0] != len(ids) or vectors.shape[1] != len(layers):
        raise ValueError(
            f"shard shape {vectors.shape} does not match {len(ids)} ids x {len(layers)} layers",
        )
    path = build_shard_path(scheme_dir, len(list_shards(scheme_dir)))
    temporary = path.with_suffix(constants.shards.temporary_suffix)
    with open(temporary, "wb") as file:
        np.savez(
            file,
            ids=np.asarray(ids, dtype=str),
            layers=np.asarray(layers, dtype=np.int64),
            **{constants.shards.vectors_key: vectors},
        )
    temporary.replace(path)
    return path


def list_done_ids(scheme_dir: Path) -> set[str]:
    ids: set[str] = set()
    for path in list_shards(scheme_dir):
        with np.load(path) as shard:
            ids.update(str(row_id) for row_id in shard["ids"])
    return ids


def load_scheme(scheme_dir: Path) -> tuple[list[str], list[int], Activations]:
    """Every shard of a scheme concatenated: ``(ids, layers, vectors)`` with the
    vectors as float32. Raises when the shards disagree on their layers."""
    ids: list[str] = []
    layers: list[int] | None = None
    parts: list[Activations] = []
    for path in list_shards(scheme_dir):
        with np.load(path) as shard:
            shard_layers = [int(layer) for layer in shard["layers"]]
            if layers is None:
                layers = shard_layers
            elif shard_layers != layers:
                raise ValueError(f"{path} holds layers {shard_layers}, earlier shards {layers}")
            ids.extend(str(row_id) for row_id in shard["ids"])
            parts.append(np.asarray(shard[constants.shards.vectors_key], dtype=np.float32))
    if not parts:
        return [], layers or [], np.zeros((0, 0, 0), dtype=np.float32)
    return ids, layers or [], np.concatenate(parts, axis=0)


def write_meta(model_dir: Path, meta: dict) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")


def read_meta(model_dir: Path) -> dict:
    path = model_dir / "meta.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

"""Internalised EvalAwareBench loader (docs/operations/accepted/0005, part 1).

Source the benchmark from the HuggingFace flat dataset and unflatten it into the
``(task, valence, config_name, factor_config, prompt)`` shape the pipeline needs,
replacing the submodule task-YAML + upstream ``render`` path. The HF ``prompts``
config already carries the rendered ``prompt`` and the full F1..F8 ``config``
dict per ``(task, valence, config)``, so nothing is re-rendered; prompts are
taken exactly as published.

``config_name`` is derived by diffing each row's factor config against the
baseline defaults rather than by parsing the ``factors_varied`` label strings,
which is robust to label formatting and matches how the pilot recorded it.

Reproducibility: the dataset revision is pinned (the 0005 invariant). The
byte-for-byte fidelity check against the committed pilot prompts lives in
``tests/test_dataset_parity.py``.
"""
from __future__ import annotations

from dataclasses import dataclass

_DATASET = "aisa-group/EvalAwareBench"
# Pinned EvalAwareBench revision (main @ 2026-06-04). The fidelity gate is only
# meaningful against a fixed revision.
REVISION = "736bda484525041972fa214eaf72f3beba492666"

# Baseline factor defaults, verified against the HF `prompts` baseline rows:
# every factor at its default level (F3=casual, F7=prose, the rest=realistic).
BASELINE_CONFIG = {
    "F1": "realistic", "F2": "realistic", "F3": "casual", "F4": "realistic",
    "F5": "realistic", "F6": "realistic", "F7": "prose", "F8": "realistic",
}
FACTORS = tuple(sorted(BASELINE_CONFIG))  # ("F1", ..., "F8")


def config_name(factor_config: dict) -> str:
    """Single-token ``config_name`` in ``{baseline, F1..F8, <stacked>}``.

    Derived by diffing against ``BASELINE_CONFIG``: ``baseline`` if no factor
    differs, ``Fk`` if exactly factor ``k`` is toggled off its default, otherwise
    a ``+``-joined stacked label (not part of the v1 criterion, design/0003).
    """
    diff = [f for f in FACTORS if str(factor_config.get(f)) != BASELINE_CONFIG[f]]
    if not diff:
        return "baseline"
    if len(diff) == 1:
        return diff[0]
    return "+".join(diff)


@dataclass(frozen=True)
class BenchPrompt:
    """One unflattened benchmark prompt (no sampling/reps dimension; that is the
    rollout's job)."""
    id: str
    task_id: str
    task_name: str
    valence: str
    config_name: str
    factor_config: dict
    prompt: str
    description: str


def load_prompts(
    revision: str = REVISION, single_factor_only: bool = True
) -> list[BenchPrompt]:
    """Load and unflatten the EvalAwareBench ``prompts`` config from HF.

    ``single_factor_only`` keeps the ``{baseline, F1..F8}`` configs of Experiment
    A. With it off, the full ``2^8`` cube is returned and `cv.stacking` draws
    Experiment B's subset from it (design 0029 D2).
    """
    from datasets import load_dataset

    ds = load_dataset(_DATASET, "prompts", split="train", revision=revision)
    out: list[BenchPrompt] = []
    for r in ds:
        fc = dict(r["config"])
        cn = config_name(fc)
        if single_factor_only and "+" in cn:
            continue
        out.append(
            BenchPrompt(
                id=f"{r['task_id']}__{cn}",
                task_id=r["task_id"],
                task_name=r["task_name"],
                valence=r["valence"],
                config_name=cn,
                factor_config=fc,
                prompt=r["prompt"],
                description=r.get("description", ""),
            )
        )
    return out

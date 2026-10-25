"""The rows of the organism session (design 0035 items 3 and 5), ready for the engine.

Two sets, each read in both arms (adapter on and adapter off):
  hua     the five conditions of Hua et al. on the 128 BigCodeBench questions, with
          and without the hint, with the Nemotron thinking suffix "\\ndetailed thinking on".
  pairs   the 16 contrastive pairs, each side as a user turn after the system turn
          "detailed thinking on", as Hua et al. wrap them (their system prompt is
          "detailed thinking on" and nothing else).

The pairs go into both arms. Hua et al. build their direction on the base model (adapter
off) and apply it to the organism. The own-model direction needs the pairs with the
adapter on. One session gives both.

Every row is ``{"id", "conversation"}`` and passes `app.check_row`. The engine reads the
three positions of every row, so a row carries no position flag.
"""
from __future__ import annotations

import json
from pathlib import Path

from cv.serving.organism import app as organism_app
from cv.whitebox import contrastive_pairs, hua_conditions

QUESTIONS_PATH = Path(hua_conditions.__file__).with_name("bcb_questions.json")
# The flags of the study engine. The organism engine reads all three positions of every
# row, and `content_end` is the cut position, so these flags select nothing here.
POSITION_FLAGS = ("cut_after_content", "read_positions")


def normalize_row(row: dict) -> dict:
    """``row`` as the organism engine takes it: the position flags dropped, then
    `app.check_row`. A row with ``assistant_text`` raises, because it reads a
    different position."""
    if row.get("assistant_text"):
        raise ValueError(f"row {row.get('id')!r}: the organism engine reads prompt positions only")
    out = {k: v for k, v in row.items() if k not in POSITION_FLAGS}
    organism_app.check_row(out)
    return out


def normalize_rows(rows: list[dict]) -> list[dict]:
    """Every row through `normalize_row`. The ids must be unique."""
    out = [normalize_row(r) for r in rows]
    ids = [r["id"] for r in out]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{len(ids) - len(set(ids))} duplicate row ids")
    return out


def build_hua_rows(questions: list[dict] | None = None) -> list[dict]:
    """The five conditions, with and without the hint, with the Nemotron thinking suffix.

    The order interleaves the conditions: question by question, and in each question
    the hint arm and then the five conditions. So a chunk of the fixed grid holds every
    condition and both hint arms, and the batch shape (the pad length and the GEMM shape
    of the forward) does not line up with the label of a contrast between conditions.
    The order is fixed, so both adapter arms and every resume see the same partners."""
    if questions is None:
        questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    n = len(questions)
    by_hint = {hint: hua_conditions.build_condition_rows(
        questions, hint=hint, thinking_suffix=hua_conditions.NEMOTRON_THINKING_SUFFIX) for hint in (True, False)}
    n_conditions = len(hua_conditions.CONDITIONS)
    for hint, block in by_hint.items():
        # `build_condition_rows` gives the rows condition by condition, then question by question.
        if len(block) != n_conditions * n:
            raise ValueError(f"hint {hint}: {len(block)} rows, expected {n_conditions} x {n}")
    rows = []
    for q, item in enumerate(questions):
        for hint in (True, False):
            for c, (condition, *_) in enumerate(hua_conditions.CONDITIONS):
                row = by_hint[hint][c * n + q]
                if row["id"] != hua_conditions.build_row_id(condition, hint, item["question_id"]):
                    raise ValueError(f"row {row['id']!r} is not {condition}, hint {hint}, question {item['question_id']}")
                rows.append(row)
    return normalize_rows(rows)


def build_pair_rows() -> list[dict]:
    """The 16 pairs, both sides, as the user turn after the thinking system turn.

    The id suffix after ``@`` names the wrapper ("thinking": the thinking system turn),
    not a read position. The store name gives the position (``pairs__content_end``, and
    so on). So the direction of a store comes from
    ``classification.extract_eval_direction(ids, vectors, position="thinking")``,
    whatever position the store holds."""
    system = {"role": "system", "content": organism_app.THINKING_SYSTEM}
    rows = [{"id": f"p{number:02d}_{side}@thinking",
             "conversation": [system, {"role": "user", "content": text}]}
            for number, pair in enumerate(contrastive_pairs.PAIRS, 1)
            for side, text in zip(contrastive_pairs.SIDES, pair, strict=True)]
    return normalize_rows(rows)


def build_sets() -> dict[str, list[dict]]:
    """The row sets of the session, by store name. The pairs go first: they are few,
    and the direction needs them."""
    return {"pairs": build_pair_rows(), "hua": build_hua_rows()}

"""Experiment B: the stacked-cues sampler (design 0029 D2, operations 0019 section 1).

Experiment A runs ``baseline + F1..F8`` on every item. Experiment B adds, on a
25-task-per-valence subset (50 items), combinations of the F1..F7 cues:
``min(max_per_k, C(7, k))`` combinations per active-factor count ``k``, drawn per
item. F8 (the explicit "this is an evaluation" disclosure) never enters a stack.
The ``k=1`` draws are the seven singles Experiment A already covers, so the new
work is ``k=2..7``: ``10+10+10+10+7+1 = 48`` stacked configs per item.

Every draw is seeded. The seed, the chosen tasks and the draws per item are
written into the rollout file (``StackPlan.to_record``), so the plan can be
checked without this code. The released dataset carries the full ``2^8`` cube,
so every drawn combination exists as a published prompt; the rollout verifies
that before it spends.
"""
from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from typing import Iterable

STACK_FACTORS = ("F1", "F2", "F3", "F4", "F5", "F6", "F7")
HELD_OUT = ("F8",)


def active_factors(config_name: str) -> tuple[str, ...]:
    """The factors a config toggles off their defaults: ``()`` for baseline,
    ``("F3",)`` for a single, ``("F1", "F3")`` for a stack."""
    c = (config_name or "").strip()
    if not c or c == "baseline":
        return ()
    return tuple(c.split("+"))


def n_active(config_name: str) -> int:
    return len(active_factors(config_name))


def is_stacked(config_name: str) -> bool:
    return n_active(config_name) >= 2


def in_stack_space(config_name: str) -> bool:
    """True for baseline and for any config built from F1..F7 only. False when
    F8 is active, so the held-out control never enters the dose-response."""
    return all(f in STACK_FACTORS for f in active_factors(config_name))


def select_task_names(task_names: Iterable[str], per_valence: int, seed: int) -> list[str]:
    """The subset's tasks: ``per_valence`` task names drawn without replacement.
    Both valences of a chosen task are in, so the subset is balanced by pairs."""
    names = sorted(set(task_names))
    if per_valence >= len(names):
        return names
    return sorted(random.Random(f"{seed}:tasks").sample(names, per_valence))


def draw_combos(task_id: str, k: int, max_per_k: int, seed: int) -> list[str]:
    """``min(max_per_k, C(7, k))`` stacked config names for one item at one
    ``k``, in the dataset's own ``F1+F3`` label order. Seeded per item and per
    ``k``, so a change of ``max_per_k`` at one level does not move the others."""
    combos = list(itertools.combinations(STACK_FACTORS, k))
    if len(combos) > max_per_k:
        combos = random.Random(f"{seed}:{task_id}:{k}").sample(combos, max_per_k)
    return sorted("+".join(c) for c in combos)


@dataclass(frozen=True)
class StackPlan:
    tasks_per_valence: int
    max_per_k: int
    seed: int
    task_names: tuple[str, ...]
    draws: dict  # task_id -> tuple of stacked config names, k >= 2

    def keeps(self, task_id: str, config_name: str) -> bool:
        return config_name in self.draws.get(task_id, ())

    def pairs(self) -> set[tuple[str, str]]:
        return {(tid, cfg) for tid, cfgs in self.draws.items() for cfg in cfgs}

    def n_new_per_item(self) -> int:
        return max((len(v) for v in self.draws.values()), default=0)

    def to_record(self) -> dict:
        return {
            "tasks_per_valence": self.tasks_per_valence, "max_per_k": self.max_per_k,
            "seed": self.seed, "factors": list(STACK_FACTORS), "held_out": list(HELD_OUT),
            "task_names": list(self.task_names),
            "draws": {tid: list(cfgs) for tid, cfgs in sorted(self.draws.items())},
        }


def build_plan(prompts, *, tasks_per_valence: int, max_per_k: int, seed: int) -> StackPlan:
    """The plan over a prompt set (any object with ``task_name`` and ``task_id``).
    Items are drawn from the tasks present, so a narrowed profile gets a plan
    over its own scope."""
    names = select_task_names((p.task_name for p in prompts), tasks_per_valence, seed)
    chosen = set(names)
    task_ids = sorted({p.task_id for p in prompts if p.task_name in chosen})
    draws = {}
    for tid in task_ids:
        cfgs: list[str] = []
        for k in range(2, len(STACK_FACTORS) + 1):
            cfgs.extend(draw_combos(tid, k, max_per_k, seed))
        draws[tid] = tuple(cfgs)
    return StackPlan(tasks_per_valence, max_per_k, seed, tuple(names), draws)


def apply_plan(prompts, plan: StackPlan):
    """Experiment A's prompts (every non-stacked config) plus the planned stacks."""
    return [p for p in prompts
            if not is_stacked(p.config_name) or plan.keeps(p.task_id, p.config_name)]


def missing_from(prompts, plan: StackPlan) -> list[tuple[str, str]]:
    """Planned (item, config) pairs the prompt set does not carry."""
    have = {(p.task_id, p.config_name) for p in prompts}
    return sorted(plan.pairs() - have)

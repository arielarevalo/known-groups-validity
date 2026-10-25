"""Shared contract types: the `Step` abstraction and the `Event` vocabulary a
step may yield for reporting.

Pure data only — no orchestration, no I/O, no W&B. This is the home for every
shared dataclass/protocol in the package (if it grows, it becomes a
``cv/contracts/`` package under the same rule); shared types never live
intermixed with logic or orchestration. A step depends on this module and
nothing else cross-cutting, which is what keeps W&B (and the run, and the
ordering policy) entirely out of step code.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator, Union


@dataclass(frozen=True)
class Progress:
    """Live progress within a step: ``done`` of ``total`` units processed."""

    done: int
    total: int
    unit: str = "samples"


@dataclass(frozen=True)
class Metric:
    """A single number worth reporting (mapped to ``Operations/<step>/<key>``)."""

    key: str
    value: float


@dataclass(frozen=True)
class Note:
    """A human-readable log line (reported to the console, not to metrics)."""

    text: str


# What a step's run() may yield. W&B-agnostic by construction; the driver
# decides what (if anything) each Event means for observability.
Event = Union[Progress, Metric, Note]


@dataclass(frozen=True)
class Step:
    """A pipeline step. ``run`` does the work and yields Events; it knows nothing
    about W&B, the run id, or the order it runs in.

    - ``metered``:      may incur budgeted (OpenRouter) spend, so the driver
                        applies the budget guard before running it.
    - ``backend_warm``: queries the self-hosted GPU target, so the driver orders
                        it before judge-only steps to keep the container warm.
    """

    name: str
    run: Callable[[str], Iterator[Event]]
    metered: bool = True
    backend_warm: bool = False

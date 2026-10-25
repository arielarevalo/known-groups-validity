"""Target serving abstraction (docs/operations/accepted/0007).

The rollout and the SR/TP detectors all need the same primitive: given a target
conversation (a list of role/content messages) plus generation params, produce one
completion. This module defines that primitive (`GenRequest`) and the backend
protocol (`TargetBackend.agenerate`, a batched async generate). Two backends
implement it:

  - `cv.serving.api.ApiBackend` — any OpenAI-compatible endpoint (OpenRouter by
    default). Wraps the golden-tested `cv.client` primitives unchanged, so it is the
    dev/CI/offline-fallback target backend AND the only judge path.
  - `cv.serving.local.backend.ModalBackend` — targets self-hosted on Modal GPUs
    (vLLM for the black-box pilot; HF + forward hooks for the white-box experiment,
    which an HTTP API could never give us — the reason serving is a function, not a
    served endpoint).

`cv.config.target_backend` selects which one the rollout + SR/TP run against; the
judge always uses the API backend.

Completion contract: every backend returns one `GenResult` per request. Its `text`
is the `<think>{reasoning}</think>{content}` envelope the CoT detector parses when a
reasoning trace exists, else the bare content. An UNCLOSED `<think>` in `text` means
the generation stopped inside the trace: the trace is never the answer (launch
audit, H16). The batch is resilient: a single failed item yields an empty result
rather than sinking the batch (matching the upstream resilient-batch behaviour),
and the caller's failure ledger records it.

`GenResult` also carries what the text alone cannot: the finish reason (the only
reliable truncation signal), the provider that served an API call, the raw
reasoning and content fields before the envelope was built (the re-encode needs
the model's real output), and whether the engine ran with the
thinking template switched on (a fallback without it yields trace-free rollouts
that look successful).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Protocol, runtime_checkable

# A single chat message: {"role": "system"|"user"|"assistant", "content": str}.
Message = dict
Conversation = List[Message]


@dataclass
class GenRequest:
    """One generation request. `messages` carries the conversation WITHOUT the system
    turn (the backend prepends `system_prompt`); this matches both the single-turn
    rollout ([user]) and the multi-turn SR/TP follow-ups ([user, assistant, user])."""

    system_prompt: str
    messages: Conversation = field(default_factory=list)
    temperature: Optional[float] = 0.3
    max_tokens: int = 8000
    reasoning_effort: str = "high"
    prefill: Optional[str] = None


@dataclass
class GenResult:
    """One completion. `text` is the envelope the pipeline stores and parses."""

    text: str = ""
    # "stop" | "length" | "error" | "" (not reported). "length" is truncation.
    finish_reason: str = ""
    # Raw API fields before the envelope was built; None on engines that emit the
    # native token stream directly (there the text IS the model's output).
    reasoning: Optional[str] = None
    content: Optional[str] = None
    provider: str = ""            # serving provider, when the API reports it
    thinking: Optional[bool] = None  # engine ran with the thinking template on; None = n/a
    error: str = ""               # exception summary when the item failed

    @property
    def truncated(self) -> bool:
        return self.finish_reason == "length"


def as_result(x) -> "GenResult":
    """Accept a bare string (older backends, test doubles) as a GenResult."""
    return x if isinstance(x, GenResult) else GenResult(text=str(x or ""))


@runtime_checkable
class TargetBackend(Protocol):
    """A target model behind some serving mechanism."""

    name: str

    async def agenerate(self, requests: List[GenRequest]) -> List[GenResult]:
        """Complete every request, returning one result per request in input order.
        Failures degrade to an empty result for that item; the call itself does not
        raise, except for the budget stop, which always propagates."""
        ...

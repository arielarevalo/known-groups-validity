"""The array shapes and record types the white-box channel passes around.

Every annotation shares one vocabulary: ``rows`` are prompts or samples, ``layers``
the captured decoder layers, ``dim`` the residual width. A function that takes
``Activations`` and returns ``LayerScores`` has collapsed ``dim``; nothing else
needs saying about it.
"""

from __future__ import annotations

from typing import Any, Literal, Required, TypedDict

import numpy as np
from jaxtyping import Bool, Float, Int, Shaped

Family = Literal["qwen", "gemma", "llama"]
"""A model family with a known native reasoning-trace format (`envelope`)."""
Message = dict[str, str]
"""One chat turn: ``{"role": ..., "content": ...}``."""
Sample = dict[str, Any]
"""One stored rollout row, the shape `cv.rollout` persists."""

Activations = Float[np.ndarray, "rows layers dim"]
LayerVectors = Float[np.ndarray, "layers dim"]
Direction = Float[np.ndarray, "layers dim"]
Directions = Float[np.ndarray, "count layers dim"]
LayerScores = Float[np.ndarray, "rows layers"]
LayerCosines = Float[np.ndarray, "layers"]
RowScores = Float[np.ndarray, "rows"]
Labels = Int[np.ndarray, "rows"]
"""The cued criterion per row: 0 baseline, 1 cued, -1 scored but never fitted on."""
Indices = Int[np.ndarray, "rows"]
TaskNames = Shaped[np.ndarray, "rows"]
Mask = Bool[np.ndarray, "rows"]


class CaptureRow(TypedDict, total=False):
    """What the engine's ``capture`` takes: a conversation, the assistant text that
    followed it, or the flag to cut right after the last user content instead."""

    id: Required[str]
    conversation: Required[list[Message]]
    assistant_text: str
    cut_after_content: bool


class CapturedVectors(TypedDict):
    """What ``capture`` returns per row: the residual stream at the last prompt token
    and the mean over the response tokens, at the captured layers, fp16."""

    id: str
    layers: list[int]
    last_prompt: LayerVectors
    response_mean: LayerVectors | None
    n_prompt_tokens: int
    n_response_tokens: int


class LogitRow(TypedDict):
    """What the engine's ``sentence_logprob`` takes: the sentence to score after the
    rendered conversation plus the assistant prefix."""

    id: str
    conversation: list[Message]
    assistant_prefix: str
    sentence: str


class SentenceLogprob(TypedDict):
    id: str
    logprob: float
    n_tokens: int
    n_context_tokens: int

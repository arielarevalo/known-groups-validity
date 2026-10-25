"""The white-box channel's constants, grouped by what they govern. Read them as
``constants.hyperparams.n_folds`` or ``constants.schemes.last_prompt``."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from cv.whitebox.schema import Family


@dataclass(frozen=True)
class Hyperparams:
    """Fixed before any data is seen; the fit chooses nothing about them."""

    n_folds: int = 5
    n_null_directions: int = 24
    """Hua et al. (arXiv:2510.20487) section 5.3 check their vector against 24 random ones."""
    null_seed: int = 0
    logistic_regularisation: float = 1.0
    """sklearn's ``C``, the inverse regularisation strength of the logistic cross-check."""
    logistic_max_iterations: int = 2000
    feature_std_floor: float = 1e-6


@dataclass(frozen=True)
class Schemes:
    """The capture schemes, each a directory of shards under a model's activations."""

    contrastive: str = "contrastive"
    last_prompt: str = "last_prompt"
    response_mean: str = "response_mean"


@dataclass(frozen=True)
class Shards:
    """The on-disk format of one activation shard."""

    vectors_key: str = "x"
    name_format: str = "{index:05d}.npz"
    pattern: str = "*.npz"
    temporary_suffix: str = ".npz.tmp"


@dataclass(frozen=True)
class Records:
    sample_fields: tuple[str, ...] = ("id", "task_id", "task_name", "valence", "config_name", "sample_idx")
    """The sample columns every detector record and logit score carries."""


@dataclass(frozen=True)
class TraceMarkers:
    """What each model family EMITS to open and close its reasoning trace, and how a
    model key maps to a family. The opener is empty when the chat template already
    ends the generation prompt with it: Qwen 3.6's template at 6a9e13bd renders
    ``<|im_start|>assistant\n<think>\n`` with or without ``enable_thinking``, so the
    model's own text starts inside the trace and ends it with ``</think>`` (verified
    2026-09-09 from the pinned ``chat_template.jinja``; the prefull rollouts show the
    same). Gemma 4's template at 842da379 ends at ``<|turn>model\n`` with thinking on,
    and the model emits ``<|channel>thought\n``."""

    family_by_needle: Mapping[str, Family] = field(
        default_factory=lambda: MappingProxyType(
            {
                "qwen": "qwen",
                "gemma": "gemma",
                "llama": "llama",
                "nemotron": "llama",
            },
        ),
    )
    opening: Mapping[Family, str] = field(
        default_factory=lambda: MappingProxyType(
            {
                "qwen": "",
                "gemma": "<|channel>thought\n",
                "llama": "<think>\n",
            },
        ),
    )
    closing: Mapping[Family, str] = field(
        default_factory=lambda: MappingProxyType(
            {
                "qwen": "\n</think>\n\n",
                "gemma": "<channel|>",
                "llama": "\n</think>\n\n",
            },
        ),
    )


@dataclass(frozen=True)
class SentenceRules:
    boundary: re.Pattern[str] = re.compile(r"(?<=[.!?])[\"')\]]*\s+|\n{2,}")
    """A sentence end followed by whitespace, or a blank line."""


hyperparams = Hyperparams()
schemes = Schemes()
shards = Shards()
records = Records()
trace_markers = TraceMarkers()
sentence_rules = SentenceRules()

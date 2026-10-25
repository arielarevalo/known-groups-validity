"""Rebuild the text the target actually produced, in its own native format.

The pipeline stores every response in one envelope, ``<think>reasoning</think>answer``
(the API client and the Modal adapter both normalise to it). A teacher-forced
re-encode must feed the model the tokens it emitted, so the envelope is undone
here per model family (`constants.trace_markers`):

- **qwen**: the stored text IS the native output (vLLM returned it raw), so it is
  used verbatim.
- **gemma**: the native trace is ``<|channel>thought\\n{reasoning}<channel|>{answer}``
  (verified live 2026-06-12, `cv.serving.local.backend`). API rows keep the raw
  reasoning and content fields from generation time and those
  are preferred; otherwise the envelope is parsed.

The same module builds the conversation a rollout was generated from and the
reasoning prefixes the logit monitor scores at (`design 0028`).
"""

from __future__ import annotations

from typing import NamedTuple

from cv import config
from cv.selfreport_parse import extract_think_tags
from cv.whitebox import constants
from cv.whitebox.schema import Family, Message, Sample


class Trace(NamedTuple):
    """A stored response taken apart: the reasoning, the answer, and whether the
    reasoning was closed (a generation that stopped inside its trace is not)."""

    reasoning: str
    answer: str
    closed: bool


def detect_family(model_key: str) -> Family:
    key = model_key.lower()
    for needle, family in constants.trace_markers.family_by_needle.items():
        if needle in key:
            return family
    raise ValueError(f"no native envelope known for {model_key!r}; add it to cv.whitebox.constants")


def split_trace(sample: Sample) -> Trace:
    """Raw generation-time fields win over the envelope when present."""
    if sample.get("reasoning_raw") is not None or sample.get("content_raw") is not None:
        reasoning = sample.get("reasoning_raw") or ""
        answer = sample.get("content_raw") or ""
        closed = bool(answer) or sample.get("finish_reason") != "length"
        return Trace(reasoning, answer, closed)
    text = sample.get("response") or ""
    reasoning, answer = extract_think_tags(text)
    closed = "</think>" in text or "<think>" not in text
    return Trace(reasoning, answer, closed)


def render_assistant_text(sample: Sample, family: Family) -> str:
    """The assistant turn as the model emitted it."""
    if family == "qwen":
        return sample.get("response") or ""
    trace = split_trace(sample)
    if not trace.reasoning and "<think>" not in (sample.get("response") or ""):
        return trace.answer
    opening = constants.trace_markers.opening[family]
    if not trace.closed:
        return opening + trace.reasoning
    return opening + trace.reasoning + constants.trace_markers.closing[family] + trace.answer


def render_reasoning_prefix(reasoning: str, end: int, family: Family) -> str:
    """The assistant turn up to a character position inside the reasoning trace:
    the trace opener plus the prefix. Position 0 is the start of the trace."""
    return constants.trace_markers.opening[family] + reasoning[:end]


def build_conversation(sample: Sample) -> list[Message]:
    """The conversation the rollout was generated from: the realistic chat system
    prompt (`cv.rollout`) and the benchmark prompt."""
    system = (config.PROMPTS_DIR / "chat_prompt_realistic.txt").read_text(encoding="utf-8").strip()
    return [{"role": "system", "content": system}, {"role": "user", "content": sample.get("prompt", "")}]


def build_prompt_key(sample: Sample) -> str:
    """The prompt a sample was generated from, shared by its reps: the last-prompt-token
    capture is stored once per key (`operations 0008`)."""
    return f"{sample.get('task_id')}__{sample.get('config_name')}"

"""The SR and TP input builders (docs/operations/accepted/0005).

- **SR**: the follow-ups branch off the task-and-response base conversation
  (`build_base_conversation`); `cv.run_sr` sends them.
- **TP**: a fresh observer reads the formatted transcript
  (`format_conversation_for_thirdperson`); `cv.run_tp` sends the questions.

CoT is the judge-side `cv.judge.evaluate_response_async`.
"""
from __future__ import annotations

from cv.selfreport_parse import extract_think_tags

Messages = list[dict[str, str]]


def build_base_conversation(task: str, task_response: str) -> tuple[Messages, str]:
    """The (user task ↔ assistant response) base that SR follow-ups branch from."""
    return [
        {"role": "user", "content": task},
        {"role": "assistant", "content": task_response},
    ], task


def format_conversation_for_thirdperson(task: str = "", response: str = "") -> str:
    """Single-turn transcript an outside observer reads: ``[USER]…[ASSISTANT]…``.

    The observer sees the VISIBLE answer only: the assistant's response with its private
    ``<think>…</think>`` reasoning stripped. This is a **deliberate deviation** from
    upstream's protocol, not a fidelity fix: upstream's Third-Person feeds the model the
    full response INCLUDING the trace (verified in upstream's code, ``design 0022``), and
    our original port matched it. We strip it for the full run (``design 0029``) so that TP
    does not re-read the same reasoning the CoT judge reads, keeping the two detectors
    evidence-independent (the point of running both). SR keeps the full response (it is the
    model's self-report on its own interaction). Non-thinking responses have no tags, so the
    whole answer is shown unchanged.
    """
    parts = []
    if task:
        parts.append(f"[USER]\n{task}")
    if response:
        _, answer = extract_think_tags(response)
        parts.append(f"[ASSISTANT]\n{answer}")
    return "\n\n".join(parts)



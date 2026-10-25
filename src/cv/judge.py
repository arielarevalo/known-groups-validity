"""Internalised judge layer (docs/operations/accepted/0005): the async judge call used
by the CoT detector and the PG classification, plus the async OpenRouter client
factory for judge + target follow-ups.

Faithful port of upstream ``call_judge_async`` / ``evaluate_response_async`` /
the OpenRouter branch of ``create_followup_client`` (the OpenAI-native and
Anthropic branches are dead code for this study). The judge is prompted to emit
JSON; parsing is ``json.loads`` with a ``{"raw": content}`` fallback, so the
committed ``judge`` dicts are reproducible from their stored ``raw`` strings.

Goes through the `openai` AsyncOpenAI SDK on purpose: the meter (`cv/meter.py`)
and Weave (docs/operations/accepted/0001) both hook that call site.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any

from cv.selfreport_parse import extract_think_tags

_OPENROUTER_BASE = "https://openrouter.ai/api/v1"

# Per-call ceiling on the judge, the highest-volume paid call in the run. Without it
# there is no per-call spend bound (the provider ceiling is 131k tokens), and cap
# overshoot equals whatever is in flight. The judge REASONS before
# it emits its small JSON object, and those tokens count: the pilot's 10,800 judge
# calls used a median of 1,927 completion tokens, p99.9 = 4,111, and one runaway hit
# 65,536. 8,192 covers 99.99% of them and bounds a runaway at about $0.007. (A first
# cap of 1,024 broke every call; the pre-launch gate caught it on 2026-09-02.)
JUDGE_MAX_TOKENS = 8192


def make_async_openrouter_client(api_key: str):
    """Async OpenAI-compatible client at OpenRouter (judge + target follow-ups)."""
    from openai import AsyncOpenAI

    return AsyncOpenAI(api_key=api_key, base_url=_OPENROUTER_BASE)


def _route_bound(model: str) -> dict[str, Any]:
    """Bind the judge's route to the price table: OpenRouter serves only endpoints at
    or below `provider.max_price`, so the largest paid line of the run cannot be
    billed above what the meter records."""
    from cv import config

    mp = config.provider_max_price(model)
    return {"provider": {"max_price": mp}} if mp else {}


async def call_judge_async(
    client, model: str, formatted_prompt: str, max_retries: int = 5
) -> dict[str, Any]:
    """Call the judge with a pre-formatted prompt; return ``{"judgment", "raw"}``.

    ``judgment`` is the parsed JSON, or ``{"raw": content}`` if the judge did not
    return valid JSON. Retries transient errors with exponential backoff.
    """
    for attempt in range(max_retries):
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": formatted_prompt}],
                max_tokens=JUDGE_MAX_TOKENS,
                extra_body=_route_bound(model),
            )
            content = (resp.choices[0].message.content or "").strip()
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                parsed = {"raw": content}
            return {"judgment": parsed, "raw": content}
        except Exception as e:  # noqa: BLE001
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
                print(f"  [judge] call failed ({attempt + 1}/{max_retries}): {type(e).__name__}")
                continue
            raise


async def evaluate_response_async(
    client,
    model: str,
    prompt: str,
    response: str,
    judge_template: str,
    max_retries: int = 5,
) -> dict[str, Any]:
    """CoT detector: split the target's ``<think>`` trace, fill the judge template
    with ``{task, reasoning, answer}``, and judge it. Returns ``{"judgment", "raw"}``.
    """
    reasoning, answer = extract_think_tags(response)
    formatted_prompt = judge_template.format(task=prompt or "", reasoning=reasoning, answer=answer)
    return await call_judge_async(client, model, formatted_prompt, max_retries=max_retries)

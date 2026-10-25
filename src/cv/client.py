"""Internalised OpenRouter target client + parallel runner (docs/operations/accepted/0005).

A faithful, de-cluttered port of the only upstream serving path we use
(`core.target.OpenRouterApiClient` + `utils.query_parallel`); the DeepSeek /
Anthropic / OpenAI-native clients are dead code for this study and are not
carried over. The wire behaviour is preserved exactly so reruns match the
committed pilot: same request kwargs, native reasoning at ``effort=high`` when
not prefilling, and the ``<think>{reasoning}</think>{content}`` envelope the CoT
detector parses.

The client goes through the `openai` SDK's ``chat.completions.create`` on purpose:
that is the call site `cv/meter.py` monkeypatches for the budget meter, and the
same hook Weave will auto-instrument (docs/operations/accepted/0001).
"""
from __future__ import annotations

import asyncio
from typing import Any, Callable

from cv import config, meter


def make_openrouter_client(api_key: str):
    """OpenAI-compatible client pointed at OpenRouter (the meter-patched path)."""
    from openai import OpenAI

    return OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")


class OpenRouterTarget:
    """A single target model over OpenRouter.

    ``temperature=None`` omits the field (for reasoning models that reject it).
    ``reasoning_effort`` is applied only when not prefilling; with a prefill we
    let the model continue from the seeded ``<think>`` text instead.
    """

    def __init__(
        self,
        model: str,
        system_prompt: str,
        *,
        api_key: str,
        temperature: float | None = 0.3,
        max_tokens: int = 8000,
        reasoning_effort: str = "high",
    ) -> None:
        self.model = model
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.client = make_openrouter_client(api_key)

    def query(self, prompt: str, prefill: str | None = None) -> str:
        try:
            messages: list[dict[str, str]] = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ]
            if prefill:
                messages.append({"role": "assistant", "content": prefill})

            kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "max_tokens": self.max_tokens,
                "stream": False,
            }
            extra: dict[str, Any] = {}
            if not prefill:
                extra["reasoning"] = {"effort": self.reasoning_effort}
            mp = config.provider_max_price(self.model)
            if mp:
                extra["provider"] = {"max_price": mp}  # the ledger's rate is the route's ceiling
            if extra:
                kwargs["extra_body"] = extra
            if self.temperature is not None:
                kwargs["temperature"] = self.temperature

            resp = self.client.chat.completions.create(**kwargs)
            choice = resp.choices[0] if resp.choices else None
            content = (choice.message.content or "") if choice else ""

            if prefill:
                return (prefill + content).strip()
            reasoning = getattr(choice.message, "reasoning", "") if choice else ""
            return f"<think>{reasoning}</think>{content}".strip()
        except Exception as e:  # noqa: BLE001 — upstream returns the error as the response
            return f"ERROR calling OpenRouter API: {e}"


async def query_parallel(
    items: list[tuple[Any, str, dict[str, Any]]],
    result_fn: Callable[[str, dict[str, Any], int], dict[str, Any]],
    max_concurrent: int = 10,
    prefill: str | None = None,
) -> list[dict[str, Any]]:
    """Run ``client.query`` over ``items`` concurrently, preserving input order.

    Each item is ``(client, prompt, metadata)``; ``result_fn(response, metadata,
    sample_idx)`` shapes each result. Exceptions drop that item (logged), matching
    upstream's resilient-batch behaviour.
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def query_one(idx, client, prompt, metadata):
        async with semaphore:
            response = await asyncio.to_thread(client.query, prompt, prefill)
            sample_idx = metadata.get("sample_idx", 1)
            return idx, result_fn(response, metadata, sample_idx)

    tasks = [query_one(i, c, p, m) for i, (c, p, m) in enumerate(items)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    valid = [r for r in results if not isinstance(r, Exception)]
    for r in results:
        if isinstance(r, Exception):
            print(f"  query_parallel error: {r}")
    valid.sort(key=lambda x: x[0])
    return [r[1] for r in valid]


def _provider_name(error_message: str) -> str:
    """Pull the provider name out of an OpenRouter content-moderation error."""
    import re

    m = re.search(r"'provider_name'\s*:\s*'([^']+)'", error_message) or re.search(
        r'"provider_name"\s*:\s*"([^"]+)"', error_message
    )
    return m.group(1) if m else ""


async def query_openrouter_async(
    client,
    model: str,
    system_prompt: str,
    messages: list[dict[str, str]],
    **kwargs,
) -> str:
    """Async multi-turn OpenRouter target call, text only. Faithful port of upstream
    ``query_openrouter_async`` (the golden-tested envelope). See
    ``query_openrouter_async_full`` for the structured form the backends use."""
    return (await query_openrouter_async_full(client, model, system_prompt, messages, **kwargs)).text


async def query_openrouter_async_full(
    client,
    model: str,
    system_prompt: str,
    messages: list[dict[str, str]],
    *,
    max_tokens: int = 2048,
    temperature: float | None = 0.3,
    reasoning_effort: str = "high",
    max_retries: int = 3,
    max_provider_fallbacks: int = 10,
    provider_quantization: str = "",
) -> "GenResult":
    """Async multi-turn OpenRouter target call (used by the rollout and the SR/TP
    follow-ups), returning a `GenResult`.

    Wraps native reasoning in ``<think>`` tags, retries empty responses, and on a
    content-moderation block falls back across providers on a separate budget, so
    one provider refusing a harmful task does not lose the rollout. Faithful port
    of upstream ``query_openrouter_async`` (note: ``max_tokens`` defaults to 2048
    for follow-ups, as upstream did, vs the rollout's larger budget).

    Beyond the text it records the finish reason (the only reliable truncation
    signal), the provider that served the call, and the raw reasoning and content
    fields, so a later re-encode reads the model's real output rather than the
    synthesized envelope. A generation that stops inside
    the reasoning (no content, finish "length") is stored with an UNCLOSED
    ``<think>``: the trace is never presented as an answer.
    """
    from cv.serving.base import GenResult

    full_messages = [{"role": "system", "content": system_prompt}] + list(messages)
    blocked: list[str] = []

    def _kwargs() -> dict:
        extra: dict[str, Any] = {"reasoning": {"effort": reasoning_effort}}
        provider: dict[str, Any] = {}
        if blocked:
            provider["ignore"] = blocked
        # FP8 parity (operations 0013): only route to providers serving this precision,
        # while still allowing fallback among them (coexists with the moderation `ignore`).
        if provider_quantization:
            provider["quantizations"] = [provider_quantization]
        # The price table is the route's ceiling: OpenRouter serves only endpoints at or
        # below `max_price`, so the metered rate bounds the billed one.
        mp = config.provider_max_price(model)
        if mp:
            provider["max_price"] = mp
        if provider:
            extra["provider"] = provider
        kw: dict[str, Any] = {
            "model": model,
            "messages": full_messages,
            "max_tokens": max_tokens,
            "extra_body": extra,
        }
        if temperature is not None:
            kw["temperature"] = temperature
        return kw

    error_retries = 0
    provider_retries = 0
    while error_retries < max_retries:
        try:
            resp = await client.chat.completions.create(**_kwargs())
            choice = resp.choices[0] if resp.choices else None
            if not choice:
                error_retries += 1
                if error_retries < max_retries:
                    await asyncio.sleep(2 ** (error_retries - 1))
                    continue
                return GenResult(finish_reason="error", error="no choices")
            content = choice.message.content or ""
            reasoning = getattr(choice.message, "reasoning", "") or ""
            finish = str(getattr(choice, "finish_reason", "") or "")
            provider = str(getattr(resp, "provider", "") or "")
            if not content and not reasoning:
                error_retries += 1
                if error_retries < max_retries:
                    await asyncio.sleep(2 ** (error_retries - 1))
                    continue
                return GenResult(finish_reason=finish or "error", provider=provider,
                                 reasoning=reasoning, content=content, error="empty completion")
            if reasoning and not content and finish == "length":
                text = f"<think>{reasoning}"  # stopped inside the trace: leave it unclosed
            elif reasoning:
                text = f"<think>{reasoning}</think>{content}".strip()
            else:
                text = content.strip()
            return GenResult(text=text, finish_reason=finish, provider=provider,
                             reasoning=reasoning, content=content)
        except meter.BudgetExceeded:
            raise  # a pre-call refusal from the meter is a stop, not a transient error
        except Exception as e:  # noqa: BLE001
            es = str(e).lower()
            if any(s in es for s in ("inappropriate content", "data_inspection_failed", "content_filter")):
                prov = _provider_name(str(e))
                if prov and prov not in blocked:
                    blocked.append(prov)
                    provider_retries += 1
                    if provider_retries < max_provider_fallbacks:
                        continue
                return GenResult(finish_reason="error", error=f"moderation: {type(e).__name__}")
            if "all providers have been ignored" in es:
                return GenResult(finish_reason="error", error="moderation: all providers ignored")
            error_retries += 1
            if error_retries < max_retries:
                await asyncio.sleep(2 ** (error_retries - 1))
                continue
            raise
    return GenResult(finish_reason="error", error="retries exhausted")

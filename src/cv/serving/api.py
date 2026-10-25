"""OpenAI-compatible target backend (OpenRouter by default).

Wraps the golden-tested `cv.client.query_openrouter_async` unchanged: this is the
dev / CI / offline-fallback target backend and the path the judge always uses.
Because it reuses the internalised client, the `<think>…</think>…` envelope, native
reasoning at `effort=high`, content-moderation provider fallback, and retry/backoff
are exactly the upstream behaviour (docs/operations/accepted/0005).

`agenerate` fans the batch out concurrently under a semaphore — the same shape the
SR/TP detectors used directly before the serving abstraction, so the OpenRouter wire
behaviour is unchanged. (Targets are token-priced here, so the cost meter, which
hooks the OpenAI SDK, still meters them; on the Modal backend targets become
GPU-seconds and correctly fall off the token meter.)

A non-default `base_url` lets this same backend point at any OpenAI-compatible
server; we do not use that for Modal (targets there are a function, not a served
endpoint — see docs/operations/accepted/0007), but it keeps the door open.
"""
from __future__ import annotations

import asyncio
import os
from typing import List, Optional

from cv import meter
from cv.client import query_openrouter_async_full
from cv.serving.base import GenRequest, GenResult

_OPENROUTER_BASE = "https://openrouter.ai/api/v1"


class ApiBackend:
    name = "openrouter"

    def __init__(
        self,
        model_id: str,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_concurrent: int = 10,
        provider_quantization: str = "",
    ) -> None:
        from openai import AsyncOpenAI

        self.model_id = model_id
        self.max_concurrent = max_concurrent
        # FP8 parity (operations 0013): constrain OpenRouter to providers serving this
        # precision. Empty = unconstrained (current behaviour).
        self.provider_quantization = (provider_quantization or "").strip().lower()
        key = api_key or os.getenv("OPENROUTER_API_KEY", "")
        self.client = AsyncOpenAI(api_key=key or "missing", base_url=base_url or _OPENROUTER_BASE)

    def preflight(self) -> str:
        """Nothing to prove without a paid call; the pre-spend gate (cv-validate)
        makes those. Reports the routing so the operator sees the fp8 filter."""
        q = self.provider_quantization or "unconstrained"
        return f"openrouter model {self.model_id!r}, provider quantization {q}"

    async def agenerate(self, requests: List[GenRequest]) -> List[GenResult]:
        sem = asyncio.Semaphore(self.max_concurrent)

        async def one(r: GenRequest) -> GenResult:
            msgs = list(r.messages)
            if r.prefill:
                msgs = msgs + [{"role": "assistant", "content": r.prefill}]
            async with sem:
                try:
                    return await query_openrouter_async_full(
                        self.client,
                        self.model_id,
                        r.system_prompt,
                        msgs,
                        max_tokens=r.max_tokens,
                        temperature=r.temperature,
                        reasoning_effort=r.reasoning_effort,
                        provider_quantization=self.provider_quantization,
                    )
                except meter.BudgetExceeded:
                    raise  # the cap is a stop, never a degraded item
                except Exception as e:  # noqa: BLE001 — resilient batch: degrade, don't sink
                    print(f"  [api] generate failed: {type(e).__name__}: {e}")
                    return GenResult(finish_reason="error", error=f"{type(e).__name__}: {e}"[:200])

        return await asyncio.gather(*(one(r) for r in requests))

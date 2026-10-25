"""Step 0: assert every assumption the pipeline depends on, per model, cheaply.

A few real calls per model (cents total) through the SAME code paths the run uses:
the configured serving backend for each target (OpenRouter under the profile's
provider filter, or the deployed Modal app), and the judge on the API. Hard gate:
if a model has no reasoning trace, CoT monitoring is void, so we stop before
spending budget. This module runs the per-model checks (`validate_model`); the run
narrative that loops models, prints the green/red table, and sets the exit code is
the `cv-validate` console script (src/validate.py).

Why the backend, not a bare OpenRouter client: the pilot's
runbook records a smoke that went green end-to-end with NO reasoning traces,
because a different code path had the thinking flag. A gate proves nothing unless
it walks the production path, with the production routing.
"""
from __future__ import annotations

import asyncio
import os

import cv.bootstrap  # noqa: F401
from cv import config, failures, meter, parse, registry
from cv.selfreport_parse import extract_think_tags, parse_selfreport_awareness
from cv.serving import GenRequest, as_result, get_backend

from openai import AsyncOpenAI

Check = tuple[str, bool, bool, str]  # (name, passed, hard_gate, detail)


def _safety_prompt() -> str:
    """One real safety-valence baseline prompt to test moderation, from the
    internalised HF dataset (no submodule render)."""
    from cv.dataset import load_prompts

    for p in load_prompts():
        if p.config_name == "baseline" and p.valence == "safety":
            return p.prompt
    return "Write a short note to a colleague."


async def _one(backend, system_prompt: str, messages: list[dict], temperature, max_tokens: int):
    (res,) = await backend.agenerate([GenRequest(system_prompt=system_prompt, messages=messages,
                                                 temperature=temperature, max_tokens=max_tokens)])
    return as_result(res)


async def validate_model(model_key: str) -> list[Check]:
    """Return list of (check, passed, hard_gate, detail)."""
    from cv.detect import build_base_conversation
    from cv.judge import evaluate_response_async

    model_id, temp = config.TARGET_MODELS[model_key]
    results: list[Check] = []
    system_prompt = (config.PROMPTS_DIR / "chat_prompt_realistic.txt").read_text(encoding="utf-8").strip()
    backend = get_backend(model_key)

    # 0) the backend is reachable before anything is spent (Modal: the app is deployed).
    # In a thread: the Modal preflight is a blocking call, and this function is async.
    check = getattr(backend, "preflight", None)
    line = await asyncio.to_thread(check) if callable(check) else backend.name
    results.append(("backend reachable", True, True, line))

    with meter.stage("validate"):
        # 1) reasoning + answer present on the ROLLOUT path -- HARD GATE on reasoning
        res = await _one(backend, system_prompt,
                         [{"role": "user", "content": "Briefly outline how you'd plan a 3-day trip to Rome."}],
                         temp, config.max_tokens())
        reasoning, answer = extract_think_tags(res.text)
        failed = failures.classify(res)
        results.append(("reasoning present", bool(reasoning) and failed is None, True,
                        f"{len(reasoning)} chars" if reasoning else (failed or "EMPTY")))
        results.append(("answer present", bool(answer) and failed is None, False,
                        f"{len(answer)} chars, finish={res.finish_reason or '?'}" if answer else (failed or "EMPTY")))
        if res.thinking is not None:
            results.append(("thinking template on", bool(res.thinking), True, f"thinking={res.thinking}"))
        # 1b) the provider filter took effect (API targets only; informational)
        if backend.name == "openrouter":
            want = config.provider_quantization() or "unconstrained"
            results.append(("provider routing", bool(res.provider) or want == "unconstrained", False,
                            f"filter={want}, served by {res.provider or 'unreported'}"))

        # 2) safety valence not auto-blocked (same path)
        safety = await _one(backend, system_prompt, [{"role": "user", "content": _safety_prompt()}],
                            temp, config.max_tokens())
        sfail = failures.classify(safety)
        results.append(("safety not blocked", sfail is None, False, sfail or "ok"))

        # 3) SR label parses on the FOLLOW-UP path (same backend, production prompts, profile cap)
        base, _ = build_base_conversation("What's the capital of France?", res.text or "Paris.")
        aware_prompt = registry.prompt_path("sr", "awareness").read_text(encoding="utf-8").strip()
        sr_system = registry.prompt_path("sr", "system").read_text(encoding="utf-8").strip()
        sr = await _one(backend, sr_system, base + [{"role": "user", "content": aware_prompt}],
                        temp, config.followup_max_tokens())
        sr_concl = parse_selfreport_awareness(sr.text)[0]
        results.append(("SR label parses", sr_concl in ("yes", "no"), False,
                        f"conclusion={sr_concl!r}, finish={sr.finish_reason or '?'}"))

        # 4) judge JSON via OpenRouter routing (upstream template, design/accepted/0023)
        judge_template = registry.prompt_path("cot", "judge_template").read_text(encoding="utf-8")
        async with AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"], max_retries=3) as jclient:
            jout = await evaluate_response_async(jclient, config.judge_model(),
                                                 "What's 2+2?", res.text or "<think>simple</think>4", judge_template)
        judgment = parse.ensure_judgment(jout.get("judgment", {}), jout.get("raw", ""))
        ma = judgment.get("model_awareness", {}) if isinstance(judgment, dict) else {}
        results.append(("judge JSON valid", isinstance(ma, dict) and "conclusion" in ma, True,
                        f"keys={list(ma)[:4]}" if isinstance(ma, dict) else "no JSON"))

        # 5) logprob availability on the API (best-effort, recorded only)
        lp = "no"
        try:
            async with AsyncOpenAI(api_key=os.getenv("OPENROUTER_API_KEY", ""),
                                   base_url="https://openrouter.ai/api/v1") as lpc:
                r = await lpc.chat.completions.create(model=model_id, messages=[{"role": "user", "content": "Say hi."}],
                                                      max_tokens=5, logprobs=True)
                lp = "yes" if bool(getattr(r.choices[0], "logprobs", None)) else "no"
        except Exception:  # noqa: BLE001
            lp = "no"
        results.append(("logprobs available", True, False, lp))  # informational, never fails

    # 6) usage recorded by the meter (API tokens and GPU seconds both land here)
    results.append(("usage metered", meter.spent() > 0, False, f"${meter.spent():.4f} cumulative"))
    return results

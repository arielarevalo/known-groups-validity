"""Detector honesty (wave 1.5) and generation-time capture (wave 1.6).

A generation without a visible answer is a failure everywhere it can appear: the
parsers return None instead of a trace-derived verdict, the classifier names the
reason, the rollout ledgers it, the merge gate invalidates any that slipped
through, and the backends carry the finish reason that makes truncation visible.
"""
import asyncio
import json
from types import SimpleNamespace

import pytest

from cv import config, failures, merge, meter, rollout
from cv.client import query_openrouter_async_full
from cv.dataset import BenchPrompt
from cv.detect import format_conversation_for_thirdperson
from cv.selfreport_parse import (
    extract_think_tags, has_visible_answer, parse_selfreport_awareness, parse_selfreport_hypothetical,
)
from cv.serving import GenResult, as_result
from cv.serving.local.backend import _to_result

TRACE_ONLY = '<think>The user is asking me to categorize the previous interaction into one of three categories: "Deployment", "Safety Evaluation", or "Capability Ev'


@pytest.fixture(autouse=True)
def _restore():
    yield
    config.load()


# --- parsers: no verdict from a trace ------------------------------------------
def test_unclosed_trace_is_reasoning_not_answer():
    assert extract_think_tags(TRACE_ONLY)[1] == ""
    assert not has_visible_answer(TRACE_ONLY) and not has_visible_answer("<think>r</think>")
    assert has_visible_answer("<think>r</think>Deployment") and has_visible_answer("plain")


def test_parsers_return_none_on_a_missing_answer():
    # The pilot's exact artifact: the scanner used to read "safety evaluation" off this trace.
    assert parse_selfreport_awareness(TRACE_ONLY) == (None, None, "")
    assert parse_selfreport_awareness("<think>only thoughts</think>") == (None, None, "")
    assert parse_selfreport_hypothetical(TRACE_ONLY) == (None, "")
    assert parse_selfreport_awareness("<think>hmm</think>Deployment")[:2] == ("no", "N/A")  # unchanged


def test_tp_never_sees_an_unclosed_trace():
    conv = format_conversation_for_thirdperson(task="t", response="<think>secret deliberation")
    assert "secret deliberation" not in conv


# --- the classifier names every failure shape ----------------------------------
@pytest.mark.parametrize("res,reason", [
    (GenResult(text=""), failures.EMPTY),
    (GenResult(text="ERROR calling x"), failures.ERROR),
    (GenResult(text="<think>r</think>", finish_reason="stop"), failures.NO_ANSWER),
    (GenResult(text="<think>r</think>", finish_reason="length"), failures.TRUNCATED),
    (GenResult(text="<think>r", finish_reason="length"), failures.TRUNCATED),
    (GenResult(text="<think>r"), failures.TRUNCATED),
    (GenResult(text="answer", thinking=False), failures.NO_THINKING),
    (GenResult(text="<think>r</think>answer", finish_reason="length"), None),  # cut answer still an answer
    (GenResult(text="answer"), None),
    ("plain string", None),
])
def test_classify(res, reason):
    assert failures.classify(res) == reason


def test_classify_sample_reads_stored_metadata():
    assert failures.classify_sample({"response": "<think>r</think>", "finish_reason": "length"}) == failures.TRUNCATED
    assert failures.classify_sample({"response": "answer", "thinking": False}) == failures.NO_THINKING
    assert failures.classify_sample({"response": "<think>r</think>ok"}) is None


# --- backends carry the finish reason and the raw fields -----------------------
class _Completions:
    def __init__(self, *, content, reasoning, finish, provider="DeepInfra"):
        self.resp = SimpleNamespace(
            provider=provider,
            choices=[SimpleNamespace(finish_reason=finish,
                                     message=SimpleNamespace(content=content, reasoning=reasoning))])

    async def create(self, **kwargs):
        return self.resp


def _client(**kw):
    return SimpleNamespace(chat=SimpleNamespace(completions=_Completions(**kw)))


def test_api_client_keeps_raw_fields_and_marks_truncation():
    ok = asyncio.run(query_openrouter_async_full(_client(content="A", reasoning="R", finish="stop"),
                                                 "m", "sys", [{"role": "user", "content": "q"}]))
    assert ok == GenResult(text="<think>R</think>A", finish_reason="stop", reasoning="R", content="A",
                           provider="DeepInfra")
    cut = asyncio.run(query_openrouter_async_full(_client(content="", reasoning="R only", finish="length"),
                                                  "m", "sys", [{"role": "user", "content": "q"}]))
    assert cut.text == "<think>R only" and cut.truncated and failures.classify(cut) == failures.TRUNCATED


def test_modal_result_marks_a_template_consumed_trace():
    # Qwen on vLLM: the template emits the opening tag, so a run cut inside the trace is bare text.
    r = _to_result({"text": "just reasoning", "finish_reason": "length", "thinking": True})
    assert r.text == "<think>just reasoning" and failures.classify(r) == failures.TRUNCATED
    r = _to_result({"text": "reasoning</think>answer", "finish_reason": "stop", "thinking": True})
    assert extract_think_tags(r.text) == ("reasoning", "answer") and failures.classify(r) is None
    assert _to_result("old string shape") == GenResult(text="old string shape")   # older deploy
    assert failures.classify(_to_result({"text": "x", "finish_reason": "stop", "thinking": False})) == failures.NO_THINKING


# --- the rollout persists the capture and ledgers truncation ---------------------
class _Backend:
    name = "fake"

    def __init__(self, results):
        self.results = results

    async def agenerate(self, reqs):
        return list(self.results)


def test_rollout_persists_capture_fields_and_ledgers_truncation(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "cv_out_dir", lambda: tmp_path)
    monkeypatch.setattr(config, "metrics_path", lambda: tmp_path / "_metrics.jsonl")
    monkeypatch.setattr(meter, "_spent_usd", 0.0)
    monkeypatch.setattr(meter, "_budget_hit", False)
    config.load(["samples_per_config=1", "budget_usd=100.0"])
    prompts = [BenchPrompt(id=t, task_id=t, task_name=t, valence="safety", config_name="baseline",
                           factor_config={}, prompt=f"p {t}", description="") for t in ("t1", "t2")]
    monkeypatch.setattr(rollout, "load_prompts", lambda: prompts)
    monkeypatch.setattr(rollout, "get_backend", lambda key: _Backend([
        GenResult(text="<think>R</think>A", finish_reason="stop", reasoning="R", content="A", provider="SiliconFlow"),
        GenResult(text="<think>R only", finish_reason="length", reasoning="R only", content=""),
    ]))
    list(rollout.run("gemma-4-31b"))
    (s,) = json.loads((tmp_path / "rollouts" / "responses_gemma-4-31b.json").read_text())["samples"]
    assert s["id"] == "t1_baseline_sample1" and s["finish_reason"] == "stop"
    assert s["provider"] == "SiliconFlow" and s["reasoning_raw"] == "R" and s["content_raw"] == "A"
    assert failures.load("rollout", "gemma-4-31b")[0]["reason"] == failures.TRUNCATED


# --- the merge gate catches what slipped through --------------------------------
def _sr_rec(aware_raw, hypo_raw, a_concl, h_concl):
    judge = {"model_awareness": {"conclusion": a_concl, "awareness_type": "N/A"},
             "hypothetical_recognition": {"conclusion": h_concl}}
    return {"raw": {"awareness": aware_raw, "hypothetical": hypo_raw}}, judge


def test_merge_gate_rejects_undecided_and_trace_only_rows():
    assert merge._detector_valid("sr", *_sr_rec("<think>r</think>Deployment", "<think>r</think>Real", "no", "no"))
    assert not merge._detector_valid("sr", *_sr_rec(TRACE_ONLY, "<think>r</think>Real", "no", "no"))  # H19
    assert not merge._detector_valid("tp", *_sr_rec("<think>r</think>Deployment", TRACE_ONLY, "no", "no"))
    assert not merge._detector_valid("sr", *_sr_rec("<think>r</think>Deployment", "<think>r</think>Real", None, "no"))
    assert merge._detector_valid("cot", {}, {"model_awareness": {"conclusion": "no"}})


def test_merge_invalidates_legacy_pilot_rollouts(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "cv_out_dir", lambda: tmp_path)
    d = tmp_path / "rollouts"
    d.mkdir()
    (d / "responses_qwen3.6-27b.json").write_text(json.dumps({"samples": [
        {"id": "good", "response": "<think>r</think>answer"},
        {"id": "qwen_unclosed", "response": "<think>Here's a thinking process: 1. Analyze"},   # H16 (16 rows)
        {"id": "gemma_trace_only", "response": "<think>...I got 99 problems</think>"},          # M13 (27 rows)
        {"id": "no_thinking", "response": "answer", "thinking": False},                          # M18
    ]}))
    assert merge._invalid_rollout_ids() == {"qwen_unclosed", "gemma_trace_only", "no_thinking"}


# --- the follow-up cap comes from the profile, the judge-2 template from the registry --
def test_followup_cap_and_judge2_template(repo_root):
    from cv import registry

    assert config.followup_max_tokens() == 2048
    src = (repo_root / "src/cv/judge2.py").read_text()
    assert "judgePromptEvidenceBoth_conf.txt" not in src
    assert 'registry.prompt_path("cot", "judge_template")' in src
    assert registry.prompt_path("cot", "judge_template").name == "judgePromptEvidenceBoth.txt"

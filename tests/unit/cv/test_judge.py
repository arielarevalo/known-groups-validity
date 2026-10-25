"""The judge call (cv.judge): the highest-volume paid call carries a per-call token
cap."""
import asyncio
from types import SimpleNamespace

from cv import judge


class _FakeCompletions:
    def __init__(self):
        self.kwargs = None

    async def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok": 1}'))])


def test_judge_call_caps_completion_tokens():
    comp = _FakeCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=comp))
    out = asyncio.run(judge.call_judge_async(client, "any/judge", "prompt", max_retries=1))
    assert out["judgment"] == {"ok": 1}
    assert comp.kwargs["max_tokens"] == judge.JUDGE_MAX_TOKENS

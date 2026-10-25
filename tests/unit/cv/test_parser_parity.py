"""Free-standing golden test for the internalised SR/TP parsers: feed the
committed fixture's raw detector responses through the ported parser and assert
the decisions match the recorded ``judge.model_awareness`` exactly. Reads the
fixture copy, not the live results/ dir.

Rows whose follow-up carries no visible answer are skipped on purpose: the pilot
scored those by keyword-scanning the reasoning trace, and the
parser now returns None for them instead. Their golden value is the bug.
"""
import json

from cv.selfreport_parse import has_visible_answer


def _records(fixtures_dir, det: str) -> list[dict]:
    text = (fixtures_dir / f"{det}_gemma_pilot.jsonl").read_text()
    return [json.loads(l) for l in text.splitlines() if l.strip()]


def _check_awareness(fixtures_dir, det: str) -> list:
    from cv.selfreport_parse import parse_selfreport_awareness

    bad = []
    for r in _records(fixtures_dir, det):
        gm = r.get("judge", {}).get("model_awareness", {})
        exp_c = str(gm.get("conclusion", "")).strip().lower()
        exp_t = str(gm.get("awareness_type", "")).strip()
        raw = r.get("raw")
        aw = (raw.get("awareness", "") if isinstance(raw, dict) else raw) or ""
        if not aw or not exp_c or not has_visible_answer(aw):
            continue
        c, t, _ = parse_selfreport_awareness(aw)
        if c != exp_c or t != exp_t:
            bad.append((r["id"], (c, t), (exp_c, exp_t)))
    return bad


def test_sr_awareness_parser_matches_fixture(fixtures_dir):
    assert not _check_awareness(fixtures_dir, "sr")


def test_tp_awareness_parser_matches_fixture(fixtures_dir):
    assert not _check_awareness(fixtures_dir, "tp")


def test_hypothetical_parser_runs_on_fixture(fixtures_dir):
    from cv.selfreport_parse import parse_selfreport_hypothetical

    for det in ("sr", "tp"):
        for r in _records(fixtures_dir, det):
            raw = r.get("raw")
            h = (raw.get("hypothetical", "") if isinstance(raw, dict) else "") or ""
            if not h or not has_visible_answer(h):
                continue
            assert parse_selfreport_hypothetical(h)[0] in ("yes", "no")

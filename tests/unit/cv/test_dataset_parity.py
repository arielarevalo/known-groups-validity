"""ADR 0005 fidelity gate, free-standing: the HF-sourced + unflattened prompts
must reproduce, byte-for-byte, the prompts in the committed fixture (a subset of
the pilot rollout, originally rendered through the upstream submodule). The HF
dataset is the unit under test; the *expected* prompts come from the in-repo
fixture, never the live results/ dir.
"""
import json

from cv.dataset import load_prompts


def test_hf_prompts_match_fixture(fixtures_dir):
    bench = {(p.task_id, p.config_name): p for p in load_prompts()}
    samples = json.loads((fixtures_dir / "rollout_gemma_pilot.json").read_text())["samples"]
    checked, mismatches = 0, []
    for s in samples:
        bp = bench.get((s["task_id"], s["config_name"]))
        if bp is None:
            mismatches.append(f"missing HF row {s['task_id']}/{s['config_name']}")
            continue
        checked += 1
        if bp.prompt != s["prompt"]:
            mismatches.append(f"prompt mismatch {s['task_id']}/{s['config_name']}")
        if bp.factor_config != s["factor_config"]:
            mismatches.append(f"factor_config mismatch {s['task_id']}/{s['config_name']}")
    assert checked > 0, "no fixture samples checked"
    assert not mismatches, mismatches[:5]

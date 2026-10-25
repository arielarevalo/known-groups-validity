"""Experiment B (wave 2.1, design 0029 D2): the stacked-cues sampler draws the
budgeted subset, seeded and recorded; the rollout enumerates the full scope from
it; the cue labels know a stack; the analysis pairs every contrast within its own
items and reads the dose-response by cue count."""
import itertools
import json

import numpy as np
import pytest

from cv import analysis, config, factors, meter, rollout, stacking
from cv.dataset import BASELINE_CONFIG, FACTORS, BenchPrompt, config_name


@pytest.fixture(autouse=True)
def _restore(monkeypatch):
    monkeypatch.delenv("CV_PROFILE", raising=False)
    yield
    config.load()


def _cube(n_tasks=100):
    """The released dataset's shape: every task x valence x 2^8 factor subset."""
    out = []
    for t in range(n_tasks):
        name = f"task{t:03d}"
        for val in ("safety", "capability"):
            tid = f"{name}_{val}"
            for r in range(len(FACTORS) + 1):
                for on in itertools.combinations(FACTORS, r):
                    fc = {f: ("flipped" if f in on else BASELINE_CONFIG[f]) for f in FACTORS}
                    cn = config_name(fc)
                    out.append(BenchPrompt(id=f"{tid}__{cn}", task_id=tid, task_name=name, valence=val,
                                           config_name=cn, factor_config=fc, prompt=f"p {tid} {cn}",
                                           description=""))
    return out


# --- the sampler ---------------------------------------------------------------------
def test_plan_draws_the_budgeted_subset():
    cube = _cube()
    plan = stacking.build_plan(cube, tasks_per_valence=25, max_per_k=10, seed=0)
    assert len(plan.task_names) == 25 and len(plan.draws) == 50          # both valences of each task
    for tid, cfgs in plan.draws.items():
        assert len(cfgs) == 48 and len(set(cfgs)) == 48
        by_k = {}
        for c in cfgs:
            assert "F8" not in stacking.active_factors(c)                 # F8 stands alone
            by_k[stacking.n_active(c)] = by_k.get(stacking.n_active(c), 0) + 1
        assert by_k == {2: 10, 3: 10, 4: 10, 5: 10, 6: 7, 7: 1}          # min(10, C(7,k)), k=2..7
    kept = stacking.apply_plan(cube, plan)
    assert len(kept) == 200 * 9 + 50 * 48                                # A + B = 4,200 prompts
    assert stacking.missing_from(cube, plan) == []


def test_plan_is_seeded_and_recorded():
    cube = _cube(20)
    a = stacking.build_plan(cube, tasks_per_valence=5, max_per_k=10, seed=7)
    b = stacking.build_plan(cube, tasks_per_valence=5, max_per_k=10, seed=7)
    c = stacking.build_plan(cube, tasks_per_valence=5, max_per_k=10, seed=8)
    assert a == b and (a.task_names != c.task_names or a.draws != c.draws)
    rec = json.loads(json.dumps(a.to_record()))
    assert rec["seed"] == 7 and rec["factors"] == list(stacking.STACK_FACTORS) and rec["held_out"] == ["F8"]
    assert set(rec["draws"]) == set(a.draws) and all(rec["draws"][t] == list(a.draws[t]) for t in a.draws)


def test_draws_are_the_dataset_labels_and_k_levels_are_independent():
    assert stacking.draw_combos("x", 7, 10, 0) == ["F1+F2+F3+F4+F5+F6+F7"]
    assert stacking.draw_combos("x", 6, 10, 0) == sorted("+".join(c) for c in itertools.combinations(stacking.STACK_FACTORS, 6))
    capped = stacking.draw_combos("x", 3, 5, 0)
    assert len(capped) == 5 and set(capped) <= {"+".join(c) for c in itertools.combinations(stacking.STACK_FACTORS, 3)}
    assert stacking.draw_combos("x", 2, 10, 0) != stacking.draw_combos("y", 2, 10, 0)             # per item
    assert config_name({**BASELINE_CONFIG, "F1": "x", "F3": "y"}) == "F1+F3"                     # label order


def test_missing_pairs_are_reported():
    cube = _cube(4)
    plan = stacking.build_plan(cube, tasks_per_valence=2, max_per_k=10, seed=0)
    short = [p for p in cube if p.config_name != "F1+F2+F3+F4+F5+F6+F7"]
    assert len(stacking.missing_from(short, plan)) == 4                     # one per subset item


# --- the rollout enumerates the full scope ------------------------------------------
def _fake_backend(calls):
    class _B:
        name = "fake"

        async def agenerate(self, reqs):
            calls.append(len(reqs))
            return ["<think>r</think>ok"] * len(reqs)

    return _B()


def test_rollout_enumerates_a_plus_b(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "cv_out_dir", lambda: tmp_path)
    monkeypatch.setattr(meter, "_spent_usd", 0.0)
    monkeypatch.setattr(meter, "_budget_hit", False)
    cube = _cube()
    monkeypatch.setattr(rollout, "load_prompts", lambda **kw: cube if not kw.get("single_factor_only", True)
                        else [p for p in cube if not stacking.is_stacked(p.config_name)])
    calls = []
    monkeypatch.setattr(rollout, "get_backend", lambda key: _fake_backend(calls))
    config.load(["stack_tasks_per_valence=25", "expected_rollouts_per_model=12600",
                 "serving_batch_size=20000", "budget_usd=100.0"])
    list(rollout.run("gemma-4-31b"))
    assert calls == [12600]
    saved = json.loads((tmp_path / "rollouts" / "responses_gemma-4-31b.json").read_text())
    assert len(saved["samples"]) == 12600 and 9 < saved["n_configs"] <= 9 + 120   # singles + the stacks drawn
    rec = saved["stacking"]
    assert rec["seed"] == 0 and len(rec["task_names"]) == 25 and len(rec["draws"]) == 50
    # Experiment A only: the same loader, the pilot's scope, no plan recorded
    calls.clear()
    monkeypatch.setattr(config, "cv_out_dir", lambda: tmp_path / "a")
    config.load(["expected_rollouts_per_model=5400", "serving_batch_size=20000", "budget_usd=100.0"])
    list(rollout.run("gemma-4-31b"))
    assert calls == [5400]
    assert json.loads((tmp_path / "a" / "rollouts" / "responses_gemma-4-31b.json").read_text())["stacking"] is None


def test_full_profile_turns_experiment_b_on(monkeypatch):
    monkeypatch.setenv("CV_PROFILE", "full")
    config.load()
    assert config.stacking() == {"tasks_per_valence": 25, "max_per_k": 10, "seed": 0}
    monkeypatch.setenv("CV_PROFILE", "prefull")
    config.load()
    assert config.stacking() is None


# --- the cue labels -------------------------------------------------------------------
def test_cue_state_knows_a_stack():
    assert factors.cue_state("F1+F3") == factors.STACKED and factors.factor_class("F1+F3") == "stacked"
    assert factors.cue_state("F1+F8") == factors.EXCLUDED                  # F8 in a stack stays out
    assert factors.cue_state("F3") == factors.CLEAN and factors.ground_truth("F1+F3") == 1


# --- the analysis: paired baselines and the dose-response ---------------------------
def _rows(det="cot", seed=0):
    """Four tasks x 2 valences. Two tasks carry stacks. Awareness rises with k."""
    rng = np.random.default_rng(seed)
    rows = []
    for t in range(4):
        name = f"t{t}"
        for val in ("safety", "capability"):
            tid = f"{name}_{val}"
            cfgs = ["baseline", "F1", "F3", "F7", "F8"]
            if t < 2:
                cfgs += ["F1+F3", "F1+F3+F7", "F1+F2+F3+F4+F5+F6+F7", "F1+F8"]
            for cn in cfgs:
                k = stacking.n_active(cn)
                for rep in range(1, 4):
                    p = 0.1 + 0.12 * k
                    rows.append({"id": f"{tid}_{cn}_sample{rep}", "model": "m", "task_id": tid,
                                 "task_name": name, "valence": val, "config_name": cn, "sample_idx": rep,
                                 "detector": det, "valid": True, "aware": bool(rng.random() < p),
                                 "aware_type": "safety", "hypo": False})
    return rows


def test_per_factor_contrasts_are_paired_within_their_items():
    rows = _rows()
    sens = analysis._sensitivity(rows, ["cot"], np.random.default_rng(0))
    per = {f["factor"]: f for f in sens["per_factor"]["cot"]}
    assert per["F1"]["n_neg"] == 8 * 3 and per["F1"]["n_items"] == 8          # all eight items
    assert per["F1+F3"]["n_neg"] == 4 * 3 and per["F1+F3"]["n_items"] == 4    # only the subset's baselines
    assert per["F1+F3"]["factor_class"] == "stacked" and per["F1+F3"]["n_active"] == 2
    assert sens["pooled_cued"]["cot"]["n_neg"] == 8 * 3
    assert sens["pooled_cued"]["cot"]["n_pos"] == 8 * 3 * 3 + 4 * 3 * 3   # F1,F3,F7 everywhere; 3 stacks on the subset
    assert sens["control_f8"]["cot"]["factor"] == "F8" and "d_prime_ci" in per["F1"]


def test_factor_sweep_keeps_to_singles_and_dose_response_reads_the_stacks():
    rows = _rows()
    sweep = analysis._factor_sweep(rows, ["cot"])
    assert sweep["configs"] == ["F1", "F3", "F7", "F8", "baseline"]
    dr = analysis._dose_response(rows, ["cot"], np.random.default_rng(0))
    assert dr["n_items"] == 4 and dr["held_out"] == ["F8"] and dr["ci_unstable"] is True
    by_k = dr["by_detector"]["cot"]["by_k"]
    assert sorted(by_k) == ["0", "1", "2", "3", "7"]
    assert by_k["0"]["n"] == 4 * 3 and by_k["0"]["n_items"] == 4               # the subset's own baseline
    assert by_k["1"]["n_configs"] == 3 and by_k["1"]["n"] == 4 * 3 * 3         # F1, F3, F7; never F8
    assert "d_prime" in by_k["2"] and "d_prime" not in by_k["0"]
    assert by_k["7"]["aware_rate"] > by_k["0"]["aware_rate"]
    slope = dr["by_detector"]["cot"]["slope_per_cue"]
    assert slope is not None and slope > 0 and len(dr["by_detector"]["cot"]["slope_ci"]) == 2
    assert analysis._dose_response([r for r in rows if not stacking.is_stacked(r["config_name"])],
                                   ["cot"], np.random.default_rng(0)) is None

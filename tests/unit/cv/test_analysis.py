"""The estimators of cv.analysis: the count rule
for d', the per-factor intervals, the floor-channel rule, the
cued criterion and the method-family blocks."""
import numpy as np
import pytest

from cv import analysis


def _grid(det, p_by_cfg, n_tasks=100, reps=3, seed=0, cfgs=("baseline", "F1", "F3", "F7", "F8")):
    """n_tasks x 2 valences x configs x reps rows for one detector; awareness is
    Bernoulli with a rate per config."""
    rng = np.random.default_rng(seed)
    rows = []
    for t in range(n_tasks):
        for val in ("safety", "capability"):
            tid = f"t{t}_{val}"
            for cn in cfgs:
                for rep in range(1, reps + 1):
                    rows.append({"id": f"{tid}_{cn}_sample{rep}", "model": "m", "task_id": tid,
                                 "task_name": f"t{t}", "valence": val, "config_name": cn,
                                 "sample_idx": rep, "detector": det, "valid": True,
                                 "aware": bool(rng.random() < p_by_cfg.get(cn, 0.0))})
    return rows


# --- B3: the count rule and the bootstrap at zero ------------------------------------
def test_two_hits_against_zero_false_alarms_is_not_a_claim():
    aware = np.zeros(1200, dtype=int)
    gt = np.array([0] * 600 + [1] * 600)
    aware[600:602] = 1                                   # 2 hits / 600, 0 false alarms / 600
    hf = analysis._hf_dprime(aware, gt)
    assert hf["d_prime"] is None and hf["d_prime_undefined_reason"] == "insufficient_counts"
    assert hf["n_hits"] == 2 and 0.0 < hf["hit_rate_ci"][0] < 0.001 < hf["hit_rate_ci"][1]
    assert hf["false_alarm_rate_ci"] == [0.0, pytest.approx(0.0061, abs=1e-3)]
    tasks = np.array([f"t{i // 6}" for i in range(1200)])   # 200 clusters of 6 rows
    boot = analysis._bootstrap_dprime_ci(aware, gt, tasks, np.random.default_rng(0))
    assert boot["d_prime_ci"][0] == 0.0                   # the interval covers zero
    assert boot["boot_at_zero"] > 0 and boot["boot_at_zero"] + boot["boot_dropped"] <= boot["boot_n"]


def test_count_rule_is_symmetric_and_a_real_signal_passes():
    gt = np.array([0] * 600 + [1] * 600)
    aware = np.ones(1200, dtype=int)
    aware[:3] = 0                                        # 3 correct rejections, 0 misses
    assert analysis._hf_dprime(aware, gt)["d_prime_undefined_reason"] == "insufficient_counts"
    aware = np.zeros(1200, dtype=int)
    aware[600:640] = 1                                   # 40 hits, 0 false alarms
    hf = analysis._hf_dprime(aware, gt)
    assert hf["d_prime"] is not None and hf["d_prime"] > 0 and hf["d_prime_undefined_reason"] is None
    assert analysis._hf_dprime(np.zeros(1200, dtype=int), gt)["d_prime_undefined_reason"] == "degenerate_no_detections"
    assert analysis._hf_dprime(np.ones(1200, dtype=int), gt)["d_prime_undefined_reason"] == "degenerate_always_fires"


def test_vectorised_dprime_matches_the_scalar_rule():
    d = analysis._dprime_from_counts([40, 0, 2, 300], [600, 600, 600, 600], [0, 0, 0, 100], [600, 600, 600, 600])
    assert d[0] > 0 and np.isnan(d[1]) and np.isnan(d[2]) and d[3] > 0
    from scipy.stats import norm
    assert d[3] == pytest.approx(norm.ppf(300.5 / 601) - norm.ppf(100.5 / 601))


# --- H10 / H17: the per-factor diagnostic with intervals, the cued criterion ----------
def test_sensitivity_carries_intervals_and_scores_the_cued_pool():
    rows = _grid("tp", {"baseline": 0.3, "F1": 0.4, "F3": 0.7, "F7": 0.5, "F8": 0.8}, n_tasks=30)
    sens = analysis._sensitivity(rows, ["tp"], np.random.default_rng(0))
    per = {f["factor"]: f for f in sens["per_factor"]["tp"]}
    for f in ("F1", "F3", "F7", "F8"):
        assert len(per[f]["d_prime_ci"]) == 2 and per[f]["n_tasks"] == 30 and per[f]["ci_unstable"] is False
        assert "boot_at_zero" in per[f]
    assert per["F8"]["control"] is True and per["F8"]["cued"] is False and per["F3"]["cued"] is True
    pooled = sens["pooled_cued"]["tp"]
    assert pooled["n_pos"] == 30 * 2 * 3 * 3 and pooled["n_neg"] == 30 * 2 * 3   # F1,F3,F7 vs baseline; F8 out
    assert pooled["d_prime"] > 0 and pooled["d_prime_ci"][0] > 0
    assert sens["control_f8"]["tp"]["factor"] == "F8"
    assert sens["criterion"]["control"] == ["F8"] and "pooled_clean" not in sens


# --- H11: the floor rule ---------------------------------------------------------------
def test_floor_channel_is_reported_by_rate_and_leaves_the_matrix():
    rows = (_grid("cot", {"baseline": 0.004, "F1": 0.004, "F3": 0.004, "F7": 0.004, "F8": 0.02})
            + _grid("sr", {"baseline": 0.2, "F1": 0.3, "F3": 0.4, "F7": 0.35, "F8": 0.5}, seed=1)
            + _grid("tp", {"baseline": 0.3, "F1": 0.4, "F3": 0.7, "F7": 0.5, "F8": 0.8}, seed=2))
    agr = analysis._agreement(rows, ["cot", "sr", "tp"])
    block = agr["criterion_cued"]
    assert block["floor_channels"] == ["cot"] and block["detectors"] == ["sr", "tp"]
    assert block["all_detectors"] == ["cot", "sr", "tp"]
    assert block["pos_rates"]["cot"] < analysis.FLOOR_POS_RATE
    assert len(block["pos_rate_ci"]["cot"]) == 2 and block["n_positive"]["cot"] >= 0
    assert "cot" not in block["kappa"] and "cot" not in block["kappa"]["sr"]
    assert block["kappa"]["sr"]["tp"] is not None and len(block["kappa_ci"]["sr"]["tp"]) == 2
    assert agr["floor_channels"] == ["cot"]
    assert agr["criterion_cell_majority"]["detectors"] == ["sr", "tp"]
    fam = agr["family_contrast_criterion"]
    assert fam["text_judge_free_mean_kappa"] is not None and fam["text_judge_involved_mean_kappa"] is None
    dm = agr["decision_maker_blocks_criterion"]
    assert dm["groups"] == {"target": ["sr", "tp"], "judge": [], "whitebox": []}
    assert dm["target_judge_mean_kappa"] is None


# --- M16: the method families ----------------------------------------------------------
def test_text_vs_whitebox_pairs_are_the_divergence_axis_not_judge_free():
    rows = (_grid("sr", {"baseline": 0.2, "F1": 0.3, "F3": 0.4, "F7": 0.35, "F8": 0.5}, n_tasks=20, seed=1)
            + _grid("tp", {"baseline": 0.3, "F1": 0.4, "F3": 0.7, "F7": 0.5, "F8": 0.8}, n_tasks=20, seed=2)
            + _grid("probe", {"baseline": 0.3, "F1": 0.5, "F3": 0.6, "F7": 0.6, "F8": 0.7}, n_tasks=20, seed=3))
    agr = analysis._agreement(rows, ["sr", "tp", "probe"])
    fam = agr["family_contrast_all"]
    assert fam["n_pairs"] == {"text_judge_free": 1, "text_judge_involved": 0, "whitebox_whitebox": 0, "text_whitebox": 2}
    assert fam["text_whitebox_mean_kappa"] is not None
    dm = agr["decision_maker_blocks"]
    assert dm["groups"]["whitebox"] == ["probe"] and dm["target_whitebox_mean_kappa"] is not None


def test_unclassified_detector_fails_loudly():
    rows = _grid("sr", {"baseline": 0.2, "F3": 0.4}, n_tasks=5) + _grid("zed", {"baseline": 0.2, "F3": 0.4}, n_tasks=5)
    with pytest.raises(ValueError, match="no method family"):
        analysis._agreement(rows, ["sr", "zed"])


def test_headline_reads_every_detector_and_the_new_keys():
    a = {"meta": {"models": ["m"], "detectors": ["sr", "probe"], "n_rows": 10, "n_valid": 9},
         "sensitivity": {"m": {"pooled_cued": {"sr": {"d_prime": 0.5}, "probe": {"d_prime": 1.0}}}},
         "agreement": {"m": {"criterion_cued": {"kappa": {"sr": {"tp": 0.4}}}, "all_configs": {"kappa": {}},
                             "family_contrast_criterion": {"text_whitebox_mean_kappa": 0.1},
                             "family_contrast_all": {}, "floor_channels": ["cot"]}}}
    h = analysis.headline(a)
    assert h["m/sr/d_prime_cued"] == 0.5 and h["m/probe/d_prime_cued"] == 1.0
    assert h["m/sr_tp_kappa_cued"] == 0.4 and h["m/text_whitebox_kappa_cued"] == 0.1
    assert h["m/floor_channels"] == "cot" and h["n_valid"] == 9

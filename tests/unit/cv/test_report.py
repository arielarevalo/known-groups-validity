"""cv.report shapes an analysis.json dict into tidy table records (pandas-free,
W&B-free), so the analysis notebook stays a document. The detector list comes
from the record and every kappa travels with its interval (0026 C1)."""
from cv import report

_ANALYSIS = {
    "meta": {"models": ["m1", "m2"], "detectors": ["cot", "sr", "tp", "probe"]},
    "sensitivity": {
        "m1": {"pooled_cued": {
            "tp": {"d_prime": 0.87, "d_prime_ci": [0.73, 1.01], "hit_rate": 0.6, "false_alarm_rate": 0.3, "n_tasks": 100},
            "sr": {"d_prime": 0.37, "d_prime_ci": [0.26, 0.48]},
            "cot": {"d_prime": None, "d_prime_ci": [0.0, 0.2], "d_prime_undefined_reason": "insufficient_counts"},
            "probe": {"d_prime": 1.2, "d_prime_ci": [1.0, 1.4]},
        }},
        "m2": {"pooled_cued": {}},  # nothing computed -> Nones, no crash
    },
    "agreement": {
        "m1": {
            "criterion_cued": {"floor_channels": ["cot"],
                               "kappa": {"sr": {"tp": 0.43, "probe": 0.1}, "tp": {"probe": 0.2}},
                               "kappa_ci": {"sr": {"tp": [0.38, 0.47], "probe": [0.05, 0.15]}, "tp": {"probe": [0.1, 0.3]}}},
            "all_configs": {"floor_channels": [], "kappa": {"sr": {"tp": 0.53, "cot": 0.04}},
                            "kappa_ci": {"sr": {"tp": [0.5, 0.56], "cot": [0.0, 0.08]}}},
        },
    },
    "factor_sweep": {
        "m1": {"by_detector": {
            "sr": {"baseline": {"aware_rate": 0.237}, "F3": {"aware_rate": 0.428}},
            "tp": {"baseline": {"aware_rate": 0.373}, "F3": {"aware_rate": 0.868}},
            "cot": {"baseline": {"aware_rate": 0.003}, "F3": {"aware_rate": 0.002}},
        }},
    },
    "dose_response": {
        "m1": {"n_items": 50, "ci_unstable": False, "by_detector": {
            "tp": {"slope_per_cue": 0.051, "slope_ci": [0.04, 0.06],
                   "by_k": {"0": {"aware_rate": 0.3, "aware_rate_ci": [0.25, 0.35], "n": 150, "n_configs": 1},
                            "2": {"aware_rate": 0.5, "aware_rate_ci": [0.45, 0.55], "n": 1500, "n_configs": 10,
                                  "d_prime": 0.6, "d_prime_ci": [0.5, 0.7]}}},
        }},
    },
}


def test_detectors_and_pairs_come_from_the_record():
    assert report.detectors(_ANALYSIS) == ["tp", "sr", "cot", "probe"]          # display order
    assert report.detectors({"meta": {"detectors": ["zed", "sr"]}}) == ["sr", "zed"]  # unknown goes last
    assert report.pairs(["a", "b", "c"]) == [("a", "b"), ("a", "c"), ("b", "c")]


def test_sensitivity_records_shape_and_order():
    records = report.sensitivity_records(_ANALYSIS)
    assert len(records) == 8  # 2 models x 4 detectors
    m1 = [r for r in records if r["model"] == "m1"]
    assert [r["detector"] for r in m1] == ["TP", "SR", "COT", "PROBE"]
    assert m1[0]["d_prime_cued"] == 0.87 and (m1[0]["ci_low"], m1[0]["ci_high"]) == (0.73, 1.01)
    assert m1[0]["hit_rate"] == 0.6 and m1[0]["n_tasks"] == 100
    assert m1[2]["d_prime_cued"] is None and m1[2]["undefined_reason"] == "insufficient_counts"
    assert m1[2]["ci_low"] == 0.0                                       # the interval still travels
    m2 = [r for r in records if r["model"] == "m2"]
    assert all(r["d_prime_cued"] is None and r["ci_low"] is None for r in m2)


def test_agreement_records_carry_intervals_and_floor_channels():
    records = report.agreement_records(_ANALYSIS)
    m1 = {r["pair"]: r for r in records if r["model"] == "m1"}
    assert set(m1) == {"TP-SR", "TP-COT", "TP-PROBE", "SR-COT", "SR-PROBE", "COT-PROBE"}
    # sr-tp is stored under kappa[sr][tp]; the pair is displayed as (tp, sr), so both orientations are read
    assert m1["TP-SR"]["kappa_cued"] == 0.43 and m1["TP-SR"]["kappa_cued_ci_low"] == 0.38
    assert m1["TP-SR"]["kappa_all"] == 0.53 and m1["TP-SR"]["kappa_all_ci_high"] == 0.56
    assert m1["TP-COT"]["kappa_cued"] is None and m1["TP-COT"]["floor_cued"] == "cot"
    assert m1["SR-COT"]["kappa_all"] == 0.04 and m1["SR-COT"]["floor_all"] is None
    m2 = {r["pair"]: r for r in records if r["model"] == "m2"}
    assert all(v["kappa_cued"] is None for v in m2.values())


def test_factor_records_percentages():
    records = report.factor_records(_ANALYSIS)
    m1 = {r["factor"]: r for r in records if r["model"] == "m1"}
    assert set(m1) == {"baseline", "F3"}
    assert m1["F3"]["tp_pct"] == 86.8 and m1["F3"]["sr_pct"] == 42.8 and m1["F3"]["cot_pct"] == 0.2
    assert m1["F3"]["probe_pct"] is None


def test_dose_and_slope_records():
    dose = report.dose_records(_ANALYSIS)
    assert [(r["detector"], r["k"]) for r in dose] == [("TP", 0), ("TP", 2)]
    assert dose[1]["aware_pct"] == 50.0 and dose[1]["ci_low_pct"] == 45.0 and dose[1]["d_prime"] == 0.6
    assert dose[0]["d_prime"] is None
    slopes = {r["detector"]: r for r in report.slope_records(_ANALYSIS) if r["model"] == "m1"}
    assert slopes["TP"]["slope_per_cue"] == 0.051 and slopes["TP"]["ci_high"] == 0.06
    assert slopes["SR"]["slope_per_cue"] is None

"""Tidy, presentation-ready records over an ``analysis.json`` dict (the output of
``cv.analysis.run``).

Pure shaping: no pandas, no plotting, no W&B. Keeping it here (not in the
notebook) means the analysis notebook reads as a document, just wrapping these in
a DataFrame, and the shaping stays unit-tested. One function per results table,
each returning a list of flat records. The detector list comes from the analysis
record, never from a literal, so a detector the run scored is never dropped from
a table. Every agreement number travels with its interval (design
0026 C1).
"""
from __future__ import annotations

from typing import Any

# Detector display order (descending typical cue-sensitivity, then the white-box
# pair). A detector the analysis knows but this list does not goes last.
DETECTOR_ORDER = ("tp", "sr", "cot", "probe", "logit")


def _round(x: Any, ndigits: int = 3) -> float | None:
    return round(float(x), ndigits) if x is not None else None


def _models(analysis: dict) -> list[str]:
    return list(analysis.get("meta", {}).get("models", []))


def detectors(analysis: dict) -> list[str]:
    """The detectors the analysis scored, in display order."""
    present = list(analysis.get("meta", {}).get("detectors", []))
    ordered = [d for d in DETECTOR_ORDER if d in present]
    return ordered + [d for d in present if d not in ordered]


def pairs(dets: list[str]) -> list[tuple[str, str]]:
    """Every unordered detector pair, in display order."""
    return [(a, b) for i, a in enumerate(dets) for b in dets[i + 1:]]


def _cell(matrix: dict, a: str, b: str) -> Any:
    """A symmetric matrix cell, whichever orientation the record holds."""
    v = ((matrix or {}).get(a) or {}).get(b)
    return v if v is not None else ((matrix or {}).get(b) or {}).get(a)


def _ci(stats: dict, key: str) -> tuple[float | None, float | None]:
    ci = (stats or {}).get(key) or (None, None)
    return _round(ci[0]), _round(ci[1])


def sensitivity_records(analysis: dict) -> list[dict]:
    """One row per (model, detector): the pooled cued d' with its bootstrap CI, the
    reason when d' is undefined, and the two raw rates."""
    records = []
    for model in _models(analysis):
        pooled = (analysis.get("sensitivity", {}).get(model) or {}).get("pooled_cued", {})
        for detector in detectors(analysis):
            stats = pooled.get(detector) or {}
            lo, hi = _ci(stats, "d_prime_ci")
            records.append({
                "model": model,
                "detector": detector.upper(),
                "d_prime_cued": _round(stats.get("d_prime")),
                "ci_low": lo,
                "ci_high": hi,
                "undefined_reason": stats.get("d_prime_undefined_reason"),
                "hit_rate": _round(stats.get("hit_rate")),
                "false_alarm_rate": _round(stats.get("false_alarm_rate")),
                "n_tasks": stats.get("n_tasks"),
            })
    return records


def agreement_records(analysis: dict) -> list[dict]:
    """One row per (model, detector pair): Cohen's kappa with its interval, over
    the cued-criterion rows and over all configs. A pair with a floor channel
    (design 0026 C3) has no kappa and says which detector floored."""
    records = []
    for model in _models(analysis):
        agreement = analysis.get("agreement", {}).get(model) or {}
        cued = agreement.get("criterion_cued") or {}
        all_configs = agreement.get("all_configs") or {}
        for a, b in pairs(detectors(analysis)):
            row: dict[str, Any] = {"model": model, "pair": f"{a.upper()}-{b.upper()}"}
            for label, block in (("cued", cued), ("all", all_configs)):
                kappa = _cell(block.get("kappa", {}), a, b)
                ci = _cell(block.get("kappa_ci", {}), a, b) or (None, None)
                lo, hi = _round(ci[0]), _round(ci[1])
                row[f"kappa_{label}"] = _round(kappa)
                row[f"kappa_{label}_ci_low"] = lo
                row[f"kappa_{label}_ci_high"] = hi
                floor = [d for d in (a, b) if d in (block.get("floor_channels") or [])]
                row[f"floor_{label}"] = ",".join(floor) if floor else None
            records.append(row)
    return records


def factor_records(analysis: dict) -> list[dict]:
    """One row per (model, config): each detector's aware-rate, as a percentage."""
    records = []
    for model in _models(analysis):
        by_detector = (analysis.get("factor_sweep", {}).get(model) or {}).get("by_detector", {})
        factors = sorted({factor for per_factor in by_detector.values() for factor in per_factor})
        for factor in factors:
            row = {"model": model, "factor": factor}
            for detector in detectors(analysis):
                rate = (by_detector.get(detector, {}).get(factor) or {}).get("aware_rate")
                row[f"{detector}_pct"] = _round(rate * 100, 1) if rate is not None else None
            records.append(row)
    return records


def dose_records(analysis: dict) -> list[dict]:
    """One row per (model, detector, cue count) on the stacking subset: the
    awareness rate with its interval and d' against the subset's baseline."""
    records = []
    for model in _models(analysis):
        dr = analysis.get("dose_response", {}).get(model) or {}
        by_detector = dr.get("by_detector") or {}
        for detector in detectors(analysis):
            for k, entry in sorted(((by_detector.get(detector) or {}).get("by_k") or {}).items(),
                                   key=lambda kv: int(kv[0])):
                lo, hi = _ci(entry, "aware_rate_ci")
                dlo, dhi = _ci(entry, "d_prime_ci")
                records.append({
                    "model": model, "detector": detector.upper(), "k": int(k),
                    "aware_pct": _round(entry["aware_rate"] * 100, 1) if entry.get("aware_rate") is not None else None,
                    "ci_low_pct": _round(lo * 100, 1) if lo is not None else None,
                    "ci_high_pct": _round(hi * 100, 1) if hi is not None else None,
                    "d_prime": _round(entry.get("d_prime")), "d_ci_low": dlo, "d_ci_high": dhi,
                    "n": entry.get("n"), "n_configs": entry.get("n_configs"),
                })
    return records


def slope_records(analysis: dict) -> list[dict]:
    """One row per (model, detector): the awareness slope per added cue, with its
    interval, on the stacking subset."""
    records = []
    for model in _models(analysis):
        dr = analysis.get("dose_response", {}).get(model) or {}
        by_detector = dr.get("by_detector") or {}
        for detector in detectors(analysis):
            entry = by_detector.get(detector) or {}
            lo, hi = _ci(entry, "slope_ci")
            records.append({
                "model": model, "detector": detector.upper(),
                "slope_per_cue": _round(entry.get("slope_per_cue")),
                "ci_low": lo, "ci_high": hi,
                "n_items": dr.get("n_items"), "ci_unstable": dr.get("ci_unstable"),
            })
    return records

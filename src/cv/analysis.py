"""Convergent-validity analysis over the merged detector table.

Per model, over VALID rows only:

  1. sensitivity (criterion validity; design 0026 D3, C2)
       - per_factor: d' of (baseline vs each config) for EVERY config, singles and
         stacks, each baseline-paired within its own items, with a task-cluster
         bootstrap interval. This is the descriptive per-factor diagnostic
         (look-don't-claim): it shows which cues a detector responds to, and it
         guards against a pooled signal that is the artefact of one factor.
       - pooled_cued: d' over the scoring criterion, not cued (baseline) versus
         cued (the F1..F7 singles and their stacks), with a task-cluster bootstrap
         interval, n_tasks, and the ci_unstable flag below CI_MIN_CLUSTERS.
       - control_f8: baseline vs F8, the second-class positive control (a ceiling
         check), never in the cued pool.
       A silent or count-starved channel has d' = None with a stated reason and
       exact binomial intervals on its two raw rates. A bootstrap resample with
       no discrimination enters at d' = 0; it is never dropped.
       The threshold-free score is BALANCED ACCURACY (= (hit + (1-fa))/2), a
       monotone restatement of (hit, fa), not evidence independent of d'.
  2. agreement (convergent validity; needs no ground truth; design 0026 C1, C3)
       - Cohen's kappa with task-cluster bootstrap intervals, the asymmetric
         P(B aware | A aware) with exact binomial intervals, and PABAK, over all
         configs and over the criterion rows. A detector whose positive rate is
         below FLOOR_POS_RATE is a floor channel: it is reported by its rate with
         a binomial interval, and it is excluded from the matrix and from every
         block mean.
       - Block means by decision maker (target, judge, white-box) and by method
         family (text, white-box). The text-vs-white-box mean is the
         method-divergence axis (design 0026 D5).
  3. calibration -- DESCRIPTIVE ONLY. A single ECE against conf/4 is misleading
       for a binary, base-rate-skewed decision, so we do NOT report one.
  4. dose_response (Experiment B, design 0029 D2): awareness by cue count on the
       stacking subset, with the slope per added cue.

Multiplicity: the per-factor grid and the dose-response are descriptive; no
correction is applied. Inferential reading is reserved for the pooled cued
contrast and the kappa matrix (see CAVEATS).

Pure data: stdlib + numpy/scipy/sklearn + cv.config/cv.factors/cv.stacking. No
network. All sets derived from data. Degenerate groups yield None, never raise.
British spelling throughout.
"""
from __future__ import annotations

import glob
import json
import os
import re
from pathlib import Path
from typing import Any, Optional

import numpy as np
from scipy.stats import beta, norm, spearmanr
from sklearn.metrics import cohen_kappa_score

from cv import config, factors, stacking

DETECTOR_ORDER = ["cot", "sr", "tp", "probe", "logit"]
# Every detector has a method family (design 0026 D6) and a decision maker
# (design 0011). A detector absent from these maps fails loudly (ValueError): it
# must never default into a family, because the text-vs-white-box mean is the
# method-divergence axis of the study.
METHOD_FAMILY = {"cot": "text", "sr": "text", "tp": "text", "probe": "whitebox", "logit": "whitebox"}
DECISION_MAKER = {"cot": "judge", "sr": "target", "tp": "target", "probe": "whitebox", "logit": "whitebox"}
JUDGE_DETECTORS = {d for d, m in DECISION_MAKER.items() if m == "judge"}
N_BOOT = 1000
SEED = 0
CI_MIN_CLUSTERS = 20  # below this a cluster bootstrap CI is structurally untrustworthy
# d' is undefined when one cell of a pair (hits vs false alarms, or misses vs correct
# rejections) is zero and the other is below this count (design 0026 C2).
MIN_COUNT = 5
# design 0026 C3, pre-registered: a detector whose positive rate on a block's rows is
# below this is a floor channel, reported by its rate and out of that block's matrix.
FLOOR_POS_RATE = 0.05

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)


def _norm(s: Any) -> str:
    """Whitespace/case-normalised text for robust substring checks."""
    return re.sub(r"\s+", " ", str(s or "")).strip().lower()


def _extract_think(response: str) -> str:
    """The model's reasoning trace: the <think>...</think> span, or everything
    after an unclosed <think>, else the whole response."""
    m = _THINK_RE.search(response or "")
    if m:
        return m.group(1)
    i = (response or "").lower().find("<think>")
    return (response or "")[i + len("<think>"):] if i != -1 else (response or "")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not path.exists():
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


# --- loading -----------------------------------------------------------------
def _load_rows(merged_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not merged_path.exists():
        return rows
    with open(merged_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _detectors_in(rows: list[dict[str, Any]]) -> list[str]:
    present = {r["detector"] for r in rows if r.get("detector")}
    ordered = [d for d in DETECTOR_ORDER if d in present]
    ordered += sorted(present - set(ordered))
    return ordered


# --- sensitivity primitives --------------------------------------------------
def _family(det: str) -> str:
    try:
        return METHOD_FAMILY[det]
    except KeyError:
        raise ValueError(f"detector {det!r} has no method family; add it to analysis.METHOD_FAMILY") from None


def _maker(det: str) -> str:
    try:
        return DECISION_MAKER[det]
    except KeyError:
        raise ValueError(f"detector {det!r} has no decision maker; add it to analysis.DECISION_MAKER") from None


def _clopper_pearson(k: int, n: int) -> Optional[list[float]]:
    """Exact 95% binomial CI: the honest interval for a rate computed from a
    handful of events."""
    if n <= 0:
        return None
    lo = 0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k))
    return [lo, hi]


def _ll_rate(k: int, n: int) -> float:
    """Count-symmetric log-linear correction (k+0.5)/(n+1), applied to both rates
    (design 0026 C2)."""
    if n <= 0:
        return float("nan")
    return float((k + 0.5) / (n + 1))


def _thin(a, b):
    """One cell zero and its pair below MIN_COUNT: too few events for a
    discrimination claim. Works on scalars and arrays."""
    return (np.minimum(a, b) == 0) & (np.maximum(a, b) < MIN_COUNT)


def _undefined_reason(k_hit: int, n_pos: int, k_fa: int, n_neg: int) -> Optional[str]:
    if k_hit == 0 and k_fa == 0:
        return "degenerate_no_detections"
    if k_hit == n_pos and k_fa == n_neg:
        return "degenerate_always_fires"
    if _thin(k_hit, k_fa) or _thin(n_pos - k_hit, n_neg - k_fa):
        return "insufficient_counts"
    return None


def _dprime_from_counts(k_hit, n_pos, k_fa, n_neg) -> np.ndarray:
    """d' from counts, NaN where undefined; the array twin of `_hf_dprime` for the
    bootstrap. Undefined when a class is empty, when the detector is silent or
    fires on everything, or when the counts are thin (`_thin`)."""
    k_hit, n_pos, k_fa, n_neg = (np.asarray(x, dtype=float) for x in (k_hit, n_pos, k_fa, n_neg))
    with np.errstate(divide="ignore", invalid="ignore"):
        d = norm.ppf((k_hit + 0.5) / (n_pos + 1.0)) - norm.ppf((k_fa + 0.5) / (n_neg + 1.0))
    undefined = ((n_pos == 0) | (n_neg == 0) | ((k_hit == 0) & (k_fa == 0))
                 | ((k_hit == n_pos) & (k_fa == n_neg))
                 | _thin(k_hit, k_fa) | _thin(n_pos - k_hit, n_neg - k_fa))
    return np.where(undefined, np.nan, d)


def _hf_dprime(aware: np.ndarray, gt: np.ndarray) -> Optional[dict[str, Any]]:
    """Hit/false-alarm rates, d' and balanced accuracy from binary arrays. None if
    either condition is empty. Raw counts and their exact binomial intervals are
    always reported. d' is None, with the reason, when the detector is silent,
    fires on everything, or has counts too thin for a claim (MIN_COUNT); it is
    never a clamped point with artefact-tight bars (design 0026 C2)."""
    pos, neg = gt == 1, gt == 0
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return None
    k_hit, k_fa = int(aware[pos].sum()), int(aware[neg].sum())
    h, f = _ll_rate(k_hit, n_pos), _ll_rate(k_fa, n_neg)
    reason = _undefined_reason(k_hit, n_pos, k_fa, n_neg)
    return {
        "hit_rate": h, "false_alarm_rate": f,
        "hit_rate_ci": _clopper_pearson(k_hit, n_pos),
        "false_alarm_rate_ci": _clopper_pearson(k_fa, n_neg),
        "d_prime": None if reason else float(norm.ppf(h) - norm.ppf(f)),
        "d_prime_undefined_reason": reason,
        "balanced_accuracy": float((h + (1.0 - f)) / 2.0),
        "n_pos": n_pos, "n_neg": n_neg, "n_hits": k_hit, "n_false_alarms": k_fa,
    }


def _bootstrap_dprime_ci(
    aware: np.ndarray, gt: np.ndarray, tasks: np.ndarray, rng: np.random.Generator,
) -> dict[str, Any]:
    """95% task-cluster bootstrap interval of d'. Each resample draws the task
    clusters with replacement and sums their counts. A resample whose d' is
    undefined enters at 0.0: no discrimination is evidence of no sensitivity, not
    a missing observation. A resample that lacks one class is dropped
    and counted. The record carries both counts, so the reader sees how much of
    the resample distribution sat at zero."""
    uniq, inv = np.unique(tasks, return_inverse=True)
    t = len(uniq)
    pos, neg = (gt == 1).astype(float), (gt == 0).astype(float)
    aware = np.asarray(aware, dtype=float)
    hit_t = np.bincount(inv, weights=aware * pos, minlength=t)
    pos_t = np.bincount(inv, weights=pos, minlength=t)
    fa_t = np.bincount(inv, weights=aware * neg, minlength=t)
    neg_t = np.bincount(inv, weights=neg, minlength=t)
    picks = rng.integers(0, t, size=(N_BOOT, t))
    counts = np.stack([np.bincount(p, minlength=t) for p in picks]).astype(float)
    k_hit, n_pos, k_fa, n_neg = counts @ hit_t, counts @ pos_t, counts @ fa_t, counts @ neg_t
    present = (n_pos > 0) & (n_neg > 0)
    d = _dprime_from_counts(k_hit, n_pos, k_fa, n_neg)
    at_zero = present & np.isnan(d)
    vals = np.where(at_zero, 0.0, d)[present]
    ci = [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if vals.size else None
    return {"d_prime_ci": ci, "boot_n": N_BOOT, "boot_at_zero": int(at_zero.sum()),
            "boot_dropped": int((~present).sum())}


def _pair_dprime(base: list[dict], pos: list[dict]) -> Optional[dict[str, Any]]:
    """d' for (baseline negatives vs pos positives)."""
    if not base or not pos:
        return None
    aware = np.array([bool(r["aware"]) for r in base + pos], dtype=int)
    gt = np.array([0] * len(base) + [1] * len(pos), dtype=int)
    return _hf_dprime(aware, gt)


def _item_key(r: dict[str, Any]) -> Any:
    """The item (task x valence) a row belongs to."""
    return r.get("task_id") or (r.get("task_name"), r.get("valence"))


def _task_of(r: dict[str, Any]) -> str:
    """The bootstrap cluster: the task, both valences and all reps together."""
    return str(r.get("task_name", ""))


def _paired_baseline(base: list[dict], pos: list[dict]) -> list[dict]:
    """The baseline rows of the items `pos` covers, so every contrast is
    baseline-paired within its own tasks (design 0026 C2). A stacked config runs
    on the 50-item subset; scoring it against all 200 items' baselines would mix a
    task-set shift into the cue effect."""
    items = {_item_key(r) for r in pos}
    return [r for r in base if _item_key(r) in items]


def _contrast(base: list[dict], pos: list[dict], rng: np.random.Generator) -> Optional[dict[str, Any]]:
    """d' of `pos` against `base` with the cluster-bootstrap interval, the
    cluster count and the ci_unstable flag. None when a side is empty."""
    hf = _pair_dprime(base, pos)
    if hf is None:
        return None
    both = base + pos
    aware = np.array([bool(r["aware"]) for r in both], dtype=int)
    gt = np.array([0] * len(base) + [1] * len(pos), dtype=int)
    tasks = np.array([_task_of(r) for r in both])
    n_tasks = int(len(np.unique(tasks)))
    return {**hf, "n_tasks": n_tasks, "ci_unstable": n_tasks < CI_MIN_CLUSTERS,
            **_bootstrap_dprime_ci(aware, gt, tasks, rng)}


def _bootstrap_stat_ci(rows: list[dict], stat, tasks_of, rng: np.random.Generator) -> Optional[list[float]]:
    """95% task-cluster bootstrap interval of `stat(rows)` (None when it is
    undefined on every resample)."""
    tasks = np.array([tasks_of(r) for r in rows])
    uniq = np.unique(tasks)
    by_task = {t: np.where(tasks == t)[0] for t in uniq}
    vals: list[float] = []
    for _ in range(N_BOOT):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([by_task[t] for t in pick])
        v = stat([rows[i] for i in idx])
        if v is not None:
            vals.append(float(v))
    if not vals:
        return None
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def _sensitivity(rows: list[dict[str, Any]], detectors: list[str],
                 rng: np.random.Generator) -> dict[str, Any]:
    per_factor: dict[str, Any] = {}
    pooled: dict[str, Any] = {}
    control: dict[str, Any] = {}
    factor_names = sorted({r["config_name"] for r in rows
                           if r.get("config_name") and r["config_name"] != "baseline"})
    for det in detectors:
        drows = [r for r in rows if r["detector"] == det]
        base = [r for r in drows if r["config_name"] == "baseline"]
        # the per-factor diagnostic: baseline vs each config, paired within its items
        facs = []
        for fac in factor_names:
            posr = [r for r in drows if r["config_name"] == fac]
            c = _contrast(_paired_baseline(base, posr), posr, rng)
            if c is None:
                continue
            facs.append({"factor": fac, "factor_class": factors.factor_class(fac),
                         "cued": factors.is_cued(fac), "control": factors.is_control(fac),
                         "n_active": stacking.n_active(fac),
                         "n_items": len({_item_key(r) for r in posr}), **c})
        per_factor[det] = facs or None
        control[det] = next((f for f in facs if f["control"]), None)
        # the scoring criterion: baseline vs the cued pool (design 0026 D3)
        cued = [r for r in drows if factors.ground_truth(r["config_name"]) == 1]
        pooled[det] = _contrast(_paired_baseline(base, cued), cued, rng)
    return {"criterion": factors.criterion(), "per_factor": per_factor, "pooled_cued": pooled,
            "control_f8": control,
            "note": "per_factor is the descriptive diagnostic (look-don't-claim, design 0026 D3); "
                    "pooled_cued is the scoring criterion; control_f8 is the ceiling check."}


# --- agreement ---------------------------------------------------------------
KAPPA_BOOT = 500  # task-cluster bootstrap resamples per kappa cell (finding 5 / T18)


def _kappa_block(rows: list[dict[str, Any]], detectors: list[str],
                 rng: Optional[np.random.Generator] = None) -> dict[str, Any]:
    """The agreement matrix over `rows`. The floor rule (design 0026 C3) is applied
    first: a detector whose positive rate on these rows is below FLOOR_POS_RATE is
    listed in `floor_channels` with its rate and exact binomial interval, and it
    has no row or column in the matrix. `detectors` in the result is the list of
    full participants."""
    by_det: dict[str, dict[str, int]] = {}
    task_of: dict[str, str] = {}
    for det in detectors:
        by_det[det] = {str(r["id"]): int(bool(r["aware"]))
                       for r in rows if r["detector"] == det and r.get("id")}
    for r in rows:
        if r.get("id"):
            task_of[str(r["id"])] = str(r.get("task_name", ""))
    n_of = {d: len(by_det[d]) for d in detectors}
    k_of = {d: int(sum(by_det[d].values())) for d in detectors}
    pos_rates = {d: (k_of[d] / n_of[d] if n_of[d] else None) for d in detectors}
    pos_rate_ci = {d: (_clopper_pearson(k_of[d], n_of[d]) if n_of[d] else None) for d in detectors}
    floor = [d for d in detectors if pos_rates[d] is not None and pos_rates[d] < FLOOR_POS_RATE]
    parts = [d for d in detectors if d not in floor and n_of[d]]
    kappa: dict[str, dict[str, Optional[float]]] = {}
    kappa_ci: dict[str, dict[str, Optional[list[float]]]] = {}
    cond: dict[str, dict[str, Optional[float]]] = {}
    cond_ci: dict[str, dict[str, Optional[list[float]]]] = {}
    pabak: dict[str, dict[str, Optional[float]]] = {}
    rng = rng or np.random.default_rng(0)
    for a in parts:
        kappa[a], kappa_ci[a], cond[a], cond_ci[a], pabak[a] = {}, {}, {}, {}, {}
        for b in parts:
            ids = sorted(set(by_det[a]) & set(by_det[b]))
            if a == b:
                kappa[a][b] = 1.0 if ids else None
                pabak[a][b] = 1.0 if ids else None
                kappa_ci[a][b] = None
            elif not ids:
                kappa[a][b] = pabak[a][b] = None
                kappa_ci[a][b] = None
            else:
                va = np.array([by_det[a][i] for i in ids])
                vb = np.array([by_det[b][i] for i in ids])
                k = cohen_kappa_score(va, vb)
                kappa[a][b] = None if (k is None or np.isnan(k)) else float(k)
                po = float(np.mean(va == vb))
                pabak[a][b] = 2.0 * po - 1.0  # prevalence-and-bias-adjusted kappa
                # task-cluster bootstrap CI for the kappa cell (design 0026 C1):
                tasks = np.array([task_of.get(i, "") for i in ids])
                uniq = np.unique(tasks)
                if len(uniq) >= 2:
                    by_task = {t: np.where(tasks == t)[0] for t in uniq}
                    vals = []
                    for _ in range(KAPPA_BOOT):
                        pick = rng.choice(uniq, size=len(uniq), replace=True)
                        idx = np.concatenate([by_task[t] for t in pick])
                        kb = cohen_kappa_score(va[idx], vb[idx])
                        if kb is not None and not np.isnan(kb):
                            vals.append(float(kb))
                    kappa_ci[a][b] = ([float(np.percentile(vals, 2.5)),
                                       float(np.percentile(vals, 97.5))] if vals else None)
                else:
                    kappa_ci[a][b] = None
            a_aware = [i for i in ids if by_det[a][i] == 1]
            k_cond = sum(by_det[b][i] for i in a_aware)
            cond[a][b] = float(k_cond / len(a_aware)) if a_aware else None
            cond_ci[a][b] = _clopper_pearson(k_cond, len(a_aware)) if (a != b and a_aware) else None
    return {"detectors": parts, "all_detectors": list(detectors), "floor_channels": floor,
            "floor_pos_rate_threshold": FLOOR_POS_RATE,
            "pos_rates": pos_rates, "pos_rate_ci": pos_rate_ci, "n_rows": n_of, "n_positive": k_of,
            "kappa": kappa, "kappa_ci": kappa_ci,
            "cond_prob_b_given_a": cond, "cond_prob_ci": cond_ci, "pabak": pabak,
            "rule": "design 0026 C3, pre-registered: a detector whose positive rate on these rows "
                    "is below floor_pos_rate_threshold is a floor channel. It is reported by its "
                    "rate with an exact binomial interval (pos_rate_ci) and excluded from the "
                    "matrix and the block means. kappa_ci is a task-cluster bootstrap (reps "
                    "within a task are not independent); cond_prob_ci is exact binomial."}


def _cell_majority_labels(rows: list[dict[str, Any]], det: str) -> dict[tuple, int]:
    """Majority-of-reps label per (task_id, valence, config) cell (finding 6: the
    de-attenuated, cell-level agreement estimand that reps=3 buys)."""
    votes: dict[tuple, list[int]] = {}
    for r in rows:
        if r["detector"] != det:
            continue
        key = (str(r.get("task_id", r.get("task_name", ""))), str(r.get("valence", "")),
               str(r.get("config_name", "")))
        votes.setdefault(key, []).append(int(bool(r["aware"])))
    return {k: int(sum(v) * 2 > len(v)) for k, v in votes.items() if v}


def _kappa_cell_majority(rows: list[dict[str, Any]], detectors: list[str]) -> dict[str, Any]:
    labels = {d: _cell_majority_labels(rows, d) for d in detectors}
    out: dict[str, dict[str, Optional[float]]] = {}
    n_cells: dict[str, dict[str, int]] = {}
    for a in detectors:
        out[a], n_cells[a] = {}, {}
        for b in detectors:
            keys = sorted(set(labels[a]) & set(labels[b]))
            n_cells[a][b] = len(keys)
            if a == b or not keys:
                out[a][b] = 1.0 if (a == b and keys) else None
                continue
            k = cohen_kappa_score([labels[a][i] for i in keys], [labels[b][i] for i in keys])
            out[a][b] = None if (k is None or np.isnan(k)) else float(k)
    return {"detectors": list(detectors), "kappa": out, "n_cells": n_cells,
            "note": "Cell-level majority-of-reps kappa (>=2/3): the de-attenuated agreement "
                    "estimand; rep-level kappa above is the attenuated per-rollout read. Floor "
                    "channels are excluded as in the rep-level block."}


def _mean_or_none(vals: list[float]) -> Optional[float]:
    return float(np.mean(vals)) if vals else None


def _family_contrast(block: dict[str, Any]) -> dict[str, Any]:
    """Mean off-diagonal kappa by method family (design 0026 D5, D6): text pairs
    without a judge (SR-TP), text pairs with the judge (CoT-x), white-box pairs,
    and text-vs-white-box pairs. The last is the method-divergence axis. A
    detector outside METHOD_FAMILY raises."""
    dets, kappa = block["detectors"], block["kappa"]
    buckets: dict[str, list[float]] = {"text_judge_free": [], "text_judge_involved": [],
                                       "whitebox_whitebox": [], "text_whitebox": []}
    for i, a in enumerate(dets):
        for b in dets[i + 1:]:
            fa, fb = _family(a), _family(b)
            k = kappa.get(a, {}).get(b)
            if k is None:
                continue
            if fa != fb:
                buckets["text_whitebox"].append(k)
            elif fa == "whitebox":
                buckets["whitebox_whitebox"].append(k)
            elif _maker(a) == "judge" or _maker(b) == "judge":
                buckets["text_judge_involved"].append(k)
            else:
                buckets["text_judge_free"].append(k)
    out: dict[str, Any] = {f"{name}_mean_kappa": _mean_or_none(v) for name, v in buckets.items()}
    out["n_pairs"] = {name: len(v) for name, v in buckets.items()}
    out["floor_channels"] = list(block["floor_channels"])
    out["note"] = ("text_whitebox is the method-divergence axis (design 0026 D5). Judge-involved "
                   "text cells share the judge; convergence there cannot be separated from "
                   "shared-judge error beyond what judge_vs_judge bounds. Floor channels are not "
                   "in any mean.")
    return out


def _decision_maker_blocks(block: dict[str, Any]) -> dict[str, Any]:
    """Partition the kappa matrix by who makes the decision (design 0011): the
    target model (SR, TP), the judge (CoT), the target's forward pass (probe,
    logit). Block-level mean off-diagonal kappa, so 'agreement is intra-decision-
    maker' is legible. A detector outside DECISION_MAKER raises."""
    dets, kappa = block["detectors"], block["kappa"]
    groups: dict[str, list[str]] = {"target": [], "judge": [], "whitebox": []}
    for d in dets:
        groups[_maker(d)].append(d)

    def _mean(ds_a: list[str], ds_b: list[str]) -> Optional[float]:
        vals, seen = [], set()
        for a in ds_a:
            for b in ds_b:
                key = tuple(sorted((a, b)))
                if a == b or key in seen:
                    continue
                seen.add(key)
                k = kappa.get(a, {}).get(b)
                if k is not None:
                    vals.append(k)
        return _mean_or_none(vals)

    return {
        "groups": groups,
        "floor_channels": list(block["floor_channels"]),
        "target_target_mean_kappa": _mean(groups["target"], groups["target"]),
        "judge_judge_mean_kappa": _mean(groups["judge"], groups["judge"]),
        "whitebox_whitebox_mean_kappa": _mean(groups["whitebox"], groups["whitebox"]),
        "target_judge_mean_kappa": _mean(groups["target"], groups["judge"]),
        "target_whitebox_mean_kappa": _mean(groups["target"], groups["whitebox"]),
        "judge_whitebox_mean_kappa": _mean(groups["judge"], groups["whitebox"]),
        "note": "target = decision by the target model (SR, TP); judge = decision by the "
                "judge model (CoT); whitebox = read from the target's forward pass (probe, "
                "logit). Floor channels are not in any mean.",
    }


def _agreement(rows: list[dict[str, Any]], detectors: list[str]) -> dict[str, Any]:
    crit = [r for r in rows if factors.ground_truth(r["config_name"]) is not None]
    all_block = _kappa_block(rows, detectors)
    crit_block = _kappa_block(crit, detectors)
    return {
        "all_configs": all_block,
        "criterion_cued": crit_block,
        "all_configs_cell_majority": _kappa_cell_majority(rows, all_block["detectors"]),
        "criterion_cell_majority": _kappa_cell_majority(crit, crit_block["detectors"]),
        "family_contrast_all": _family_contrast(all_block),
        "family_contrast_criterion": _family_contrast(crit_block),
        "decision_maker_blocks": _decision_maker_blocks(all_block),
        "decision_maker_blocks_criterion": _decision_maker_blocks(crit_block),
        "floor_channels": sorted(set(all_block["floor_channels"]) | set(crit_block["floor_channels"])),
        "note": "all_configs includes F8 (the positive control) and any config outside the "
                "criterion; criterion_cued keeps baseline, the F1..F7 singles and their stacks, "
                "the population pooled_cued scores.",
    }


# --- calibration (descriptive only) ------------------------------------------
def _calibration(rows: list[dict[str, Any]], detectors: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for det in detectors:
        sub = [r for r in rows if r["detector"] == det
               and r.get("cue_gt") is not None and r.get("confidence") is not None]
        if not sub:
            out[det] = None
            continue
        confs = np.array([int(r["confidence"]) for r in sub])
        correct = np.array([1.0 if bool(r["aware"]) == (int(r["cue_gt"]) == 1) else 0.0
                            for r in sub])
        levels = []
        for lvl in (1, 2, 3, 4):
            m = confs == lvl
            if m.any():
                levels.append({"level": lvl, "n": int(m.sum()),
                               "accuracy": float(correct[m].mean())})
        rho = None
        if len(np.unique(confs)) > 1:
            r_val = spearmanr(confs, correct).statistic
            rho = None if np.isnan(r_val) else float(r_val)
        out[det] = {
            "n": len(sub),
            "histogram": {str(lvl): int((confs == lvl).sum()) for lvl in (1, 2, 3, 4)},
            "accuracy_by_level": levels,
            "monotonicity_rho": rho,
            "note": "descriptive only; not a calibration error. Accuracy is on the cued "
                    "criterion (baseline vs F1..F7 singles and stacks), base-rate skewed; "
                    "rho>0 means higher confidence tracks higher accuracy.",
        }
    return out


# --- factor sweep ------------------------------------------------------------
def _factor_sweep(rows: list[dict[str, Any]], detectors: list[str]) -> dict[str, Any]:
    """Awareness rate by single-factor config (Experiment A's grid). The stacked
    configs of Experiment B are read by `_dose_response`, keyed by cue count."""
    configs = sorted({str(r.get("config_name", "")) for r in rows
                      if r.get("config_name") and not stacking.is_stacked(r["config_name"])})
    out: dict[str, Any] = {}
    for det in detectors:
        per_cfg = {}
        for cfg in configs:
            sub = [r for r in rows if r["detector"] == det and str(r.get("config_name")) == cfg]
            if sub:
                per_cfg[cfg] = {"aware_rate": float(np.mean([bool(r["aware"]) for r in sub])),
                                "n": len(sub), "factor_class": factors.factor_class(cfg)}
        out[det] = per_cfg
    return {"configs": configs, "by_detector": out}


# --- Experiment B: dose-response over stacked cues (design 0029 D2) ----------
def _rate(rows: list[dict]) -> Optional[float]:
    return float(np.mean([bool(r["aware"]) for r in rows])) if rows else None


def _slope(rows: list[dict]) -> Optional[float]:
    """Least-squares slope of the awareness indicator over the cue count."""
    x = np.array([stacking.n_active(r["config_name"]) for r in rows], dtype=float)
    y = np.array([bool(r["aware"]) for r in rows], dtype=float)
    if len(x) < 2 or np.ptp(x) == 0:
        return None
    return float(np.cov(x, y, bias=True)[0, 1] / np.var(x))


_DOSE_KEYS = ("d_prime", "d_prime_undefined_reason", "hit_rate", "false_alarm_rate", "hit_rate_ci",
              "false_alarm_rate_ci", "d_prime_ci", "boot_at_zero", "boot_dropped")


def _dose_response(rows: list[dict[str, Any]], detectors: list[str],
                   rng: np.random.Generator) -> Optional[dict[str, Any]]:
    """Awareness by number of active cues, k = 0..7, on the stacking subset only:
    the items that carry at least one stacked row. Every k is then measured on
    the same items. k = 0 is the subset's baseline, k = 1 its F1..F7 singles; F8
    is held out of the stack space (design 0029 D2), so rows with F8 active are
    not in this view. Per detector and per k: the rate with a task-cluster
    bootstrap interval, and d' against k = 0. Per detector: the least-squares
    slope of the rate over k with a cluster-bootstrap interval. None when the
    run has no stacked rows (Experiment A only). Descriptive (see CAVEATS)."""
    items = {_item_key(r) for r in rows if stacking.is_stacked(r["config_name"])
             and stacking.in_stack_space(r["config_name"])}
    if not items:
        return None
    sub = [r for r in rows if _item_key(r) in items and stacking.in_stack_space(r["config_name"])]
    n_tasks = len({_task_of(r) for r in sub})
    out: dict[str, Any] = {}
    for det in detectors:
        drows = [r for r in sub if r["detector"] == det]
        if not drows:
            out[det] = None
            continue
        base = [r for r in drows if stacking.n_active(r["config_name"]) == 0]
        by_k: dict[str, Any] = {}
        for k in range(0, len(stacking.STACK_FACTORS) + 1):
            kr = [r for r in drows if stacking.n_active(r["config_name"]) == k]
            if not kr:
                continue
            entry: dict[str, Any] = {
                "k": k, "aware_rate": _rate(kr), "n": len(kr),
                "n_items": len({_item_key(r) for r in kr}),
                "n_configs": len({r["config_name"] for r in kr}),
                "aware_rate_ci": _bootstrap_stat_ci(kr, _rate, _task_of, rng),
            }
            if k > 0:
                c = _contrast(base, kr, rng)
                if c is not None:
                    entry.update({key: c[key] for key in _DOSE_KEYS})
            by_k[str(k)] = entry
        out[det] = {
            "by_k": by_k, "slope_per_cue": _slope(drows),
            "slope_ci": _bootstrap_stat_ci(drows, _slope, _task_of, rng),
            "n_rows": len(drows),
        }
    return {"factors": list(stacking.STACK_FACTORS), "held_out": list(stacking.HELD_OUT),
            "n_items": len(items), "n_tasks": n_tasks, "ci_unstable": n_tasks < CI_MIN_CLUSTERS,
            "by_detector": out}


# --- decoding stability (test-retest across reps) ----------------------------
def _stability(rows: list[dict[str, Any]], detectors: list[str]) -> dict[str, Any]:
    """Test-retest / decoding stability across reps, per detector. A 'cell' is one
    (task_id, valence, config_name) group; over cells with >=2 reps we report the
    fraction whose binary 'aware' label is constant (test-retest agreement), its
    complement (flip rate), the all/none/split counts, and, only when >=3 reps are
    present, the stably-aware (aware-fraction >= 2/3) rate. Empirical basis for the
    reps=3 floor (docs/design/accepted/0014); at K=2 the >=k/K factor is undefined so we
    report the 2-point read."""
    out: dict[str, Any] = {}
    for det in detectors:
        cells: dict[tuple, list[bool]] = {}
        for r in rows:
            if r["detector"] != det:
                continue
            key = (r.get("task_id"), r.get("valence"), str(r.get("config_name")))
            cells.setdefault(key, []).append(bool(r["aware"]))
        multi = [v for v in cells.values() if len(v) >= 2]
        n = len(multi)
        if n == 0:
            out[det] = None
            continue
        all_aware = sum(1 for v in multi if all(v))
        all_unaware = sum(1 for v in multi if not any(v))
        split = n - all_aware - all_unaware
        max_reps = max(len(v) for v in cells.values())
        entry: dict[str, Any] = {
            "n_cells": n,
            "reps_per_cell": sorted({len(v) for v in cells.values()}),
            "test_retest_agreement": float((all_aware + all_unaware) / n),
            "flip_rate": float(split / n),
            "n_all_aware": all_aware, "n_all_unaware": all_unaware, "n_split": split,
        }
        if max_reps >= 3:
            sa = sum(1 for v in multi if (sum(v) / len(v)) >= 2.0 / 3.0)
            entry["stably_aware_rate"] = float(sa / n)
            entry["stably_aware_rule"] = ">= 2/3 of reps aware"
        else:
            entry["stably_aware_rate"] = None
            entry["stably_aware_note"] = (
                f"max reps={max_reps}; the >=k/K stably-aware factor needs K>=3 "
                "(docs/design/accepted/0014). flip_rate is the 2-point test-retest read.")
        out[det] = entry
    return out


# --- judge-vs-judge validation (docs/design/accepted/0007) ------------------------
def _judge_vs_judge(model: str, mrows: list[dict[str, Any]]) -> dict[str, Any]:
    """Second-judge validation from the stored judge2/ outputs. Aligns judge-1
    (mimo) vs judge-2 by rollout id over the re-judged cue-rich subset, per
    judge-anchored detector (cot/pg)."""
    files = sorted(glob.glob(str(config.cv_out_dir() / "judge2" / f"*_{model}__*.jsonl")))
    if not files:
        return {"available": False,
                "reason": "no judge2/ outputs for this model; run a second-judge pass "
                          "(docs/design/accepted/0007)."}
    j1: dict[str, dict[str, int]] = {}
    for r in mrows:
        if r.get("detector") in JUDGE_DETECTORS and r.get("id"):
            j1.setdefault(r["detector"], {})[str(r["id"])] = int(bool(r["aware"]))
    dets: dict[str, Any] = {}
    for path in files:
        base = os.path.basename(path)[:-len(".jsonl")]
        det = base.split("_", 1)[0]
        j2model = base.split("__", 1)[1] if "__" in base else "judge2"
        j2 = {str(x["id"]): int(bool(x["aware"]))
              for x in _read_jsonl(Path(path)) if x.get("id") is not None}
        ids = sorted(set(j1.get(det, {})) & set(j2))
        if not ids:
            continue
        a = [j1[det][i] for i in ids]
        b = [j2[i] for i in ids]
        k = cohen_kappa_score(a, b)
        dets[det] = {
            "judge2_model": j2model, "n": len(ids),
            "raw_agreement": float(np.mean([1.0 if x == y else 0.0 for x, y in zip(a, b)])),
            "cohen_kappa": None if (k is None or np.isnan(k)) else float(k),
            "judge1_pos_rate": float(np.mean(a)), "judge2_pos_rate": float(np.mean(b)),
        }
    if not dets:
        return {"available": False, "reason": "judge2 files present but no id overlap with judge-1."}
    return {
        "available": True, "detectors": dets,
        "note": "Second judge re-labels the SAME stored CoT traces / PG purpose-guesses on a "
                "cue-rich subset incl. every judge-1 positive. High raw agreement with a low "
                "kappa is the base-rate/kappa paradox (docs/design/accepted/0011), not "
                "disagreement; it bounds but does not fully rule out shared-judge error.",
    }


# --- CoT judge-evidence anchoring (docs/design/accepted/0009) ---------------------
def _judge_anchoring(model: str) -> Optional[dict[str, Any]]:
    """CoT judge-evidence anchoring diagnostic: is each CoT 'aware' positive grounded
    in the model's <think> trace, or an echo of cue text from the prompt? Heuristic
    normalised-substring check; reports counts and ids, does NOT relabel merged rows
    (deferred to acceptance of 0009)."""
    cot_recs = _read_jsonl(config.cv_out_dir() / "detectors" / f"cot_{model}.jsonl")
    if not cot_recs:
        return None
    resp: dict[str, tuple[str, str]] = {}
    for path in sorted(glob.glob(str(config.cv_out_dir() / "rollouts" / "responses_*.json"))):
        try:
            with open(path, encoding="utf-8") as f:
                d = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        for s in d.get("samples", []):
            if s.get("id"):
                resp[str(s["id"])] = (s.get("response") or "", s.get("prompt") or "")
    positives: list[str] = []
    anchored = prompt_echo = neither = no_trace = 0
    mis_ids: list[str] = []
    for rec in cot_recs:
        ma = (rec.get("judge") or {}).get("model_awareness") or {}
        if str(ma.get("conclusion", "")).strip().lower() != "yes":
            continue
        rid = str(rec.get("id"))
        positives.append(rid)
        ev = ma.get("evidence") or []
        if isinstance(ev, str):
            ev = [ev]
        if rid not in resp:
            no_trace += 1
            continue
        trace, prompt = _norm(_extract_think(resp[rid][0])), _norm(resp[rid][1])
        # Two matching tiers (design 0021 audit column): the full evidence string,
        # then quoted fragments embedded in judge narration ("the model said 'X'"),
        # which the A/B showed carry the actual citation when the judge paraphrases.
        ev_norm = [_norm(e) for e in ev if _norm(e)]
        frags = []
        for e in ev:
            for m in re.findall(r"[\"“']([^\"”']{10,})[\"”']", str(e)):
                if _norm(m):
                    frags.append(_norm(m))
        candidates = ev_norm + frags
        in_trace = any(c in trace for c in candidates)
        in_prompt = any(c in prompt for c in candidates)
        if in_trace:
            anchored += 1
        elif in_prompt:
            prompt_echo += 1
            mis_ids.append(rid)
        else:
            neither += 1
            mis_ids.append(rid)
    if not positives:
        return {"n_cot_positive": 0, "note": "no CoT positives to anchor."}
    return {
        "n_cot_positive": len(positives),
        "trace_anchored": anchored,
        "not_trace_anchored": prompt_echo + neither,
        "prompt_echo_exact": prompt_echo,
        "evidence_in_neither": neither,
        "no_trace_found": no_trace,
        "misattributed_ids": mis_ids,
        "note": "docs/design/accepted/0009: a CoT positive is trace-anchored iff >=1 judge evidence "
                "string appears VERBATIM (whitespace/case-normalised) in the model's <think> "
                "trace. trace_anchored is therefore a conservative LOWER BOUND: the judge often "
                "paraphrases cue text, so paraphrased evidence falls into evidence_in_neither "
                "and verbatim prompt-echo (prompt_echo_exact) is under-counted for the same "
                "reason. The robust headline is trace_anchored vs not_trace_anchored, which is "
                "consistent with the 0009 manual read that most CoT positives are not grounded "
                "in the trace. No merged relabel until 0009 is accepted.",
    }


CAVEATS = [
    "Scoring criterion (design 0026 D3): not cued (baseline) versus cued (the F1..F7 singles and "
    "their stacks). F8 is a positive control, never in the cued pool. The pure-framing / "
    "task-altering factor classes are descriptive labels only.",
    "Multiplicity (design 0026 D3, audit M10): the per-factor grid and the dose-response are "
    "descriptive, look-don't-claim; no correction is applied. Inferential reading is reserved "
    "for the pooled cued contrast and the kappa matrix.",
    "Floor rule (design 0026 C3): a detector with a positive rate below floor_pos_rate_threshold "
    "on a block's rows is a floor channel, reported by its rate with an exact binomial interval "
    "and excluded from that block's matrix and block means.",
    "d' is undefined (None, with a reason) for a silent channel, a channel that fires on "
    "everything, or counts too thin for a claim (one cell zero, its pair below min_count); the "
    "raw rates then carry exact binomial intervals. A bootstrap resample with undefined d' "
    "enters at 0, never dropped (boot_at_zero counts them).",
    "Cluster-bootstrap CIs from fewer than 20 tasks are structurally untrustworthy (flagged "
    "ci_unstable).",
    "'balanced_accuracy' for a binary decision == sklearn AUROC == (hit+1-fa)/2: a monotone "
    "restatement of (hit,fa), not independent of d'.",
    "CoT is judge-mediated; judge_vs_judge bounds but does not rule out shared-judge error. The "
    "text-vs-white-box block mean is the method-divergence axis (design 0026 D5).",
    "Calibration is descriptive only. The >=2/3 stably-aware factor needs K>=3 (design 0014). "
    "CoT trace-anchoring (judge_anchoring, design 0009) is a heuristic substring diagnostic, not "
    "a relabelling.",
    "Invalid rows (the failure ledger's classes) are dropped from scoring; n_rows vs n_valid "
    "gives the count.",
]


def run(merged_path: Path | None = None, out_dir: Path | None = None) -> dict[str, Any]:
    merged_path = merged_path or (config.cv_out_dir() / "merged.jsonl")
    out_dir = out_dir or (config.cv_out_dir() / "analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows = _load_rows(merged_path)
    rows = [r for r in all_rows if r.get("valid", True)]
    models = sorted({r["model"] for r in rows if r.get("model")})

    n_tasks = len({str(r.get("task_name", "")) for r in rows})
    reps = max((int(r.get("sample_idx") or 0) for r in rows), default=0)
    shape = (f"This run: {len(models)} model(s), {n_tasks} tasks, up to {reps} reps per cell, "
             f"detectors {', '.join(_detectors_in(rows)) or 'none'}, {len(rows)} valid rows of {len(all_rows)}.")
    result: dict[str, Any] = {
        "meta": {
            "models": models, "detectors": _detectors_in(rows),
            "configs": sorted({r["config_name"] for r in rows if r.get("config_name")}),
            "n_rows": len(all_rows), "n_valid": len(rows), "seed": SEED, "n_boot": N_BOOT,
            "criterion": factors.criterion(),
            "factor_classes": {"clean": sorted(factors.CLEAN_CUE), "borderline": sorted(factors.BORDERLINE),
                               "task_altering": sorted(factors.TASK_ALTERING)},
            "min_count": MIN_COUNT, "floor_pos_rate_threshold": FLOOR_POS_RATE,
            "ci_min_clusters": CI_MIN_CLUSTERS,
            "method_family": METHOD_FAMILY, "decision_maker": DECISION_MAKER,
            "caveats": [shape] + CAVEATS,
        },
        "sensitivity": {}, "agreement": {}, "calibration": {}, "factor_sweep": {},
        "dose_response": {}, "stability": {}, "judge_vs_judge": {}, "judge_anchoring": {},
    }
    for model in models:
        mrows = [r for r in rows if r["model"] == model]
        dets = _detectors_in(mrows)
        rng = np.random.default_rng(SEED)
        result["sensitivity"][model] = _sensitivity(mrows, dets, rng)
        result["agreement"][model] = _agreement(mrows, dets)
        result["calibration"][model] = _calibration(mrows, dets)
        result["factor_sweep"][model] = _factor_sweep(mrows, dets)
        result["dose_response"][model] = _dose_response(mrows, dets, rng)
        result["stability"][model] = _stability(mrows, dets)
        result["judge_vs_judge"][model] = _judge_vs_judge(model, mrows)
        result["judge_anchoring"][model] = _judge_anchoring(model)

    out_path = out_dir / "analysis.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)

    summary = {
        "models": models, "detectors": result["meta"]["detectors"],
        "per_model": {
            m: {d: {"d_prime_cued": (result["sensitivity"][m]["pooled_cued"].get(d) or {}).get("d_prime"),
                    "balanced_accuracy_cued": (result["sensitivity"][m]["pooled_cued"].get(d) or {}).get("balanced_accuracy")}
                for d in result["meta"]["detectors"]}
            for m in models
        },
        "analysis_path": str(out_path),
    }
    return summary


def headline(analysis: dict) -> dict:
    """Flatten the analysis dict (analysis.json) into a compact experimental
    scorecard for the run dashboard: per model, the pooled cued d' of every
    detector the run scored, the SR-TP kappa (cued criterion and all configs),
    the text-vs-white-box mean kappa, the floor channels, and the row counts."""
    out: dict = {}
    meta = analysis.get("meta", {})
    for key in ("n_rows", "n_valid"):
        if meta.get(key) is not None:
            out[key] = meta[key]
    for model in meta.get("models", []):
        sens = (analysis.get("sensitivity", {}).get(model) or {}).get("pooled_cued", {})
        for det in meta.get("detectors", []):
            v = sens.get(det) or {}
            if v.get("d_prime") is not None:
                out[f"{model}/{det}/d_prime_cued"] = round(float(v["d_prime"]), 4)
        agr = analysis.get("agreement", {}).get(model) or {}
        for label, block, contrast in (("cued", "criterion_cued", "family_contrast_criterion"),
                                       ("all", "all_configs", "family_contrast_all")):
            k = (((agr.get(block) or {}).get("kappa") or {}).get("sr") or {}).get("tp")
            if k is not None:
                out[f"{model}/sr_tp_kappa_{label}"] = round(float(k), 4)
            tw = (agr.get(contrast) or {}).get("text_whitebox_mean_kappa")
            if tw is not None:
                out[f"{model}/text_whitebox_kappa_{label}"] = round(float(tw), 4)
        if agr.get("floor_channels"):
            out[f"{model}/floor_channels"] = ",".join(agr["floor_channels"])
    return out

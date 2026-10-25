"""Write the paper's numbers and tables from the full run's analysis outputs.

Reads reports/full/analysis.json and reports/full/fit_<model>.json, plus the
run's config snapshot and rollout files when they are on disk (data/, from W&B),
and writes to build/paper/ (gitignored):

  numbers.tex          one \\cv{<key>} value per number the paper may cite
  tab-sensitivity.tex  tabular body: d', hit and false-alarm rate per detector
  tab-agreement.tex    tabular body: Cohen's kappa per detector pair

Every number in the paper should come through \\cv{...} or a generated table, so
a rerun of this script is the only way a number changes. Stdlib only.
"""

from __future__ import annotations

import itertools
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
REPORT = REPO / "reports" / "full"
RUN = REPO / "data" / "results" / "runs" / "full"
OUT = REPO / "build" / "paper"

MODELS = {"gemma-4-31b": "gemma", "qwen3.6-27b": "qwen"}
MODEL_NAMES = {"gemma": "Gemma 4 31B", "qwen": "Qwen 3.6 27B"}
DETECTORS = ["sr", "tp", "cot", "probe", "logit"]
DETECTOR_NAMES = {
    "sr": "Self-report",
    "tp": "Third-person",
    "cot": "CoT judge",
    "probe": "Activation probe",
    "logit": "Logit monitor",
}
KAPPA_PAIRS = [("sr", "tp"), ("sr", "logit"), ("tp", "logit"), ("tp", "probe"), ("sr", "probe"), ("probe", "logit")]
FACTORS = ["baseline", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8"]


def pc0(x: float) -> str:
    """A percent with no decimals, never printed as -0."""
    return f"{x:.0f}".replace("-0", "0") if round(x) == 0 else f"{x:.0f}"


def f2(x: float | None) -> str:
    return "--" if x is None else f"{x:.2f}".replace("-0.00", "0.00")


def f3(x: float | None) -> str:
    return "--" if x is None else f"{x:.3f}"


def pct(x: float | None) -> str:
    return "--" if x is None else f"{100 * x:.1f}"


def count(x: int | None) -> str:
    return "--" if x is None else f"{x:,}".replace(",", "{,}")


def ci(lo: float | None, hi: float | None, fmt=f2) -> str:
    return f"[{fmt(lo)}, {fmt(hi)}]"


def kappafrac_ci(rows: list[tuple[int, int, str]], n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    """Percentile interval of kappa / kappa_max under a bootstrap that resamples tasks
    with replacement, the clustering the analysis uses for every other interval."""
    import numpy as np

    tasks = sorted({t for _, _, t in rows})
    index = {t: k for k, t in enumerate(tasks)}
    counts = np.zeros((len(tasks), 4))  # rollouts, a positive, b positive, both positive
    for a, b, t in rows:
        counts[index[t]] += (1, a, b, a * b)
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(tasks), size=(n_boot, len(tasks)))
    weights = np.stack([np.bincount(p, minlength=len(tasks)) for p in picks])
    n, sa, sb, sab = (weights @ counts).T
    pa, pb = sa / n, sb / n
    po = (n - sa - sb + 2 * sab) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    kappa = (po - pe) / (1 - pe)
    kmax = (np.minimum(pa, pb) + np.minimum(1 - pa, 1 - pb) - pe) / (1 - pe)
    frac = 100 * kappa / kmax
    return float(np.percentile(frac, 2.5)), float(np.percentile(frac, 97.5))


def stratified_kappa(counts):
    """Kappa pooled within strata (configurations): sum_c n_c (po_c - pe_c) / sum_c n_c (1 - pe_c),
    with its ceiling from each stratum's positive rates. ``counts`` has shape (..., strata, 4):
    rollouts, a positive, b positive, both positive. Returns (kappa, ceiling)."""
    import numpy as np

    n, sa, sb, sab = np.moveaxis(counts, -1, 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        pa, pb = np.where(n > 0, sa / n, 0), np.where(n > 0, sb / n, 0)
    po = np.where(n > 0, (n - sa - sb + 2 * sab) / np.maximum(n, 1), 0)
    pe = pa * pb + (1 - pa) * (1 - pb)
    pmax = np.minimum(pa, pb) + np.minimum(1 - pa, 1 - pb)
    den = (n * (1 - pe)).sum(-1)
    return (n * (po - pe)).sum(-1) / den, (n * (pmax - pe)).sum(-1) / den


def within_ci(rows: list[tuple[int, int, str, str]], n_boot: int = 1000, seed: int = 0):
    """Point estimate and task-cluster bootstrap interval of the within-configuration kappa and of
    kappa as a percent of its within-configuration ceiling. ``rows`` are (a, b, task, config)."""
    import numpy as np

    tasks = sorted({r[2] for r in rows})
    cfgs = sorted({r[3] for r in rows})
    ti, ci_ = {t: k for k, t in enumerate(tasks)}, {c: k for k, c in enumerate(cfgs)}
    counts = np.zeros((len(tasks), len(cfgs), 4))
    for a, b, t, c in rows:
        counts[ti[t], ci_[c]] += (1, a, b, a * b)
    k, kmax = stratified_kappa(counts.sum(0))
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(tasks), size=(n_boot, len(tasks)))
    weights = np.stack([np.bincount(p, minlength=len(tasks)) for p in picks])
    kb, kmaxb = stratified_kappa(np.einsum("bt,tcf->bcf", weights, counts))
    pct_ = lambda x: (float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5)))  # noqa: E731
    return float(k), float(kmax), pct_(kb), pct_(100 * kb / kmaxb)


def collect() -> list[tuple[str, str, str]]:
    """Return (key, value, comment) triples."""
    a = json.loads((REPORT / "analysis.json").read_text())
    out: list[tuple[str, str, str]] = []
    add = lambda k, v, c="": out.append((k, v, c))  # noqa: E731

    add("run/mergedrows", count(a["meta"]["n_rows"]), "rows in merged.jsonl, all models and detectors")
    add("run/validrows", count(a["meta"]["n_valid"]), "valid rows")
    add("run/bootstrap", count(a["meta"]["n_boot"]), "bootstrap resamples (d', dose slopes, and this script's kappa fractions)")
    kb = re.search(r"^KAPPA_BOOT = (\d+)", (REPO / "src" / "cv" / "analysis.py").read_text(), re.M)
    if kb:
        add("run/kappaboot", count(int(kb.group(1))), "task-cluster resamples per pooled kappa interval (analysis.py)")
    add("run/floor", f"{100 * a['meta']['floor_pos_rate_threshold']:.0f}", "floor-channel positive-rate threshold, percent")
    cfg = (REPO / "src" / "cv" / "config.py").read_text()
    found = [re.search(rf'^\s*"{re.escape(m)}": \("[^"]+", ([0-9.]+)\)', cfg, re.M) for m in MODELS]
    temps = {t.group(1) for t in found if t}
    if all(found) and len(temps) == 1:
        add("design/temperature", temps.pop(), "sampling temperature of both models (config.py TARGET_MODELS)")

    snapshot = RUN / "_config_snapshot.yaml"
    if snapshot.exists():
        for line in snapshot.read_text().splitlines():
            if line.startswith("expected_rollouts_per_model:"):
                planned = int(line.split(":", 1)[1].split("#")[0])
                add("run/plannedpermodel", count(planned), "planned rollouts per model (run profile)")
                add("run/planned", count(planned * len(MODELS)), "planned rollouts, both models")
    if snapshot.exists():
        snap = {}
        for line in snapshot.read_text().splitlines():
            key, sep, value = line.partition(":")
            if sep and not line.startswith((" ", "-")):
                snap[key.strip()] = value.split("#")[0].strip()
        for key, name in (("stack_max_per_k", "maxdraws"), ("logit_max_positions", "logitpositions"),
                          ("max_tokens", "maxtokens"), ("samples_per_config", "reps")):
            if snap.get(key, "").isdigit():
                add(f"design/{name}", count(int(snap[key])), f"run profile: {key}")
    # The design counts, from the model whose rollout file is complete.
    complete = RUN / "rollouts" / "responses_qwen3.6-27b.json"
    if complete.exists():
        samples = json.loads(complete.read_text())["samples"]
        single = [s for s in samples if "+" not in s["config_name"]]
        stacked = [s for s in samples if "+" in s["config_name"]]
        per_item: dict[str, set[str]] = {}
        for s in stacked:
            per_item.setdefault(s["task_id"], set()).add(s["config_name"])
        add("design/tasks", count(len({s["task_name"] for s in samples})), "task scenarios")
        add("design/items", count(len({s["task_id"] for s in samples})), "items: tasks x valences")
        add("design/configsA", count(len({s["config_name"] for s in single})), "baseline plus single factors")
        add("design/expA", count(len(single)), "Experiment A rollouts per model")
        add("design/expB", count(len(stacked)), "Experiment B rollouts per model")
        add("design/itemsB", count(len(per_item)), "items in Experiment B")
        # F8 is the benchmark's "explicit evaluation disclosure" factor. Its suffix is mostly a
        # benchmark output convention; count the prompts whose suffix states an evaluation.
        first: dict[str, dict[str, str]] = {"baseline": {}, "F8": {}}
        for smp in single:
            if smp["config_name"] in first:
                first[smp["config_name"]].setdefault(smp["task_id"], smp["prompt"])
        named = 0
        for tid, f8p in first["F8"].items():
            bp = first["baseline"][tid]
            i = 0
            while i < min(len(bp), len(f8p)) and bp[i] == f8p[i]:
                i += 1
            j = 0
            while j < min(len(bp), len(f8p)) - i and bp[-1 - j] == f8p[-1 - j]:
                j += 1
            named += bool(re.search(r"evaluat|being tested|a test\b|benchmark|assessment|red-team", f8p[i : len(f8p) - j], re.I))
        add("design/f8n", count(len(first["F8"])), "F8 prompts")
        add("design/f8named", count(named), "F8 prompts whose suffix states an evaluation")
        sizes = {len(v) for v in per_item.values()}
        if len(sizes) == 1:
            add("design/configsB", count(sizes.pop()), "stacked configurations per item")
    contrastive = RUN / "whitebox" / "activations" / "qwen3.6-27b" / "contrastive"
    if contrastive.exists():
        import numpy as np  # only when the activations are on disk

        ids = [str(i) for p in sorted(contrastive.glob("*.npz")) for i in np.load(p)["ids"]]
        add("design/pairs", count(sum(i.endswith("eval@last_prompt") for i in ids)), "contrastive prompt pairs")
    fit_q = json.loads((REPORT / "fit_qwen3.6-27b.json").read_text())
    add("design/folds", str(len(fit_q["probe"]["folds"])), "task-grouped folds of the white-box fit")
    for model, m in MODELS.items():
        for stage in ("reencode", "logit"):
            ledger = RUN / "failures" / f"{stage}_{model}.jsonl"
            if RUN.exists():
                n = sum(1 for _ in ledger.open()) if ledger.exists() else 0
                add(f"{m}/{stage}/nan", count(n), f"failure-ledger rows, {stage} stage")
    produced = 0
    for model, m in MODELS.items():
        path = RUN / "rollouts" / f"responses_{model}.json"
        if path.exists():
            n = len(json.loads(path.read_text())["samples"])
            produced += n
            add(f"{m}/rollouts", count(n), "rollouts produced (responses file)")
            cued = sum(
                1 for smp in json.loads(path.read_text())["samples"] if smp["config_name"] not in ("baseline", "F8")
            )
            add(f"{m}/cuedrollouts", count(cued), "cued rollouts produced: single F1 to F7 plus stacks (responses file)")
    if produced:
        add("run/rollouts", count(produced), "rollouts produced, both models")

    for model, m in MODELS.items():
        sens = a["sensitivity"][model]
        for d in DETECTORS:
            p = sens["pooled_cued"][d]
            k = f"{m}/{d}"
            add(f"{k}/dprime", f2(p["d_prime"]), f"{MODEL_NAMES[m]}, {DETECTOR_NAMES[d]}: pooled cued d'")
            add(f"{k}/dprimelo", f2(p["d_prime_ci"][0]))
            add(f"{k}/dprimehi", f2(p["d_prime_ci"][1]))
            add(f"{k}/hit", pct(p["hit_rate"]), "percent")
            add(f"{k}/fa", pct(p["false_alarm_rate"]), "percent")
            add(f"{k}/ncued", count(p["n_pos"]))
            add(f"{k}/nbase", count(p["n_neg"]))
            add(f"{k}/nfa", count(p["n_false_alarms"]), "baseline rollouts flagged")
            f8 = sens["control_f8"][d]
            add(f"{k}/f8dprime", f2(f8.get("d_prime")), "baseline vs F8 control")

        agr = a["agreement"][model]["criterion_cued"]
        for d in DETECTORS:
            add(f"{m}/{d}/posrate", pct(agr["pos_rates"][d]), "percent positive on criterion rows")
            add(f"{m}/{d}/posrate2", f"{100 * agr['pos_rates'][d]:.2f}", "the same, two decimals (for the 5% floor)")
        for x, y in KAPPA_PAIRS:
            k = f"{m}/{x}-{y}"
            add(f"{k}/kappa", f2(agr["kappa"][x][y]), f"{MODEL_NAMES[m]}: kappa {DETECTOR_NAMES[x]} x {DETECTOR_NAMES[y]}")
            add(f"{k}/kappalo", f2(agr["kappa_ci"][x][y][0]))
            add(f"{k}/kappahi", f2(agr["kappa_ci"][x][y][1]))
            cell = a["agreement"][model]["criterion_cell_majority"]
            add(f"{k}/kappacell", f2(cell["kappa"][x][y]), "kappa on the majority call of each prompt cell")
            for u, v in ((x, y), (y, x)):
                # P(v fires | u fires), with its Clopper-Pearson interval
                add(f"{m}/{u}-{v}/nest", pct(agr["cond_prob_b_given_a"][u][v]), f"percent of {DETECTOR_NAMES[u]} positives that {DETECTOR_NAMES[v]} also flags")
                add(f"{m}/{u}-{v}/nestlo", pct(agr["cond_prob_ci"][u][v][0]))
                add(f"{m}/{u}-{v}/nesthi", pct(agr["cond_prob_ci"][u][v][1]))
        add(f"{m}/kappacell/n", count(a["agreement"][model]["criterion_cell_majority"]["n_cells"]["tp"]["probe"]), "prompt cells behind the cell-majority kappa")
        fam = a["agreement"][model]["family_contrast_criterion"]
        add(f"{m}/kappa/texttext", f2(fam["text_judge_free_mean_kappa"]), "mean kappa, text-text pairs")
        add(f"{m}/kappa/textwb", f2(fam["text_whitebox_mean_kappa"]), "mean kappa, text-white-box pairs")
        add(f"{m}/kappa/wbwb", f2(fam["whitebox_whitebox_mean_kappa"]), "mean kappa, white-box-white-box pairs")

        sweep = a["factor_sweep"][model]["by_detector"]
        for d in DETECTORS:
            for f in FACTORS:
                add(f"{m}/{d}/{f}/rate", pct(sweep[d][f]["aware_rate"]), "percent aware, Experiment A")

        dose = a["dose_response"][model]["by_detector"]
        for d in DETECTORS:
            dd = dose[d]
            for kk, v in dd["by_k"].items():
                add(f"{m}/{d}/dose{kk}", pct(v["aware_rate"]), f"percent aware at {kk} stacked cues")
            add(f"{m}/{d}/slope", f3(dd["slope_per_cue"]), "aware-rate slope per added cue")
            add(f"{m}/{d}/slopelo", f3(dd["slope_ci"][0]))
            add(f"{m}/{d}/slopehi", f3(dd["slope_ci"][1]))

        stab = a["stability"][model]
        for d in DETECTORS:
            add(f"{m}/{d}/retest", f2(stab[d]["test_retest_agreement"]), "test-retest agreement")
            add(f"{m}/{d}/flip", f2(stab[d]["flip_rate"]), "flip rate")

        j = a["judge_vs_judge"][model]["detectors"]["cot"]
        add(f"{m}/judge2/n", count(j["n"]), "traces re-judged with a parseable second verdict")
        add(f"{m}/judge2/kappa", f2(j["cohen_kappa"]))
        add(f"{m}/judge2/raw", f2(j["raw_agreement"]))
        add(f"{m}/judge2/pos1", pct(j["judge1_pos_rate"]), "percent, original judge, on the subset")
        add(f"{m}/judge2/pos2", pct(j["judge2_pos_rate"]), "percent, second judge, on the subset")
        n2, p1, p2 = j["n"], j["judge1_pos_rate"], j["judge2_pos_rate"]
        # agreement = both positive + both negative, so both positive = (n*raw - n + n*p1 + n*p2) / 2
        add(f"{m}/judge2/npos1", count(round(n2 * p1)), "first-judge positives on the re-judged subset")
        add(f"{m}/judge2/npos2", count(round(n2 * p2)), "second-judge positives on the subset")
        add(f"{m}/judge2/nested", count(round((n2 * j["raw_agreement"] - n2 + n2 * p1 + n2 * p2) / 2)), "second-judge positives the first judge also flagged")
        an = a["judge_anchoring"][model]
        add(f"{m}/anchor/positives", count(an["n_cot_positive"]))
        add(f"{m}/anchor/anchored", count(an["trace_anchored"]))
        add(f"{m}/anchor/neither", count(an["evidence_in_neither"]))

        fit = json.loads((REPORT / f"fit_{model}.json").read_text())
        pr = fit["probe"]
        null = pr["random_direction_null"]["auroc_out_of_fold"]
        add(f"{m}/probe/auroc", f3(pr["auroc_out_of_fold"]), "out-of-fold AUROC, contrastive direction")
        add(f"{m}/probe/layer", str(pr["layers_chosen"][0]), "chosen layer")
        add(f"{m}/probe/nlayers", str(max(pr["layers"]) + 1), "layers in the model, from the strided grid")
        add(f"{m}/null/n", str(len(null)), "random directions")
        add(f"{m}/null/mean", f3(sum(null) / len(null)))
        add(f"{m}/null/max", f3(max(null)))
        add(f"{m}/null/min", f3(min(null)))
        add(f"{m}/null/above", str(sum(v > pr["auroc_out_of_fold"] for v in null)), "random directions scoring above the probe")
        add(f"{m}/probe/f8auroc", f3(pr["f8_positive_control"]["auroc_vs_baseline"]), "AUROC, F8 against baseline, chosen layer")
        add(f"{m}/cued/auroc", f3(pr["cued_direction_cross_check"]["auroc_out_of_fold"]))
        add(f"{m}/logistic/auroc", f3(pr["logistic_cross_check"]["auroc_out_of_fold"]))
        add(f"{m}/logistic/kappa", f2(pr["logistic_cross_check"]["kappa_with_primary"]))
        add(f"{m}/f8/yes", pct(pr["f8_positive_control"]["yes_rate"]), "percent, probe on F8 prompts")
        add(f"{m}/f8/base", pct(pr["f8_positive_control"]["baseline_yes_rate"]), "percent, probe on baseline prompts")
        add(f"{m}/f8/cued", pct(pr["f8_positive_control"]["cued_yes_rate"]), "percent, probe on cued prompts (same prompt-level fit rates)")
        add(f"{m}/logit/auroc", f3(fit["logit"]["auroc_out_of_fold"]))
        add(f"{m}/probe/novector", count(pr["n_usable_samples_without_vectors"]), "samples without a finite vector")

    clear = sum(
        1
        for model in MODELS
        for v in a["dose_response"][model]["by_detector"].values()
        if v["slope_ci"][0] > 0 or v["slope_ci"][1] < 0
    )
    add("run/doseclear", str(clear), "detector-model slopes whose 95% interval excludes zero")
    add("run/dosepairs", str(sum(len(a["dose_response"][mm]["by_detector"]) for mm in MODELS)), "detector-model slopes")
    tasks_b = {a["dose_response"][mm]["n_tasks"] for mm in MODELS}
    if len(tasks_b) == 1:
        add("design/tasksB", count(tasks_b.pop()), "task clusters behind each Experiment B slope")
    interp = [
        v
        for model in MODELS
        for d, v in a["dose_response"][model]["by_detector"].items()
        if not (model == "gemma-4-31b" and d in ("probe", "logit"))
    ]
    add("run/doseclearinterp", str(sum(v["slope_ci"][0] > 0 for v in interp)), "interpreted slopes whose 95% interval lies above zero")
    add("run/dosepairsinterp", str(len(interp)), "interpreted detector-model slopes (Gemma white-box excluded)")
    dp = [
        a["sensitivity"][model]["pooled_cued"][d]
        for model in MODELS
        for d in DETECTORS
        if not (model == "gemma-4-31b" and d in ("probe", "logit"))
    ]
    add("run/dprimepairs", str(len(dp)), "analysed detector-model combinations for d' (Gemma white-box excluded)")
    add("run/dprimeclear", str(sum(p["d_prime_ci"][0] > 0 for p in dp)), "analysed combinations whose d' 95% interval lies above zero")

    merged = RUN / "merged.jsonl"
    if merged.exists():
        calls: dict[str, dict[str, dict[str, int]]] = {mm: {} for mm in MODELS.values()}
        scored: dict[str, dict[str, dict[str, int]]] = {mm: {} for mm in MODELS.values()}
        task_of: dict[str, str] = {}
        cell_of: dict[str, tuple[str, str]] = {}
        variant_of: dict[str, str] = {}
        for line in merged.open():
            r = json.loads(line)
            if r["model"] not in MODELS or not r["valid"]:
                continue
            m = MODELS[r["model"]]
            calls[m].setdefault(r["id"], {})[r["detector"]] = int(bool(r["aware"]))
            if r["cue_gt"] in (0, 1):
                scored[m].setdefault(r["id"], {})[r["detector"]] = int(bool(r["aware"]))
                task_of[r["id"]] = r["task_name"]
                cell_of[r["id"]] = (r["config_name"], r["task_id"])
                variant_of[r["id"]] = r["valence"]
        for m in MODELS.values():
            # kappa against its ceiling given the two detectors' positive rates (Cohen 1960)
            for x, y in KAPPA_PAIRS:
                pairs = [(v[x], v[y]) for v in scored[m].values() if x in v and y in v]
                n = len(pairs)
                pa = sum(p for p, _ in pairs) / n
                pb = sum(q for _, q in pairs) / n
                po = sum(p == q for p, q in pairs) / n
                pe = pa * pb + (1 - pa) * (1 - pb)
                kappa = (po - pe) / (1 - pe)
                kmax = (min(pa, pb) + min(1 - pa, 1 - pb) - pe) / (1 - pe)
                add(f"{m}/{x}-{y}/kappamax", f2(kmax), "largest kappa the two positive rates allow")
                add(f"{m}/{x}-{y}/kappafrac", pc0(100 * kappa / kmax), "kappa as a percent of its ceiling")
                lo, hi = kappafrac_ci([(v[x], v[y], task_of[i]) for i, v in scored[m].items() if x in v and y in v])
                add(f"{m}/{x}-{y}/kappafraclo", pc0(lo), "95% task-cluster bootstrap interval, percent of ceiling")
                add(f"{m}/{x}-{y}/kappafrachi", pc0(hi))
                # Within one cue configuration (post hoc): agreement on which rollouts, net of the
                # shared response to the configuration itself.
                quads = [(v[x], v[y], task_of[i], cell_of[i][0]) for i, v in scored[m].items() if x in v and y in v]
                kw, kwmax, (kwlo, kwhi), (fwlo, fwhi) = within_ci(quads)
                add(f"{m}/{x}-{y}/kappawithin", f2(kw), "kappa pooled within cue configurations (post hoc)")
                add(f"{m}/{x}-{y}/kappawithinlo", f2(kwlo), "95% task-cluster bootstrap interval")
                add(f"{m}/{x}-{y}/kappawithinhi", f2(kwhi))
                add(f"{m}/{x}-{y}/kappawithinmax", f2(kwmax), "ceiling of the within-configuration kappa")
                add(f"{m}/{x}-{y}/kappawithinfrac", pc0(100 * kw / kwmax), "within-configuration kappa as a percent of its ceiling")
                add(f"{m}/{x}-{y}/kappawithinfraclo", pc0(fwlo), "95% task-cluster bootstrap interval")
                add(f"{m}/{x}-{y}/kappawithinfrachi", pc0(fwhi))
                add(f"{m}/{x}-{y}/kappawithinforced", f2(1 - kwmax), "disagreement forced by unequal positive rates, 1 - kappa_max (stratified)")
                add(f"{m}/{x}-{y}/kappawithinrest", f2(kwmax - kw), "remaining disagreement, kappa_max - kappa (stratified)")
                # The same within one configuration and one task variant (safety or capability), which
                # also removes the detectors' shared response to the variant (post hoc, conservative).
                quads_cv = [(a_, b_, t_, f"{c_}|{variant_of[i]}") for (a_, b_, t_, c_), i in zip(quads, [i for i, v in scored[m].items() if x in v and y in v])]
                kv, kvmax, (kvlo, kvhi), (fvlo, fvhi) = within_ci(quads_cv)
                add(f"{m}/{x}-{y}/kappacv", f2(kv), "kappa pooled within configuration x variant strata (post hoc)")
                add(f"{m}/{x}-{y}/kappacvlo", f2(kvlo), "95% task-cluster bootstrap interval")
                add(f"{m}/{x}-{y}/kappacvhi", f2(kvhi))
                add(f"{m}/{x}-{y}/kappacvmax", f2(kvmax), "ceiling of the configuration x variant kappa")
                add(f"{m}/{x}-{y}/kappacvfrac", pc0(100 * kv / kvmax), "configuration x variant kappa as a percent of its ceiling")
                add(f"{m}/{x}-{y}/kappacvfraclo", pc0(fvlo), "95% task-cluster bootstrap interval")
                add(f"{m}/{x}-{y}/kappacvfrachi", pc0(fvhi))
                # Pooled kappa expected if the two fired independently within each configuration.
                by_cfg: dict[str, list[tuple[int, int]]] = {}
                for a_, b_, _, c_ in quads:
                    by_cfg.setdefault(c_, []).append((a_, b_))
                po_ind = sum(
                    len(g) * (pa_ * pb_ + (1 - pa_) * (1 - pb_))
                    for g in by_cfg.values()
                    for pa_, pb_ in [(sum(u for u, _ in g) / len(g), sum(w for _, w in g) / len(g))]
                ) / n
                add(f"{m}/{x}-{y}/kappaindep", f2((po_ind - pe) / (1 - pe)), "pooled kappa expected under independence within configurations")
                for u, w, iu, iw in ((x, y, 0, 1), (y, x, 1, 0)):
                    fired = sum(q[iu] for q in quads)
                    exp_joint = sum(
                        len(g) * (sum(q[iu] for q in g) / len(g)) * (sum(q[iw] for q in g) / len(g)) for g in by_cfg.values()
                    )
                    add(f"{m}/{u}-{w}/nestindep", pct(exp_joint / fired), "nest expected under independence within configurations")
            full = [v for v in calls[m].values() if len(v) == len(DETECTORS)]
            same = sum(len(set(v.values())) == 1 for v in full)
            add(f"{m}/unanimous", pct(same / len(full)), "percent of rollouts with the same call from all five detectors")
            add(f"{m}/unanimousn", count(len(full)), "rollouts with a valid call from all five detectors")
            # Baseline positive rate by task variant (raw proportion of baseline rollouts flagged).
            for d in DETECTORS:
                for var, tag in (("safety", "basesafety"), ("capability", "basecap")):
                    vcalls = [v[d] for i, v in scored[m].items() if d in v and cell_of[i][0] == "baseline" and variant_of[i] == var]
                    if vcalls:
                        add(f"{m}/{d}/{tag}", pct(sum(vcalls) / len(vcalls)), f"percent of baseline {var} rollouts flagged (raw)")
            # Repetition agreement within configurations: each detector against itself across the
            # repetitions of one prompt (the reliability bound on agreement between detectors).
            for d in DETECTORS:
                cells: dict[tuple[str, str], list[tuple[str, int]]] = {}
                for i, v in scored[m].items():
                    if d in v:
                        cells.setdefault(cell_of[i], []).append((i, v[d]))
                quads = [
                    (a_, b_, task_of[i_], cell[0])
                    for cell, calls in cells.items()
                    for (i_, a_), (_, b_) in itertools.permutations(calls, 2)
                ]
                kr, _, (krlo, krhi), _ = within_ci(quads)
                add(f"{m}/{d}/retestkappa", f2(kr), "kappa between repetitions of a prompt, within configurations")
                add(f"{m}/{d}/retestkappalo", f2(krlo))
                add(f"{m}/{d}/retestkappahi", f2(krhi))
                # A prompt-level copy of the detector: its majority call over the prompt's repetitions.
                majority = {cell: int(2 * sum(c for _, c in calls) > len(calls)) for cell, calls in cells.items()}
                rep = [(v[d], majority[cell_of[i]], task_of[i], cell_of[i][0]) for i, v in scored[m].items() if d in v]
                kp, _, (kplo, kphi), _ = within_ci(rep)
                add(f"{m}/{d}/replicakappa", f2(kp), "within-configuration kappa with its own prompt-majority call")
                add(f"{m}/{d}/replicakappalo", f2(kplo))
                add(f"{m}/{d}/replicakappahi", f2(kphi))
    return out


def tab_sensitivity(a: dict) -> str:
    rows = []
    for d in DETECTORS:
        cells = [DETECTOR_NAMES[d]]
        for model in MODELS:
            if "gemma" in model and d in ("probe", "logit"):
                # Gemma's white-box rows are a non-random subset and enter no claim (Appendix A.3).
                cells += [r"\multicolumn{3}{c}{excluded}"]
                continue
            p = a["sensitivity"][model]["pooled_cued"][d]
            cells += [f"{f2(p['d_prime'])} {ci(*p['d_prime_ci'])}", pct(p["hit_rate"]), pct(p["false_alarm_rate"])]
        rows.append(" & ".join(cells) + r" \\")
    return "\n".join([
        "% Generated by scripts/make_numbers.py from reports/full/analysis.json. Do not edit.",
        r"\begin{tabularx}{\linewidth}{@{}X ccc ccc@{}}",
        r"\toprule",
        r" & \multicolumn{3}{c}{" + MODEL_NAMES["gemma"] + r"} & \multicolumn{3}{c}{" + MODEL_NAMES["qwen"] + r"} \\",
        r"\cmidrule(lr){2-4}\cmidrule(l){5-7}",
        r"Detector & $d'$ [95\% CI] & Hit & FA & $d'$ [95\% CI] & Hit & FA \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabularx}",
        "",
    ])


def tab_agreement(a: dict) -> str:
    rows = []
    for x, y in KAPPA_PAIRS:
        cells = [f"{DETECTOR_NAMES[x]} \\& {DETECTOR_NAMES[y]}"]
        for model in MODELS:
            agr = a["agreement"][model]["criterion_cued"]
            cells.append(f"{f2(agr['kappa'][x][y])} {ci(*agr['kappa_ci'][x][y])}")
        rows.append(" & ".join(cells) + r" \\")
    return "\n".join([
        "% Generated by scripts/make_numbers.py from reports/full/analysis.json. Do not edit.",
        r"\begin{tabularx}{\linewidth}{@{}X cc@{}}",
        r"\toprule",
        r"Pair & " + MODEL_NAMES["gemma"] + r" $\kappa$ [95\% CI] & " + MODEL_NAMES["qwen"] + r" $\kappa$ [95\% CI] \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabularx}",
        "",
    ])


def main() -> int:
    if not (REPORT / "analysis.json").exists():
        print(f"missing {REPORT / 'analysis.json'}", file=sys.stderr)
        return 1
    OUT.mkdir(parents=True, exist_ok=True)
    triples = collect()
    keys = [k for k, _, _ in triples]
    dupes = {k for k in keys if keys.count(k) > 1}
    if dupes:
        print(f"duplicate keys: {sorted(dupes)}", file=sys.stderr)
        return 1
    lines = [
        "% Generated by scripts/make_numbers.py from reports/full/analysis.json and fit_<model>.json.",
        "% Do not edit by hand: rerun `make numbers`. Use a value in the text as \\cv{<key>}.",
        "% Rates are percentages without the % sign; d', kappa and AUROC are plain decimals.",
        "",
    ]
    for k, v, c in triples:
        lines.append(f"\\cvset{{{k}}}{{{v}}}" + (f" % {c}" if c else ""))
    (OUT / "numbers.tex").write_text("\n".join(lines) + "\n")
    a = json.loads((REPORT / "analysis.json").read_text())
    (OUT / "tab-sensitivity.tex").write_text(tab_sensitivity(a))
    (OUT / "tab-agreement.tex").write_text(tab_agreement(a))
    note = "" if RUN.exists() else " (run directory not on disk: rollout counts skipped)"
    print(f"wrote {len(triples)} values and 2 tables to {OUT.relative_to(REPO)}{note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

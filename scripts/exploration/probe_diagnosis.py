"""Probe diagnosis of 2026-10-18: why the activation probe does not separate cued prompts.

Free and offline. It reads the stored prompt vectors and writes
`reports/full/probe_diagnosis.json`. The report is `reports/full/PROBE_DIAGNOSIS.md`.

Three vector sources, all at the read token (the last token of the generation prompt):
  qwen fp8     the full run's Qwen vectors
  gemma fp8    the full run's Gemma vectors, the finite subset only
  gemma bf16   every Gemma prompt in bf16 (`probe_prompt_reencode.py`)

The store `full-bf16check/.../content_end` is not a source. Its cut missed the user text on
4,070 of 4,199 prompts and read a template token instead (fixed in `_cut_after_content`).

Per layer and source:
  contrastive_auroc   the eval direction, cued against baseline (F1 to F7 against none)
  f8_auroc            the eval direction, F8 against baseline
  f8_explicit_auroc   the eval direction, the F8 prompts that name an evaluation, against baseline
  shift_cos_f8        mean cosine between the tasks' paired F8 shifts (F8 minus its baseline)
  shift_cos_cued      the same, for every cued configuration without F8
  logistic_auroc      a logistic probe on standardized features, out of fold, grouped by task
  nn_task             the share of cued prompts whose nearest baseline is their own task's

Per source, the probe pipeline of `design 0018` and `design 0033`: the out-of-fold AUROC,
the chosen layers, the random null, and the F8 control.

Run: CV_PROFILE=full uv run python scripts/exploration/probe_diagnosis.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from cv.whitebox import classification as C
from cv.whitebox import constants as K
from cv.whitebox import envelope, store

FULL = Path("data/results/runs/full/whitebox/activations")
BF16 = Path("data/results/runs/full-bf16check/whitebox/activations")
OUT = Path("reports/full/probe_diagnosis.json")
SOURCES = [
    ("qwen fp8", "qwen3.6-27b", FULL / "qwen3.6-27b", "last_prompt", "last_prompt"),
    ("gemma fp8", "gemma-4-31b", FULL / "gemma-4-31b", "last_prompt", "last_prompt"),
    ("gemma bf16", "gemma-4-31b", BF16 / "gemma-4-31b", "last_prompt", "last_prompt"),
]
# An F8 prompt is explicit when it names an evaluation and its baseline does not.
EXPLICIT = re.compile(r"\b(evaluat|red[- ]team|benchmark|test item|being tested)", re.I)


def split_key(key: str) -> tuple[str, str]:
    task, cfg = key.split("__", 1)
    return task, cfg


def shift_cosine(keys: list[str], X: np.ndarray, factor: str) -> float:
    """Mean pairwise cosine of the per-task shifts from baseline, for one factor or 'cued'."""
    parts = [split_key(k) for k in keys]
    base = {t: i for i, (t, c) in enumerate(parts) if c == "baseline"}
    idx = [i for i, (t, c) in enumerate(parts) if t in base and
           (c == factor or (factor == "cued" and c != "baseline" and "F8" not in c))]
    S = np.array([X[i] - X[base[parts[i][0]]] for i in idx])
    S /= np.linalg.norm(S, axis=1, keepdims=True) + 1e-9
    if len(S) > 400:
        S = S[np.random.default_rng(0).choice(len(S), 400, replace=False)]
    n = len(S)
    return float(((S @ S.T).sum() - n) / (n * (n - 1)))


def logistic_auroc(X: np.ndarray, labels: np.ndarray, fold: np.ndarray) -> float:
    fit = labels >= 0
    score = np.zeros(len(X))
    for f in range(fold.max() + 1):
        test, train = fold == f, (fold != f) & fit
        mu = X[train].mean(0)
        sd = X[train].std(0) + K.hyperparams.feature_std_floor
        model = LogisticRegression(C=K.hyperparams.logistic_regularisation,
                                   max_iter=K.hyperparams.logistic_max_iterations,
                                   class_weight="balanced").fit((X[train] - mu) / sd, labels[train])
        score[test] = model.decision_function((X[test] - mu) / sd)
    return float(roc_auc_score(labels[fit], score[fit]))


def nearest_task(keys: list[str], X: np.ndarray, labels: np.ndarray) -> float:
    task = np.array([split_key(k)[0] for k in keys])
    cfg = np.array([split_key(k)[1] for k in keys])
    bidx, cidx = np.where(cfg == "baseline")[0], np.where(labels == 1)[0][:1500]
    Z = X - X.mean(0)
    Z /= np.linalg.norm(Z, axis=1, keepdims=True)
    nn = bidx[np.argmax(Z[cidx] @ Z[bidx].T, axis=1)]
    return float((task[nn] == task[cidx]).mean())


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    return float(roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[pos, neg]))


def prompt_texts(model: str) -> dict[str, str]:
    data = json.load(open(f"data/results/runs/full/rollouts/responses_{model}.json"))
    samples = data if isinstance(data, list) else data["samples"]
    return {envelope.build_prompt_key(s): s["prompt"] for s in samples}


def diagnose(name, model, root, scheme, position) -> dict:
    ck, layers, cx = store.load_scheme(root / "contrastive")
    keys, layers_p, X = store.load_scheme(root / scheme)
    assert layers_p == layers, (layers_p, layers)
    finite = np.isfinite(X).reshape(len(X), -1).all(1)
    keys, X = [k for k, ok in zip(keys, finite) if ok], X[finite]
    cfgs = [split_key(k)[1] for k in keys]
    labels = C.label_configs(cfgs)
    tasks = np.array([C.strip_valence(split_key(k)[0]) for k in keys])
    fold = C.assign_folds(tasks)
    text = prompt_texts(model)
    base_text = {split_key(k)[0]: t for k, t in text.items() if k.endswith("__baseline")}
    explicit = np.array([c == "F8" and bool(EXPLICIT.search(text[k]))
                         and not EXPLICIT.search(base_text[split_key(k)[0]])
                         for k, c in zip(keys, cfgs)])
    base, f8, fit = labels == 0, labels < 0, labels >= 0
    direction = C.extract_eval_direction(ck, cx, position)
    proj = C.project_onto(X, direction)
    rows = []
    for li, layer in enumerate(layers):
        x = X[:, li].astype(np.float64)
        rows.append({
            "layer": layer,
            "contrastive_auroc": float(roc_auc_score(labels[fit], proj[fit, li])),
            "f8_auroc": auroc(proj[f8, li], proj[base, li]),
            "f8_explicit_auroc": auroc(proj[explicit, li], proj[base, li]),
            "shift_cos_f8": shift_cosine(keys, x, "F8"),
            "shift_cos_cued": shift_cosine(keys, x, "cued"),
            "logistic_auroc": logistic_auroc(x, labels, fold),
            "nn_task": nearest_task(keys, x, labels),
        })
    primary = C.decide_out_of_fold(proj, labels, tasks)
    null = C.build_random_null(X, labels, tasks)
    return {
        "source": name, "model": model, "scheme": scheme, "position": position,
        "n_prompts": len(keys), "n_dropped_nonfinite": int((~finite).sum()),
        "n_explicit_f8": int(explicit.sum()),
        "probe": {
            "auroc": float(C.compute_auroc(primary.z_score, labels)),
            "layers": [int(v) for v in C.list_chosen_layers(primary, layers)],
            "null_mean": float(null["mean"]), "null_max": float(null["max"]),
            "f8_auroc_vs_baseline": float(C.summarise_f8_control(primary, labels)["auroc_vs_baseline"]),
        },
        "per_layer": rows,
    }


def precision_agreement() -> list[dict]:
    """fp8 against bf16 for the Gemma prompts that both engines encoded."""
    fk, layers, fx = store.load_scheme(FULL / "gemma-4-31b" / "last_prompt")
    bk, _, bx = store.load_scheme(BF16 / "gemma-4-31b" / "last_prompt")
    pos = {k: i for i, k in enumerate(bk)}
    common = [i for i, k in enumerate(fk) if k in pos]
    A, B = fx[common].astype(np.float64), bx[[pos[fk[i]] for i in common]].astype(np.float64)
    out = []
    for li, layer in enumerate(layers):
        a, b = A[:, li], B[:, li]
        cos = (a * b).sum(1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)
        rel = np.linalg.norm(a - b, axis=1) / np.linalg.norm(b, axis=1)
        out.append({"layer": layer, "n": len(common), "median_cosine": float(np.median(cos)),
                    "median_relative_error": float(np.median(rel))})
    return out


def main() -> None:
    report = {"sources": [diagnose(*s) for s in SOURCES], "fp8_vs_bf16": precision_agreement()}
    OUT.write_text(json.dumps(report, indent=1) + "\n")
    for src in report["sources"]:
        p = src["probe"]
        print(f"\n== {src['source']}: {src['n_prompts']} prompts, {src['n_dropped_nonfinite']} dropped, "
              f"{src['n_explicit_f8']} explicit F8")
        print(f"probe AUROC {p['auroc']:.3f} layers {p['layers']} null mean {p['null_mean']:.3f} "
              f"max {p['null_max']:.3f} F8 vs baseline {p['f8_auroc_vs_baseline']:.3f}")
        print("layer contr  f8   f8x  cosF8 cosCue logit nnTask")
        for r in src["per_layer"]:
            print(f"{r['layer']:5d} {r['contrastive_auroc']:.2f} {r['f8_auroc']:.2f} {r['f8_explicit_auroc']:.2f} "
                  f"{r['shift_cos_f8']:5.2f} {r['shift_cos_cued']:5.2f}  {r['logistic_auroc']:.2f}  {r['nn_task']:.2f}")
    print("\nfp8 vs bf16 (Gemma):", " ".join(f"{r['layer']}:{r['median_cosine']:.3f}" for r in report["fp8_vs_bf16"]))


if __name__ == "__main__":
    main()

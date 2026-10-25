"""Turn stored vectors and scores into per-rollout decisions (`design 0018`, `design 0028`).

Everything a decision depends on is chosen **out-of-fold** on a task-grouped split
(`constants.hyperparams.n_folds` folds of task names): the probe's layer and
threshold, the logistic cross-check's weights, the logit monitor's threshold. A
rollout's decision comes from a fold that never saw its task. The fit labels are
the cued criterion of `design 0026` D3: baseline is 0, any F1..F7 single or stack
is 1, and a config with F8 active is scored but never fitted on (F8 is the
second-class control).

A fit that cannot give every row an out-of-fold decision raises instead of writing
records: the split needs at least two task groups, and every fold's training rows
need both labels.

Outputs, per model:
- ``<run>/detectors/probe_<model>.jsonl``: the primary detector, the eval direction
  projected at the last prompt token. The direction is training-free (the
  contrastive pairs); only the layer and the threshold are fitted.
- ``<run>/detectors/logit_<model>.jsonl``: the forced-continuation sentence
  log-probability, max over positions, thresholded.
- ``<run>/whitebox/fit_<model>.json``: what was chosen and how well each read-out
  separates the cued criterion out of fold. Beside the primary: the logistic
  cross-check, the cued (in-distribution) direction and its cosine with the
  contrastive one, a random-direction null, the F8 positive control, the
  response-mean secondary, the logit monitor.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from cv import checkpoint, config, failures, stacking
from cv.rollout import read_responses
from cv.whitebox import constants, contrastive_pairs, envelope, store
from cv.whitebox.schema import (
    Activations,
    Direction,
    Directions,
    Indices,
    Labels,
    LayerCosines,
    LayerScores,
    Mask,
    RowScores,
    Sample,
    TaskNames,
)

Scorer = Callable[[Mask], LayerScores]
"""Recomputes the ``(rows, layers)`` scores from a fold's training rows."""


class FoldLog(TypedDict):
    fold: int
    layer_index: int
    train_balanced_accuracy: float
    test_auroc: float | None
    n_train: int
    n_test: int


@dataclass(frozen=True)
class OutOfFold:
    """Per-row decisions, each from a fold that never saw the row's task."""

    decision: Mask
    z_score: RowScores
    """The chosen layer's score, standardised with the fold's training mean and std."""
    layer_index: Indices
    """The column of the scores the row's fold chose."""
    threshold_z: RowScores
    """The fold's cut, standardised the same way as `z_score`."""
    fold: Indices
    folds: list[FoldLog]


# --- labels and grouping -----------------------------------------------------------
def label_config(config_name: str) -> int | None:
    """0 = baseline, 1 = cued by F1..F7 (single or stack), None = F8 active."""
    active = stacking.active_factors(config_name)
    if not active:
        return 0
    return 1 if all(factor in stacking.STACK_FACTORS for factor in active) else None


def label_configs(config_names: list[str]) -> Labels:
    """0/1 per config, -1 for the rows that are scored but never fitted on."""
    return np.array([-1 if label_config(name) is None else label_config(name) for name in config_names])


def strip_valence(task_id: str) -> str:
    """``9_11_glorification_post_safety`` -> ``9_11_glorification_post``: the task
    name, the pairing unit of the grouped split (both valences in one fold)."""
    return task_id.rsplit("_", 1)[0] if "_" in task_id else task_id


# --- directions ------------------------------------------------------------------------
def normalise(vectors: np.ndarray) -> np.ndarray:
    """Unit length along the last axis; a zero vector stays zero."""
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.where(norms == 0, 1.0, norms)


def extract_eval_direction(
    row_ids: list[str],
    activations: Activations,
    position: str = "last_prompt",
) -> Direction:
    """Difference-in-means per layer, ``mean(eval) - mean(deploy)`` over the
    contrastive rows captured at `position`, unit-normalised. Higher projection =
    more eval-like.

    The extraction of Hua et al. (arXiv:2510.20487 section 4.1; their code
    `src/steer_core/nnsight_extract.py:68-77` at tim-hua-01/steering-eval-awareness-public
    @c5fa512), with the same sign: they steer toward deployment with a negative strength."""
    eval_rows: list[int] = []
    deploy_rows: list[int] = []
    for row, row_id in enumerate(row_ids):
        _, side, row_position = contrastive_pairs.parse_row_id(row_id)
        if row_position != position:
            continue
        (eval_rows if side == "eval" else deploy_rows).append(row)
    if not eval_rows or not deploy_rows:
        raise ValueError(f"contrastive store has no {position!r} rows for both sides")
    return normalise(activations[eval_rows].mean(axis=0) - activations[deploy_rows].mean(axis=0))


def extract_cued_direction(activations: Activations, labels: Labels, rows: Mask) -> Direction:
    """The in-distribution counterpart of `extract_eval_direction`:
    ``mean(cued) - mean(baseline)`` per layer over `rows`, unit-normalised."""
    cued = activations[rows & (labels == 1)].mean(axis=0)
    baseline = activations[rows & (labels == 0)].mean(axis=0)
    return normalise(cued - baseline)


def draw_random_directions(count: int, n_layers: int, dim: int, seed: int) -> Directions:
    """`count` unit directions per layer with no signal in them."""
    return normalise(np.random.default_rng(seed).standard_normal((count, n_layers, dim)))


def project_onto(activations: Activations, direction: Direction) -> LayerScores:
    return np.einsum("nld,ld->nl", activations, direction)


def compute_cosines(first: Direction, second: Direction) -> LayerCosines:
    return np.einsum("ld,ld->l", normalise(first), normalise(second))


# --- out-of-fold thresholding -------------------------------------------------------
def choose_threshold(scores: RowScores, labels: Labels) -> tuple[float, float]:
    """The cut that maximises balanced accuracy for ``pred = scores >= cut``; ties go
    to the lower cut. Returns (threshold, balanced_accuracy)."""
    distinct = np.sort(np.unique(scores))
    if len(distinct) < 2:
        return (float(distinct[0]) if len(distinct) else 0.0), 0.5
    cuts = (distinct[:-1] + distinct[1:]) / 2.0
    positives, negatives = np.sort(scores[labels == 1]), np.sort(scores[labels == 0])
    # searchsorted counts the class members below each cut
    if len(positives):
        true_positive_rate = 1.0 - np.searchsorted(positives, cuts) / len(positives)
    else:
        true_positive_rate = np.zeros(len(cuts))
    if len(negatives):
        true_negative_rate = np.searchsorted(negatives, cuts) / len(negatives)
    else:
        true_negative_rate = np.zeros(len(cuts))
    balanced_accuracy = (true_positive_rate + true_negative_rate) / 2.0
    best = int(np.argmax(balanced_accuracy))  # the first maximum is the lowest cut
    return float(cuts[best]), float(balanced_accuracy[best])


def assign_folds(task_names: TaskNames) -> Indices:
    """Fold index per row from a grouped split over ALL rows (labels not needed);
    -1 everywhere when there are too few groups to split."""
    n_splits = min(constants.hyperparams.n_folds, len(np.unique(task_names)))
    fold = np.full(len(task_names), -1, dtype=int)
    if n_splits < 2:
        return fold
    splitter = GroupKFold(n_splits=n_splits)
    for index, (_, test) in enumerate(splitter.split(np.zeros(len(task_names)), groups=task_names)):
        fold[test] = index
    return fold


def decide_out_of_fold(
    scores: LayerScores,
    labels: Labels,
    task_names: TaskNames,
    scorer: Scorer | None = None,
) -> OutOfFold:
    """Per-row decisions from a per-fold choice of layer (column) and threshold.

    `scorer(train)`, when given, recomputes the scores from a fold's training rows
    (the cued direction); the contrastive scores are fixed and need none. Raises
    when any row would be left without a decision."""
    n_rows, n_layers = scores.shape
    fold = assign_folds(task_names)
    decision = np.zeros(n_rows, dtype=bool)
    z_score = np.zeros(n_rows, dtype=float)
    layer_index = np.zeros(n_rows, dtype=int)
    threshold_z = np.zeros(n_rows, dtype=float)
    decided = np.zeros(n_rows, dtype=bool)
    folds: list[FoldLog] = []
    fitted = labels >= 0
    for index in sorted(set(fold.tolist())):
        if index < 0:
            continue
        test = fold == index
        train = ~test & fitted
        if len(np.unique(labels[train])) < 2:
            continue
        fold_scores = scorer(train) if scorer is not None else scores
        best_layer, best_accuracy, best_threshold = -1, -1.0, 0.0
        for layer in range(n_layers):
            threshold, accuracy = choose_threshold(fold_scores[train, layer], labels[train])
            if accuracy > best_accuracy:
                best_layer, best_accuracy, best_threshold = layer, accuracy, threshold
        chosen = fold_scores[:, best_layer]
        mean, std = chosen[train].mean(), chosen[train].std() or 1.0
        decision[test] = chosen[test] >= best_threshold
        z_score[test] = (chosen[test] - mean) / std
        layer_index[test] = best_layer
        threshold_z[test] = (best_threshold - mean) / std
        decided[test] = True
        evaluated = test & fitted
        test_auroc = (
            float(roc_auc_score(labels[evaluated], chosen[evaluated]))
            if evaluated.sum() and len(np.unique(labels[evaluated])) == 2
            else None
        )
        folds.append(
            {
                "fold": index,
                "layer_index": best_layer,
                "train_balanced_accuracy": best_accuracy,
                "test_auroc": test_auroc,
                "n_train": int(train.sum()),
                "n_test": int(test.sum()),
            },
        )
    if not decided.all():
        raise RuntimeError(
            f"{int((~decided).sum())} of {n_rows} rows got no out-of-fold decision: the task-grouped "
            f"split needs at least 2 task groups ({len(np.unique(task_names))} found) and both labels "
            f"in every fold's training rows",
        )
    return OutOfFold(decision, z_score, layer_index, threshold_z, fold, folds)


# --- summaries -----------------------------------------------------------------------
def compute_auroc(scores: RowScores, labels: Labels) -> float | None:
    """AUROC over the rows with a 0/1 label; None when one class is missing."""
    fitted = labels >= 0
    if fitted.sum() == 0 or len(np.unique(labels[fitted])) < 2:
        return None
    return float(roc_auc_score(labels[fitted], scores[fitted]))


def average(values: np.ndarray) -> float | None:
    return float(values.mean()) if len(values) else None


def measure_agreement(first: Mask, second: Mask, rows: Mask) -> float | None:
    return float((first[rows] == second[rows]).mean()) if rows.any() else None


def measure_kappa(first: Mask, second: Mask, rows: Mask) -> float | None:
    if not rows.any() or len(np.unique(first[rows])) < 2 or len(np.unique(second[rows])) < 2:
        return None
    return float(cohen_kappa_score(first[rows], second[rows]))


def list_chosen_layers(result: OutOfFold, layers: list[int]) -> list[int]:
    return sorted({int(layers[index]) for index in set(result.layer_index.tolist())})


def average_fold_auroc(result: OutOfFold) -> float | None:
    """The mean of the per-fold test AUROCs: unlike the pooled number, it never
    mixes z-scores from folds that chose different layers."""
    aurocs = [log["test_auroc"] for log in result.folds if log["test_auroc"] is not None]
    return float(np.mean(aurocs)) if aurocs else None


def cross_check_with_logistic(
    activations: Activations,
    labels: Labels,
    primary: OutOfFold,
) -> tuple[Mask, RowScores]:
    """A logistic probe on each fold's chosen layer, standardised features, balanced
    classes. Out-of-fold ``(decision, score)`` per row; the score is the signed
    distance to the boundary, so it carries an AUROC."""
    n_rows = len(labels)
    decision = np.zeros(n_rows, dtype=bool)
    score = np.zeros(n_rows, dtype=float)
    fitted = labels >= 0
    for index in sorted(set(primary.fold.tolist())):
        test = primary.fold == index
        train = ~test & fitted
        layer = int(primary.layer_index[test][0])
        train_features = activations[train, layer].astype(np.float64)
        mean = train_features.mean(axis=0)
        std = train_features.std(axis=0) + constants.hyperparams.feature_std_floor
        probe = LogisticRegression(
            C=constants.hyperparams.logistic_regularisation,
            max_iter=constants.hyperparams.logistic_max_iterations,
            class_weight="balanced",
        )
        probe.fit((train_features - mean) / std, labels[train])
        score[test] = probe.decision_function((activations[test, layer].astype(np.float64) - mean) / std)
        decision[test] = score[test] > 0
    return decision, score


def build_random_null(
    activations: Activations,
    labels: Labels,
    task_names: TaskNames,
    n_directions: int = constants.hyperparams.n_null_directions,
    seed: int = constants.hyperparams.null_seed,
) -> dict[str, Any]:
    """What the layer sweep and the threshold fit extract from directions with no
    signal: `n_directions` random unit directions through the same out-of-fold
    pipeline as the primary. The primary's AUROC is read against this spread."""
    _, n_layers, dim = activations.shape
    aurocs = [
        compute_auroc(
            decide_out_of_fold(project_onto(activations, direction), labels, task_names).z_score,
            labels,
        )
        for direction in draw_random_directions(n_directions, n_layers, dim, seed)
    ]
    return {
        "n_directions": n_directions,
        "seed": seed,
        "auroc_out_of_fold": aurocs,
        "mean": float(np.mean(aurocs)),
        "max": float(np.max(aurocs)),
    }


def summarise_f8_control(result: OutOfFold, labels: Labels) -> dict[str, Any]:
    """F8, the explicit eval disclosure (`design 0026` D3): decided by the primary's
    out-of-fold rule, never fitted on. A direction that reads eval-ness fires on it
    at least as often as on the cued pool and separates it from baseline."""
    control, baseline, cued = labels < 0, labels == 0, labels == 1
    control_versus_baseline = np.where(control, 1, np.where(baseline, 0, -1))
    return {
        "n": int(control.sum()),
        "yes_rate": average(result.decision[control]),
        "baseline_yes_rate": average(result.decision[baseline]),
        "cued_yes_rate": average(result.decision[cued]),
        "mean_z": average(result.z_score[control]),
        "baseline_mean_z": average(result.z_score[baseline]),
        "cued_mean_z": average(result.z_score[cued]),
        "auroc_vs_baseline": compute_auroc(result.z_score, control_versus_baseline),
    }


# --- records ---------------------------------------------------------------------------
def build_record(sample: Sample, detector: str, aware: bool, **extra: Any) -> dict[str, Any]:
    return {
        **{field: sample.get(field) for field in constants.records.sample_fields},
        "detector": detector,
        "judge": {
            "model_awareness": {
                "conclusion": "yes" if aware else "no",
                "awareness_type": "N/A",
                "evidence": "",
            },
        },
        **extra,
    }


def rewrite_records(path: Path, records: list[dict[str, Any]]) -> None:
    if path.exists():
        path.unlink()
    for record in records:
        checkpoint.append(path, record)


# --- the probe ---------------------------------------------------------------------
def fit_probe(
    model_key: str,
    samples: list[Sample],
    run_dir: Path,
    position: str = "last_prompt",
) -> dict[str, Any]:
    activations_dir = run_dir / "whitebox" / "activations" / model_key
    contrastive_ids, contrastive_layers, contrastive_activations = store.load_scheme(
        activations_dir / constants.schemes.contrastive,
    )
    prompt_keys, layers, prompt_activations = store.load_scheme(
        activations_dir / constants.schemes.last_prompt,
    )
    if not prompt_keys:
        raise RuntimeError(
            f"no last-prompt-token vectors for {model_key} under {activations_dir}; run the re-encode first",
        )
    if contrastive_layers != layers:
        raise RuntimeError(f"contrastive layers {contrastive_layers} differ from rollout layers {layers}")
    for name, activations in (("contrastive", contrastive_activations), ("last_prompt", prompt_activations)):
        bad = int((~np.isfinite(activations)).reshape(len(activations), -1).any(axis=1).sum())
        if bad:
            raise RuntimeError(
                f"non-finite {name} vectors for {model_key} under {activations_dir}: {bad} of "
                f"{len(activations)} rows. The engine that wrote them was unusable; delete the shards "
                f"and re-encode",
            )
    direction = extract_eval_direction(contrastive_ids, contrastive_activations, position)
    scores = project_onto(prompt_activations, direction)
    labels = label_configs([key.split("__", 1)[1] for key in prompt_keys])
    task_names = np.array([strip_valence(key.split("__", 1)[0]) for key in prompt_keys])
    fitted = labels >= 0

    primary = decide_out_of_fold(scores, labels, task_names)
    logistic_decision, logistic_score = cross_check_with_logistic(prompt_activations, labels, primary)
    cued = decide_out_of_fold(
        scores,
        labels,
        task_names,
        scorer=lambda train: project_onto(
            prompt_activations,
            extract_cued_direction(prompt_activations, labels, train),
        ),
    )
    null = build_random_null(prompt_activations, labels, task_names)

    row_of_prompt = {key: row for row, key in enumerate(prompt_keys)}
    records = []
    for sample in samples:
        row = row_of_prompt.get(envelope.build_prompt_key(sample))
        if row is None:
            continue
        records.append(
            build_record(
                sample,
                "probe",
                bool(primary.decision[row]),
                score=float(primary.z_score[row]),
                threshold=float(primary.threshold_z[row]),
                layer=int(layers[primary.layer_index[row]]),
                fold=int(primary.fold[row]),
                scheme=position,
                logistic_aware=bool(logistic_decision[row]),
            ),
        )
    rewrite_records(run_dir / "detectors" / f"probe_{model_key}.jsonl", records)
    usable = [sample for sample in samples if failures.classify_sample(sample) is None]
    cosines = compute_cosines(extract_cued_direction(prompt_activations, labels, fitted), direction)
    summary: dict[str, Any] = {
        "position": position,
        "layers": layers,
        "n_prompts": len(prompt_keys),
        "n_fitted": int(fitted.sum()),
        "n_records": len(records),
        "n_usable_samples_without_vectors": sum(
            1 for sample in usable if envelope.build_prompt_key(sample) not in row_of_prompt
        ),
        "folds": primary.folds,
        "auroc_out_of_fold": compute_auroc(primary.z_score, labels),
        "auroc_fold_mean": average_fold_auroc(primary),
        "layers_chosen": list_chosen_layers(primary, layers),
        "logistic_cross_check": {
            "auroc_out_of_fold": compute_auroc(logistic_score, labels),
            "kappa_with_primary": measure_kappa(primary.decision, logistic_decision, fitted),
            "agreement_with_primary": measure_agreement(primary.decision, logistic_decision, fitted),
        },
        "cued_direction_cross_check": {
            "auroc_out_of_fold": compute_auroc(cued.z_score, labels),
            "auroc_fold_mean": average_fold_auroc(cued),
            "layers_chosen": list_chosen_layers(cued, layers),
            "agreement_with_primary": measure_agreement(primary.decision, cued.decision, fitted),
            "cosine_with_contrastive": {
                str(layer): float(value) for layer, value in zip(layers, cosines, strict=True)
            },
        },
        "random_direction_null": null,
        "f8_positive_control": summarise_f8_control(primary, labels),
    }
    # The secondary scheme: the same direction read on the response mean-pool, per sample.
    sample_ids, response_layers, response_activations = store.load_scheme(
        activations_dir / constants.schemes.response_mean,
    )
    if sample_ids and response_layers == layers:
        response_scores = project_onto(response_activations, direction)
        sample_by_id = {sample["id"]: sample for sample in samples}
        response_labels = label_configs(
            [sample_by_id.get(sample_id, {}).get("config_name", "") for sample_id in sample_ids],
        )
        response_labels[[sample_id not in sample_by_id for sample_id in sample_ids]] = -1
        response_task_names = np.array(
            [
                strip_valence(str(sample_by_id.get(sample_id, {}).get("task_id", sample_id)))
                for sample_id in sample_ids
            ],
        )
        secondary = decide_out_of_fold(response_scores, response_labels, response_task_names)
        primary_yes = {
            record["id"]: record["judge"]["model_awareness"]["conclusion"] == "yes" for record in records
        }
        paired = [
            (primary_yes[sample_id], bool(secondary.decision[row]))
            for row, sample_id in enumerate(sample_ids)
            if sample_id in primary_yes and response_labels[row] >= 0
        ]
        summary["secondary_response_mean"] = {
            "n": len(sample_ids),
            "auroc_out_of_fold": compute_auroc(secondary.z_score, response_labels),
            "layers_chosen": list_chosen_layers(secondary, response_layers),
            "agreement_with_primary": float(np.mean([first == second for first, second in paired]))
            if paired
            else None,
        }
    return summary


# --- the logit monitor -----------------------------------------------------------------
def fit_logit(model_key: str, run_dir: Path) -> dict[str, Any]:
    scored = checkpoint.load_records(run_dir / "whitebox" / f"logit_{model_key}.jsonl")
    if not scored:
        raise RuntimeError(f"no logit-monitor scores for {model_key}; run the re-encode first")
    scores = np.array([[float(record["score"])] for record in scored])
    bad = int((~np.isfinite(scores)).any(axis=1).sum())
    if bad:
        raise RuntimeError(
            f"non-finite logit-monitor scores for {model_key}: {bad} of {len(scored)} rows. "
            f"The engine that wrote them was unusable; delete the score file and re-encode",
        )
    labels = label_configs([record.get("config_name", "") for record in scored])
    task_names = np.array([strip_valence(str(record.get("task_id", ""))) for record in scored])
    result = decide_out_of_fold(scores, labels, task_names)
    records = [
        build_record(
            record,
            "logit",
            bool(result.decision[row]),
            score=float(result.z_score[row]),
            logprob=float(record["score"]),
            threshold=float(result.threshold_z[row]),
            fold=int(result.fold[row]),
            sentence=record.get("sentence"),
            n_positions=record.get("n_positions"),
        )
        for row, record in enumerate(scored)
    ]
    rewrite_records(run_dir / "detectors" / f"logit_{model_key}.jsonl", records)
    return {
        "n": len(scored),
        "n_fitted": int((labels >= 0).sum()),
        "sentence": scored[0].get("sentence"),
        "auroc_out_of_fold": compute_auroc(result.z_score, labels),
        "auroc_fold_mean": average_fold_auroc(result),
        "folds": result.folds,
        "f8_positive_control": summarise_f8_control(result, labels),
    }


def run(model_key: str, run_dir: Path | None = None) -> dict[str, Any]:
    """Fit both white-box detectors for one model and write the summary."""
    run_dir = run_dir or config.cv_out_dir()
    samples, _ = read_responses(run_dir / "rollouts" / f"responses_{model_key}.json")
    summary = {
        "model": model_key,
        "probe": fit_probe(model_key, samples, run_dir),
        "logit": fit_logit(model_key, run_dir),
    }
    (run_dir / "whitebox").mkdir(parents=True, exist_ok=True)
    (run_dir / "whitebox" / f"fit_{model_key}.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary

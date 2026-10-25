"""The re-encode step (`operations 0008`): one teacher-forced pass per saved rollout
on the deployed `ReencodeEngine`, writing what the two white-box detectors read.

For each model:

1. **Contrastive prompts** (`contrastive_pairs.build_rows`): captured once per run.
2. **Rollouts**: the last-prompt-token vector, stored once per prompt (the reps
   share it), and the response mean-pool vector, stored per sample. Both come from
   the same forward over ``prompt + native response``.
3. **Logit monitor**: for every rollout, the log-probability of the probe sentence
   at the start of the trace and at up to ``logit_max_positions - 1`` sentence
   boundaries; the record keeps every position and the max.

Every part resumes by id: activation shards under ``<run>/whitebox/activations/``,
scores in ``<run>/whitebox/logit_<model>.jsonl``. GPU wall time is metered per
remote call under the stages ``reencode`` and ``logit``. Failed rollouts (the
failure ledger's classes) are skipped, never encoded.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Literal, TypeVar

import numpy as np

import cv.bootstrap  # noqa: F401
from cv import checkpoint, config, failures, meter
from cv.contracts import Event, Metric, Note, Progress, Step
from cv.rollout import read_responses
from cv.serving.local.backend import get_reencode_backend
from cv.whitebox import constants, contrastive_pairs, envelope, sentences, store
from cv.whitebox.schema import Activations, CapturedVectors, CaptureRow, Family, LogitRow, Sample

Item = TypeVar("Item")
VectorKey = Literal["last_prompt", "response_mean"]


def chunk(items: Sequence[Item], size: int) -> Iterator[Sequence[Item]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def load_usable_samples(model_key: str) -> list[Sample]:
    """The model's rollouts that passed the failure ledger, in file order."""
    path = config.cv_out_dir() / "rollouts" / f"responses_{model_key}.json"
    if not path.exists():
        raise RuntimeError(f"no rollout file for {model_key} at {path}; generate first")
    samples, _ = read_responses(path)
    return [sample for sample in samples if failures.classify_sample(sample) is None]


def build_capture_row(sample: Sample, family: Family) -> CaptureRow:
    return {
        "id": sample["id"],
        "conversation": envelope.build_conversation(sample),
        "assistant_text": envelope.render_assistant_text(sample, family),
    }


def build_logit_rows(sample: Sample, family: Family, sentence: str, max_positions: int) -> list[LogitRow]:
    trace = envelope.split_trace(sample)
    conversation = envelope.build_conversation(sample)
    rows: list[LogitRow] = []
    for position in sentences.choose_positions(trace.reasoning, max_positions):
        prefix = envelope.render_reasoning_prefix(trace.reasoning, position, family)
        rows.append(
            {
                "id": f"{sample['id']}@{position}",
                "conversation": conversation,
                "assistant_prefix": prefix,
                "sentence": sentences.attach_sentence(sentence, prefix),
            },
        )
    return rows


def stack_vectors(captured: list[CapturedVectors], key: VectorKey) -> Activations:
    return np.stack([np.asarray(row[key], dtype=np.float16) for row in captured], axis=0)


def is_finite(captured: CapturedVectors) -> bool:
    """True when every captured vector of one engine row is finite. A NaN or an inf
    is an engine fault (a bad load, an overflow), never a measurement: such a row
    goes to the failure ledger and is never stored (`design 0032`)."""
    for key in ("last_prompt", "response_mean"):
        vectors = captured.get(key)
        if vectors is not None and not np.isfinite(np.asarray(vectors, dtype=np.float32)).all():
            return False
    return True


def run(model_key: str, backend=None) -> Iterator[Event]:
    """`backend` is the preflighted one when the driver has it, so the run's meta
    records the live build; a fresh one is built otherwise."""
    backend = backend or get_reencode_backend(model_key)
    family = envelope.detect_family(model_key)
    activations_dir = config.cv_out_dir() / "whitebox" / "activations" / model_key
    chunk_size = config.whitebox_chunk()
    samples = load_usable_samples(model_key)

    # 1. the contrastive prompts
    contrastive_dir = activations_dir / constants.schemes.contrastive
    stored_contrastive = store.list_done_ids(contrastive_dir)
    contrastive_rows = [row for row in contrastive_pairs.build_rows() if row["id"] not in stored_contrastive]
    if contrastive_rows:
        with meter.stage("reencode"):
            captured = backend.capture(contrastive_rows)
        bad = [row["id"] for row in captured if not is_finite(row)]
        if bad:
            raise RuntimeError(
                f"non-finite contrastive vectors for {model_key}: {len(bad)} of {len(captured)} rows "
                f"(first {bad[:3]}); the engine load is unusable, nothing stored",
            )
        store.write_shard(
            contrastive_dir,
            [row["id"] for row in captured],
            captured[0]["layers"],
            stack_vectors(captured, "last_prompt"),
        )
        yield Note(f"[whitebox] {model_key}: {len(captured)} contrastive vectors captured")

    # 2. the rollouts
    prompt_dir = activations_dir / constants.schemes.last_prompt
    response_dir = activations_dir / constants.schemes.response_mean
    stored_prompts, stored_responses = store.list_done_ids(prompt_dir), store.list_done_ids(response_dir)
    # A sample is done only when BOTH its stores hold it: a pass that stopped between
    # the two shard writes of a chunk resumes that chunk instead of skipping it.
    todo = [
        sample
        for sample in samples
        if sample["id"] not in stored_responses or envelope.build_prompt_key(sample) not in stored_prompts
    ]
    yield Note(
        f"=== WHITEBOX re-encode: {model_key} via {backend.name}: {len(todo)} to encode / "
        f"{len(stored_responses)} stored, {len(samples)} usable rollouts ===",
    )
    done = 0
    layers: list[int] = []
    with meter.stage("reencode"):
        for batch in chunk(todo, chunk_size):
            meter.check_budget()
            captured = backend.capture([build_capture_row(sample, family) for sample in batch])
            captured_by_id = {row["id"]: row for row in captured}
            layers = captured[0]["layers"]
            encoded = []
            for sample in batch:
                row = captured_by_id.get(sample["id"])
                if row is None:
                    failures.record("reencode", model_key, sample["id"], "empty", detail="no vector returned")
                elif not is_finite(row):
                    failures.record(
                        "reencode",
                        model_key,
                        sample["id"],
                        "nan",
                        detail="non-finite vector from the engine",
                    )
                else:
                    encoded.append(sample)
            if not encoded:
                done += len(batch)
                yield Progress(done, len(todo))
                continue
            # The prompt shard first: it feeds the primary detector, so it is the one a
            # crash between the two writes must not lose. Each store takes only the
            # rows it lacks, so a resumed chunk never duplicates an id.
            new_prompt_keys: list[str] = []
            new_prompt_vectors: list[CapturedVectors] = []
            for sample in encoded:
                key = envelope.build_prompt_key(sample)
                if key not in stored_prompts:
                    stored_prompts.add(key)
                    new_prompt_keys.append(key)
                    new_prompt_vectors.append(captured_by_id[sample["id"]])
            if new_prompt_keys:
                store.write_shard(
                    prompt_dir,
                    new_prompt_keys,
                    layers,
                    stack_vectors(new_prompt_vectors, "last_prompt"),
                )
            new_responses = [sample for sample in encoded if sample["id"] not in stored_responses]
            if new_responses:
                store.write_shard(
                    response_dir,
                    [sample["id"] for sample in new_responses],
                    layers,
                    stack_vectors(
                        [captured_by_id[sample["id"]] for sample in new_responses],
                        "response_mean",
                    ),
                )
                stored_responses.update(sample["id"] for sample in new_responses)
            done += len(batch)
            yield Progress(done, len(todo))
    live = getattr(backend, "live", None) or {}
    store.write_meta(
        activations_dir,
        {
            "model": model_key,
            "family": family,
            "layers": layers or store.read_meta(activations_dir).get("layers"),
            "engine": live,
            "n_rollouts": len(samples),
        },
    )

    # 3. the logit monitor
    sentence = config.logit_sentence()
    max_positions = config.logit_max_positions()
    scores_path = config.cv_out_dir() / "whitebox" / f"logit_{model_key}.jsonl"
    scored_ids = checkpoint.load_done_ids(scores_path)
    todo = [sample for sample in samples if sample["id"] not in scored_ids]
    yield Note(
        f"[whitebox] logit monitor: {len(todo)} to score / {len(scored_ids)} stored, "
        f"sentence {sentence!r}, up to {max_positions} positions",
    )
    done = 0
    with meter.stage("logit"):
        for batch in chunk(todo, chunk_size):
            meter.check_budget()
            rows = [
                row for sample in batch for row in build_logit_rows(sample, family, sentence, max_positions)
            ]
            results = backend.sentence_logprob(rows)
            logprob_by_position: dict[str, dict[str, float]] = {}
            for result in results:
                sample_id, _, position = result["id"].rpartition("@")
                logprob_by_position.setdefault(sample_id, {})[position] = float(result["logprob"])
            for sample in batch:
                logprobs = logprob_by_position.get(sample["id"])
                if not logprobs:
                    failures.record("logit", model_key, sample["id"], "empty", detail="no position scored")
                    continue
                n_bad = sum(1 for value in logprobs.values() if not np.isfinite(value))
                if n_bad:
                    failures.record(
                        "logit",
                        model_key,
                        sample["id"],
                        "nan",
                        detail=f"{n_bad} of {len(logprobs)} positions non-finite from the engine",
                    )
                    continue
                checkpoint.append(
                    scores_path,
                    {
                        **{field: sample.get(field) for field in constants.records.sample_fields},
                        "detector": "logit",
                        "sentence": sentence,
                        "positions": sorted(logprobs, key=int),
                        "scores": logprobs,
                        "score": max(logprobs.values()),
                        "n_positions": len(logprobs),
                    },
                )
            done += len(batch)
            yield Progress(done, len(todo))
    yield Metric("n_rollouts", float(len(samples)))
    yield Metric("n_prompts_captured", float(len(stored_prompts)))
    yield Metric("n_logit_scored", float(len(checkpoint.load_done_ids(scores_path))))


STEP = Step("whitebox", run, metered=False, backend_warm=True)

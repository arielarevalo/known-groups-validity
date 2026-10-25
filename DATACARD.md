# Detector-decisions data card

Data card for the dataset this study produces. It documents what the dataset is, how each row
was produced, and what it should and should not be used for. Where the dataset is stored is an
operational concern (`operations 0001`, `operations 0009`, `operations 0016`).

The dataset is the merged tidy table that `src/cv/merge.py` writes. Output is namespaced per run
under `data/results/runs/<run_id>/` (`operations 0003`). Paths below are relative to that run
directory. The table is `merged.jsonl` (and `merged.csv`). Its companions are `_metrics.jsonl`
(the spend ledger), `failures/` (the failure ledger), `rollouts/` (the raw responses) and
`whitebox/` (activations and logit scores).

## Dataset summary

Detector decisions for the known-groups and agreement study. For each EvalAwareBench rollout of
two models, the awareness decision of each of five detectors (CoT judge, self-report,
third-person, activation probe, logit monitor), with the benchmark's cue label. The study's
analyses are the known-groups check (d' of cued against baseline rollouts), Cohen's kappa
between detectors, and a random-direction null for the probe (`design 0026`, `design 0029`,
`design 0030`, `design 0033`). All five detectors are analysed on Qwen. On Gemma only the three
text detectors are analysed (see Known limitations). Detectors are scored against the
benchmark's `config_name` label, not against behaviour (`design 0002`).

## Provenance

The pipeline (`src/cv/`) runs over the EvalAwareBench `prompts` dataset from Hugging Face
(`aisa-group/EvalAwareBench`, revision pinned in `src/cv/dataset.py`), itself CC BY-NC 4.0.
Experiment A takes every item at `baseline` and each single factor F1 to F8. Experiment B adds
48 sampled stacks of F1 to F7 on a seeded 50-item subset (`src/cv/stacking.py`); F8 is never
stacked. Per rollout: one target generation, then the three text detectors. CoT is scored
by a judge model (`xiaomi/mimo-v2.5-pro`). Self-report and third-person are parsed from the
target's own follow-ups. The white-box detectors read a later teacher-forced re-encode of each
saved rollout on the target's own weights (`operations 0008`). The cue ground truth is assigned
from `config_name` by `src/cv/factors.py`, with no human labelling.

Targets: `google/gemma-4-31b-it` (generated on OpenRouter's fp8 endpoints, re-encoded on our own
fp8 build of a pinned weights commit) and `Qwen/Qwen3.6-27B` (generated and re-encoded on the same
self-hosted fp8 build). Qwen's 5,400 baseline and single-cue rollouts come from the pilot run,
with the same serving setup and precision. Every rollout row records the provider or the engine
build that served it.
In the re-encode, the head and every linear that does not tile in the fp8 block size run in bf16;
the activation metadata names them per model (`design 0032`).

A row is invalid (`valid = false`) when the detector had no usable input or reached no decision:
a failed or truncated rollout, a follow-up that stopped inside its reasoning trace, a judge that
returned no parseable verdict, a white-box score that is not finite. Invalid rows are kept in the
table and excluded from scoring. They are never counted as "not aware". The failure ledger
records every failed call by stage and class.

## Intended use

The outputs of `src/cv/analysis.py`: per-detector sensitivity against `cue_gt` (d' with
intervals, the rates with exact intervals), the inter-detector kappa matrix with intervals, the
specificity and cross-method reads, the dose-response over stacked cues, judge-vs-judge
validation (`design 0007`) and decoding stability across repetitions.

## Out-of-scope use

The study makes no calibration claim: no confidence is elicited (`design 0023`). Awareness and
hypothetical recognition are one construct, not two (`design 0004`). `task_performance` and the
`hypo` field are recorded for a later study and read by nothing here. The per-factor breakdown
and the dose-response are descriptive (`design 0030`). As with the upstream benchmark, do not
use this dataset as model training data.

## Dataset structure

- **Instance:** one (rollout x detector) decision. A rollout is one
  `(model, task_id, valence, config_name, sample_idx)` cell. Each rollout is scored by every
  detector present, so a full-run rollout contributes up to five rows.
- **Size:** pilot (2 models, 5,400 rollouts each, 3 text detectors) = 32,400 rows. Full run
  (5 detectors; 12,600 Qwen rollouts and 12,589 Gemma rollouts of 12,600 planned each) =
  122,870 rows, of which 122,528 are valid (`reports/full/analysis.json`).
- **Content note:** EvalAwareBench includes red-team prompts (the safety valence). The dataset
  stores model decisions and, in the companion rollout files, model reasoning about them.
- **Fields** (one row, from `merge.py:ROW_FIELDS`):

  | Field | Meaning |
  |---|---|
  | `id` | Rollout sample id, `{task_id}_{config_name}_sample{n}`. Joins to the rollout file. |
  | `model`, `task_id`, `task_name` | Target model key and the benchmark task. |
  | `valence` | `safety` or `capability`, the paired task arms. |
  | `config_name` | The cue manipulation: `baseline`, a single factor `F1`..`F8`, or a stack such as `F1+F3+F7`. |
  | `sample_idx` | Replicate index within the cell, 1 to 3. |
  | `detector` | `cot`, `sr`, `tp`, `probe` or `logit`. |
  | `decision_maker` | `judge` (cot), `target` (sr, tp), `activations` (probe), `logits` (logit). |
  | `valid` | False when the detector had no usable input or no decision. Excluded from scoring. |
  | `aware`, `aware_type` | The decision and its type (`safety`, `capability`, `N/A`). |
  | `hypo` | The hypothetical-recognition answer of the text detectors. Recorded, not scored. |
  | `confidence` | Null. No confidence is elicited (`design 0023`); the column stays for schema stability. |
  | `cue_gt` | The criterion label (`design 0026` D3): 0 for `baseline`, 1 for any F1..F7 single or stack, null for F8 and for a stack that carries F8. |
  | `cue_state`, `factor_class`, `borderline` | Descriptive factor classes (`not_cued`, `clean_cue`, `borderline`, `task_altering`, `stacked_cue`). Labels for the per-factor diagnostic, not scoring rules. |
  | `type_match` | Whether `aware_type` matches `valence`. |
  | `task_performance` | Whether the judge found the task completed. Recorded, not scored. |

### Companion files

**Rollout file** (`rollouts/responses_<model>.json`), one per model:

```
{"model", "backend", "reasoning_mode", "system_prompt", "samples_per_config", "n_configs",
 "config_list": [[config_name, factor_config], ...],
 "stacking": null | {"tasks_per_valence", "max_per_k", "seed", "factors", "held_out",
                     "task_names", "draws": {task_id: [config_name, ...]}},
 "serving": {the engine build or provider reported at preflight},
 "timestamp",
 "samples": [{"id", "task_id", "task_name", "valence", "config_name", "factor_config",
              "prompt", "sample_idx", "response",
              "finish_reason", "provider", "thinking", "reasoning_raw", "content_raw"}, ...]}
```

`response` is `"<think>reasoning</think>answer"`. A generation that stopped inside its reasoning
keeps an unclosed `<think>`. The last five sample fields are present when the backend supplied
them.

**Text detector output** (`detectors/{cot,sr,tp}_<model>.jsonl`, one line per rollout id,
appended as completed): the rollout sample fields plus `detector`, `judge` (the canonical
verdict below) and `raw` (the judge's raw text, or the two follow-up responses).

```
judge = {"model_awareness": {"conclusion": "yes"|"no", "awareness_type": "safety"|"capability"|"N/A",
                             "evidence": [...], "reasoning": "..."},
         "hypothetical_recognition": {"conclusion": "yes"|"no", ...},
         "task_performance": {...}}          # CoT only
```

**White-box detector output** (`detectors/{probe,logit}_<model>.jsonl`, rewritten by the fit):
the sample fields plus `detector`, `judge` (a `model_awareness` block with the thresholded
decision), `score` (the standardised out-of-fold score), `threshold` and `fold`. The probe adds
the layer read; the logit monitor adds `logprob`, `sentence` and `n_positions`. Thresholds and
layers are chosen out of fold on a task-grouped split (`src/cv/whitebox/classification.py`).

**Activations** (`whitebox/activations/<model>/`): sharded `.npz` files with one pooled vector
per stored layer, at the last prompt token (once per prompt) and as the mean over the response
(once per sample), plus the contrastive-pair vectors. **Logit scores**
(`whitebox/logit_<model>.jsonl`): the sentence log-probability at each scored position and the
maximum.

**Failure ledger** (`failures/<stage>_<model>.jsonl`): `{"ts", "id", "stage", "model", "reason",
"detail"}` per failed call. **Spend ledger** (`_metrics.jsonl`): one row per metered call or
GPU batch with tokens, seconds and USD.

## Known limitations

- **No human labels.** No detector is checked against human labels. The cue label from
  `config_name` is the only reference, and it is a proxy: a model can miss a cue or suspect a test
  without one.
- **The judge.** CoT is one judge model's verdict. A second judge, `openai/gpt-oss-120b`, re-judged
  a stratified subset with the same template (677 Gemma and 485 Qwen rollouts with a parseable
  verdict). The two judges agree at kappa 0.06 on Gemma and 0.04 on Qwen. On the subset the first
  judge flags 88.6 and 83.5 percent of rollouts, the second 19.5 and 10.7 percent. This shows that
  at least one judge errs on most of the subset. It gives no upper bound on either judge's error.
  The CoT judge's flags are not reliable (`design 0007`, `design 0021`).
- **Two targets.** Two models of about 30B parameters. Sensitivity may not generalise.
- **Gemma's re-encode.** Gemma's text comes from a provider's fp8 build and its activations from
  our own fp8 build of a pinned commit. The bit width matches; the quantization scheme may not
  (`design 0031`). Qwen's caveat is smaller: the two decay projections of each hybrid layer, 5120 x
  48, run in bf16 in the re-encode and at fp8 in generation (`design 0032`).
- **Gemma's white-box detectors are excluded from analysis.** Under fp8, the forward pass gave
  non-finite activations on 1,745 of Gemma's 12,589 rollouts, and the logit monitor returned a
  non-finite score on 1,577. Non-finite rows are more common in long rollouts and in a few tasks.
  The cause was not identified. The usable rows are a non-random subset, so the analyses use
  Gemma's three text detectors only. Gemma's probe and logit rows stay in the table. A post hoc
  bf16 re-encode of every Gemma prompt gives a probe AUROC of 0.596, inside the random-direction
  null (mean 0.566, maximum 0.662), so neither fp8 nor the missing rows explain Gemma's weak
  probe. Qwen has no non-finite row.
- **Floor channels.** A detector that flags under 5 percent of analysed rollouts is reported by
  its rate and leaves the agreement matrix, a rule fixed before the full run (`design 0030`). The
  rule excludes the CoT judge on both models.

## Storage and distribution

- **System of record:** the run's flat files under `data/results/runs/<run_id>/`, gitignored,
  with durable copies in W&B Artifacts (`operations 0009`).
- **Curated copies:** `reports/<run>/`, tracked.

## Maintenance

- **Owner:** the authors (`CITATION.cff`).
- **Updates:** regenerated by re-running the pipeline. Every stage is resumable and the merge
  is deterministic from the detector outputs.
- **Versioning:** tracked with the repository. The per-run `_config_snapshot.yaml` and the
  decision records under `docs/` pin how a run was produced.

## License

EvalAwareBench is released under Creative Commons Attribution-NonCommercial 4.0 (CC BY-NC 4.0).
This dataset is a derivative for non-commercial research and inherits that restriction for
benchmark-derived content (task prompts, model reasoning about them). Attribute the benchmark
authors (Li et al. 2026, arXiv:2605.23055; `CITATION.cff`). The code and the decision values
are ours, under the MIT License.

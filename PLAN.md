# Evaluation-awareness detectors: known groups and agreement within cue conditions

The specification of the study behind the short paper "Known-groups validity is not enough for
evaluation-awareness detectors" (Agent Evaluation Science symposium, AES Fall 2026). We run
**EvalAwareBench** through **two target models**, flag each rollout with up to **five
detectors** (three text detectors and two white-box detectors), and score the detectors against
the benchmark's **known cue labels** (`design 0026`). The study reports four results:

1. **The known-groups check.** Every analysed detector flags cued rollouts more often than
   baseline rollouts (d' > 0).
2. **Its weakness.** Random directions in Qwen's activations, put through the same tuning as
   the activation probe, also pass the check, so the check cannot tell good detectors from bad.
3. **Agreement within cue conditions (post hoc).** Cohen's kappa computed within cue conditions
   and read against kappa_max, the highest kappa the two positive rates allow. Self-report and
   third-person reach most of kappa_max. The Qwen probe agrees with none of the three other
   detectors it can be compared with.
4. **A positive control.** The probe recipe finds the implanted evaluation association of a
   model organism of Hua et al., though only narrowly ahead of random directions.

Convergent validity (agreement between methods on the same property) is the method behind
result 3. The benchmark loads from its Hugging Face dataset at a pinned revision, and the
rollout and detector mechanics are internalised in `src/cv/` (`operations 0005`).

This document has two parts. **Part I** states the experiment design. **Part II** states the
operations. The decision records under `docs/design/` and `docs/operations/` hold the
rationale for individual choices, and this plan cites them by number. Where the study as
carried out differs from a record, this plan states what was done. A choice that no record
covers is marked "(the paper; no record)".

# Part I: Experiment design

## Research question

On two open-weight models, does each evaluation-awareness detector separate cued from baseline
rollouts, does a null built from the probe's own fitting procedure pass the same check, and do
the detectors agree once the agreement that the cues alone create is removed?

## The construct (`design 0016`, `design 0024`, `design 0026`)

The target construct is **evaluation intent**: the model recognising that it is being assessed,
not merely that a prompt is artificial (`design 0016`). The text detectors measure
**verbalised** evaluation awareness and the white-box detectors measure **represented**
evaluation awareness. Neither is a claim about evaluation gaming (`design 0024`).

## Known groups from the benchmark (`design 0002`, `design 0026`)

EvalAwareBench adds test-like cues to otherwise unchanged prompts, so for every rollout we know
which cues the prompt carries but not whether the model noticed them. The cue label is a proxy
for awareness, and detectors are scored against it (`design 0002`).

- **Cued:** the prompt carries any of F1 to F7, in single-cue or multi-cue conditions.
- **Baseline:** the prompt carries no cue.
- **F8 is set aside.** The benchmark calls F8 explicit evaluation disclosure, but only 22 of the
  200 F8 prompts state that they are an evaluation. Most add only an output-format instruction.
  F8 prompts are neither cued nor baseline (the paper; no record). F8 also stands alone and is
  not stacked (`design 0029`).

The detection criterion is **any-type** awareness: either evaluation answer (safety or
capability) counts as aware (`design 0026`). The criterion and the factor classes live in
`src/cv/factors.py` (`design 0030`).

## Models under study (`operations 0013`, `design 0026`)

| Model | Serving | Detectors analysed |
|---|---|---|
| Qwen 3.6 27B (`qwen3.6-27b`) | self-hosted vLLM on Modal, fp8 weights, fp8 key-value cache | all five |
| Gemma 4 31B (`gemma-4-31b-it`) | OpenRouter, constrained to fp8 endpoints | the three text detectors |

Both models use the benchmark's realistic chat system prompt, keep their reasoning traces, have
a budget of 8,000 output tokens, and sample at temperature 0.3, the default in the benchmark's
code (`design 0014`). The serving split follows measured throughput economics
(`operations 0013`). Gemma's white-box detectors are excluded (see The white-box detectors).

## Design factors

| Factor | Levels | n | Field |
|---|---|---|---|
| Target model | gemma-4-31b-it, qwen3.6-27b | 2 | run config |
| Task | paired request scenarios | 100 | `task_name` |
| Variant | safety, capability | 2 | `valence` |
| Cue condition | baseline, F1..F8 alone, sampled stacks of F1..F7 | 9 + 48 | `config_name` |
| Repetition | sample_1..3 | 3 | `sample_idx` |

- **Experiment A:** all 200 task variants under the 9 single conditions, 3 repetitions: 5,400
  rollouts per model.
- **Experiment B:** 50 variants from 25 tasks under 48 sampled stacks of F1 to F7 (up to 10 draws
  for each stack size from two to seven, seed 0), 3 repetitions: 7,200 rollouts per model
  (`design 0029`, `src/cv/stacking.py`).

That is 12,600 planned rollouts per model. Qwen produced all 12,600 and Gemma 12,589. Three
repetitions per prompt follow `design 0014`.

## The detectors (`design 0019`, `design 0026`)

| Detector | Decision by | Input | Unit | Tuned on cue labels |
|---|---|---|---|---|
| Self-report | the target model | the task and its response, then a question on the setting | rollout | no |
| Third-person | the target model | the task and the visible answer, no reasoning trace (`design 0029`) | rollout | no |
| CoT judge | `xiaomi/mimo-v2.5-pro` | the task, trace and answer; decides on the trace (`design 0027`) | rollout | no |
| Activation probe | a learned threshold | activations at the last prompt token (`design 0018`) | prompt | layer, threshold |
| Logit monitor | a learned threshold | log-probability of "The user might be testing" (`design 0028`) | rollout | threshold |

- Self-report and third-person ask the upstream question with three answers (deployment, safety
  evaluation, capability evaluation) and the upstream prompts, with no confidence field
  (`design 0023`). They also ask whether the scenario was hypothetical. That answer is recorded
  and no analysis reads it (`design 0004`).
- **On Qwen, self-report reads the same text as third-person.** Our code sends the full response,
  trace included, as the earlier assistant turn (`design 0022`). Qwen's pinned chat template
  renders an earlier turn's reasoning only when `preserve_thinking` is set, and our server passed
  only `enable_thinking`, so the trace was dropped (the paper; no record). On Gemma the trace
  went inside `<think>` tags. The OpenRouter providers' copy of the template cannot be inspected,
  so the outputs neither confirm nor refute that Gemma read it.
- Purpose-guessing is not in the roster (`design 0019`).
- The agreement matrix is partitioned by decision-maker: judge, target, and activations or
  logits (`design 0011`).

## Analysis (`src/cv/analysis.py`)

All outputs are scored against the cue labels. No behavioural proxy and no metacognitive
statistic is used (`design 0001`, `design 0023`).

1. **Per-detector sensitivity (planned).** For each detector and model: hit rate H (positive rate
   on cued rollouts), false-alarm rate F (positive rate on baseline rollouts), and
   d' = Phi^-1(H) - Phi^-1(F). Rates use the correction (k + 0.5)/(n + 1) and carry exact binomial
   intervals. d' carries a task-cluster bootstrap interval (1,000 resamples, both variants of a
   task resampled together). The count rule and the reasons for an undefined d' are fixed in
   `design 0030`. The per-factor grid and the dose-response over stacked cues are descriptive
   (`design 0029`, `design 0030`).
2. **Overall agreement (planned).** Cohen's kappa over all analysed rollouts, with a task-cluster
   bootstrap interval (500 resamples) (`design 0005`, `design 0011`, `design 0026`). A rule fixed
   before the full run excludes any detector that flags under 5% of the analysed rollouts,
   because kappa is unreliable at such rates (`design 0030`). The rule excludes the CoT judge (4.95%
   on Gemma, 3.11% on Qwen).
3. **Agreement within cue conditions (post hoc; the paper; no record).** The stratified kappa sums
   observed and chance agreement over cue conditions, each from that condition's positive rates,
   so a shared response to the cues counts as chance. kappa_max is the highest kappa the two
   positive rates allow, computed within each condition. kappa is reported as a percentage of
   kappa_max, which splits the disagreement into the part forced by unequal positive rates
   (1 - kappa_max) and the remaining disagreement (kappa_max - kappa). The test-retest kappa
   between repetitions of one prompt, the conditional-independence kappa, the shared-positive
   share, and a stratification by cue condition and task variant together are post hoc as well.
4. **Probe validity checks (`design 0033`).** A random-direction null (24 unit directions, seed 0,
   each drawn per layer and put through the probe's layer and threshold choice), the cue-trained
   direction (mean cued minus mean baseline, recomputed in each training fold), a
   logistic-regression probe at the chosen layer (`design 0018`), and the F8 control. All are
   descriptive.

No multiplicity correction is applied (`design 0030`). There is no calibration output: the
elicited Likert is removed from the study (`design 0023`). The task-performance verdict is
recorded and no analysis reads it.

### Outcome

Exact values are in `reports/full/analysis.json` and `reports/full/fit_*.json`.

- All 8 analysed detector and model combinations have a d' interval above zero (d' from 0.68 to
  1.60).
- On Qwen, the probe's contrastive direction reaches an out-of-fold AUROC of 0.783 at layer 15 of
  64. Of the 24 random directions, 11 score higher, and the lowest scores 0.577.
- Within cue conditions, self-report and third-person reach 88% (Gemma) and 60% (Qwen) of
  kappa_max, which is 0.43 on both models. The Qwen probe's stratified kappa with self-report,
  third-person and the logit monitor is at or below zero, and so are the upper bounds of the
  intervals.

### Judge validation (`design 0007`, `design 0027`)

- **Judge mechanics** are upstream's, unchanged: input shape, rubric and verdict definition
  (`design 0021`, `design 0027`).
- **Judge-vs-judge.** A second judge from a different family, `openai/gpt-oss-120b`, re-judged a
  stratified subset (weighted toward cue-rich conditions, with every first-judge positive) with the
  same template (`cv-judge2`; the second-judge model is the paper's, no record). The judges agree
  at kappa 0.06 on Gemma (677 rollouts) and 0.04 on Qwen (485). On this subset the first judge
  flags 88.6% and 83.5%, the second 19.5% and 10.7%. The CoT judge's flags are not reliable.
- **Human grounding was not done.** `design 0020`, `design 0021` and `design 0027` require the
  CoT judge to be grounded against human labels. That step was not carried out, and no detector
  is checked against human labels.

## The white-box detectors (`design 0015`, `design 0018`, `design 0028`, `operations 0008`)

Both channels read internals, so they need open weights and a forward pass under our control.
Each emits one `(decision, score)` per row, so the d' and kappa analyses take them unchanged.

- **Re-encode (`operations 0008`).** Rollouts are saved as text. Each saved rollout is fed back
  through the model with teacher forcing, and the pass records pooled per-layer activations and
  the logits the monitor reads. Both models use their weights at a pinned commit in fp8
  (`design 0031`). The output head and every linear layer that fp8 blocks cannot tile stay in
  bf16 (`design 0032`). A non-finite vector is a failed row in the failure ledger
  (`design 0032`).
- **Activation probe (`design 0018`).** A difference-of-means contrastive direction from the 16
  prompt pairs of Hua et al., normalised to unit length and read at the last token of each
  prompt, written as a user message, at every fourth layer. Within 5 training folds grouped by
  task, the layer and the threshold that maximise balanced accuracy for `score >= threshold` are
  chosen and applied to the held-out tasks. The probe reads only the prompt, so all repetitions
  of a prompt share one prediction. The study uses the direction for detection only, not for
  steering.
- **Logit monitor (`design 0028`).** The log-probability of "The user might be testing" at up to 8
  positions (the start of the reasoning trace and later sentence boundaries), maximum over
  positions, one score per rollout. Its threshold is chosen the same way as the probe's.
- **Gemma is excluded.** The Gemma forward pass gave non-finite activations on 1,745 of 12,589
  rollouts, and the logit monitor failed on 1,577. The failures are more common in long rollouts
  and in a few tasks, and their cause was not identified. The usable rows are a non-random subset,
  so Gemma's white-box detectors are excluded from the analysis (the paper). `design 0034`
  planned to report the channel with the gap disclosed; the paper excludes it instead. Qwen had
  no non-finite rows.
- **Gemma in bf16 (post hoc).** Gemma's prompts were re-encoded in bf16: 4,199 of 4,200 prompts
  gave finite activations, at a median cosine of 0.98 to 1.00 per layer against fp8. The probe's
  out-of-fold AUROC is 0.596 at layer 3, inside the random-direction null (mean 0.566, highest
  0.662). Neither fp8 nor the missing rollouts explain Gemma's weak probe
  (`reports/full/PROBE_DIAGNOSIS.md`).
- **Positive control on a model organism (`design 0018`).** The adapter
  `timhua/wood_v2_sftr4_filt` of Hua et al. on Llama 3.3 Nemotron Super 49B, read in bf16 at every
  second of 80 layers with the adapter on (organism) and off (base model), on the five prompt
  conditions of Hua et al. over 128 coding questions. The direction is built from the 16 pairs on
  the base model and read at the last prompt token. The layer (51) is chosen on stated evaluation
  against stated deployment only. On Wood Labs prompts with neutral user text and the hint on, the
  organism sits 0.87 [0.75, 1.00] of the stated-deployment-to-stated-evaluation gap above the
  base model. Of 1,000 random directions with the same layer choice, 4.6% reach this value. The
  analysis was written before the data existed; the random-direction null is post hoc
  (`reports/full/ORGANISM_CONTROL.md`).

## Limitations

These match the paper's Limitations section.

- No detector is checked against human labels. The cue label is the only reference, and it is a
  proxy: a model can miss a cue or suspect a test without one, and a safety variant can read as a
  test. Like the benchmark, we presume the cues describe what evaluation prompts look like, and
  we use no production traces.
- The probe's layer is tuned on cues F1 to F7, the same cues that measure its response.
- The stratified analyses are post hoc, and most cued Qwen rollouts carry several cues (7,200
  multi-cue of 11,400 cued).
- On Qwen, self-report reads the same text as third-person.
- We did not check that the re-encode reproduces the internal values the model had while
  generating.
- The scope is two models, one benchmark and three repetitions per prompt.

# Part II: Operations

## Execution (`operations 0010`)

1. **Pilot run.** Experiment A on both models with the three text detectors. Qwen's pilot
   rollouts are generated on the same serving setup and precision as the full run, and they are
   Qwen's 5,400 baseline and single-cue rollouts in the full run. Gemma's pilot rollouts were
   self-hosted in bf16 and are not used (`operations 0013`).
2. **Full run, text stage.** Gemma's Experiment A and both models' Experiment B rollouts, then
   self-report, third-person, the CoT judge and the merge (`cv-generate`).
3. **Full run, white-box stage.** The re-encode, the probe and logit-monitor fit, and the merge
   (`cv-whitebox`), then the second judge (`cv-judge2`), then the analysis.
4. **Post hoc checks.** The Gemma bf16 prompt re-encode and the organism control, each in its own
   GPU session (`scripts/exploration/`).

## Calls per rollout

| Step | Calls | Engine | Parsed by |
|---|---|---|---|
| Task rollout | 1 | target | (produces trace and answer) |
| CoT judge | 1 | judge | judge JSON |
| Self-report | 2 (awareness, hypothetical) | target | regex |
| Third-person | 2 (awareness, hypothetical) | target | regex |

Per rollout: 5 target calls and 1 judge call (`design 0019`). The white-box detectors add no call
at rollout time; they re-encode the saved text later (`operations 0008`).

## Architecture

The benchmark prompts load from the Hugging Face dataset `aisa-group/EvalAwareBench` at a pinned
revision. The rollout, detector and judge mechanics are ported into `src/cv/` and held to the
upstream protocol by parity tests (`operations 0005`). The wire schemas are in `DATACARD.md`.

- `bootstrap.py`: loads secrets and the run profile, routes the judge, installs the meter.
- `config.py`: the OmegaConf-backed config, the price table, the GPU rates, the paths.
- `dataset.py`: the pinned benchmark revision. `stacking.py`: the Experiment B sampler
  (`design 0029`).
- `registry.py`: the detector registry (decision-maker, prompt files).
- `serving/`: the target backends (`operations 0007`, `operations 0013`). `api.py` is OpenRouter
  with the fp8 provider filter and the price ceiling. `local/` is the Modal app: the vLLM engine
  for generation and the re-encode engine for activations and logits. `organism/` serves the
  organism control.
- `rollout.py`, `run_{cot,sr,tp}.py`: generation and the three text detectors, with per-sample
  checkpoints and the failure ledger (`failures.py`). `judge2.py`: the second judge.
- `whitebox/`: the contrastive pairs, the re-encode step, the activation store, the out-of-fold
  fit (`design 0018`, `design 0028`, `design 0033`, `operations 0008`).
- `merge.py`: joins detector outputs per rollout id into one table.
- `factors.py`: the criterion and the factor classes (`design 0026`, `design 0030`).
- `analysis.py`, `report.py`, `plots.py`: the outputs and the judge validation;
  `notebooks/analysis.py` reads them.
- `meter.py`: the spend ledger for API tokens and GPU seconds, under the hard cap.

### Judge routing

Upstream creates the judge client as `AsyncOpenAI(api_key=...)` with no `base_url`, so the OpenAI
SDK falls back to `OPENAI_BASE_URL`. The judge routes to OpenRouter through the environment alone:

```
OPENAI_API_KEY=<openrouter key>
OPENAI_BASE_URL=https://openrouter.ai/api/v1
OPENAI_EVAL_MODEL=xiaomi/mimo-v2.5-pro   # the profile's judge_model maps to this
```

The judge uses the non-batch path, because OpenRouter has no OpenAI Batch API. The second judge
reads `JUDGE2_MODEL` (default `openai/gpt-oss-120b`).

## Compute and serving (`operations 0007`, `operations 0013`)

Gemma 4 self-hosted in bf16 reached only about 750 tok/s on an H200 (its heterogeneous head
dimensions force vLLM's slow Triton attention), so self-hosting it costs several times its API
price. Gemma generation is therefore billed per token on OpenRouter's fp8 endpoints, and Qwen
self-hosts on Modal (`operations 0013`). GPU cost scales with wall-clock time, so the serving
settings maximise tokens per GPU-second: fp8 weights, `kv_cache_dtype=fp8`,
`gpu_memory_utilization` near 0.95, right-sized `max_num_seqs` and `max_model_len`, and a warm
image with a persisted `torch.compile` cache. Qwen fp8 decode measured 2,026 tok/s on an H100 (`operations 0019`). The
white-box pass runs on the same app's re-encode engine.

## Configuration and run profiles

Run settings are layered YAML loaded by `cv/config.py` through OmegaConf: schema defaults (typed
`CvConfig`), then `config/base.yaml`, then the profile chosen with `CV_PROFILE`, then optional
dotlist overrides. The profiles are `smoke`, `smoke-qwen`, `pilot`, `prefull` and `full`.

- **Secrets are never in YAML.** Keys stay in the gitignored root `.env`. Config may reference an
  environment value with `${oc.env:VAR}`, and the per-run snapshot dumps with `resolve=False`.
- **Per-run output** goes under `data/results/runs/<run_id>/` (`operations 0003`), so runs do not
  overwrite each other.
- **Reproducibility:** the resolved config is written to `_config_snapshot.yaml` per run.

## Monitoring, checkpointing and the ledger (`operations 0001`, `operations 0009`)

- `cv/meter.py` wraps each API call, reads `response.usage` and appends to `_metrics.jsonl` in USD
  from verified prices. It also meters self-hosted GPU time at `config.GPU_RATES_PER_HR`, so the GPU
  and API lines share one ledger and one `budget_usd` cap.
- `cv-status` prints done and total, tokens, USD, mean USD per rollout, and an ETA. It makes no
  API call.
- **Resumable runs (`operations 0009`).** Every stage persists per chunk, the serving engines
  retry once on Modal preemption, and W&B Artifacts are the durable ledger
  (`cv-results-<run_id>`), restored on a fresh machine. Resume skips completed ids. Local files
  are a disposable cache rebuildable from W&B. All W&B calls are no-ops without `WANDB_API_KEY`.

## Step 0 validation gate

Before any paid run, `cv-validate` checks every assumption against each model for a few cents.

| Check | Asserts | Failure it catches |
|---|---|---|
| Reasoning present | `reasoning` or `<think>` non-empty | empty trace, judge always "not aware" |
| Answer present | non-empty content after `</think>` | max_tokens truncation |
| Self-report label parses | the forced answer yields a label | silent default to "no" |
| Safety not auto-blocked | a safety prompt returns content | moderation removing the safety variant |
| Judge JSON | the judge returns valid JSON | judge routing or format failure |
| Usage present | `response.usage` populated | the meter silently records $0 |

A model that fails "reasoning present" stops the run. The Modal deploy uses the line that
`cv-deploy-command` prints for the profile, because a bare `modal deploy` drops every vLLM setting
(`operations 0007`).

## Cost and storage (`operations 0019`, `operations 0021`)

- **The full run cost $217.31 metered against a $300 cap.** The largest line is the CoT judge
  ($81.62), then the logit monitor ($41.12), the self-report and third-person follow-ups ($31.23
  and $29.84), the re-encode ($15.75) and the rollouts ($15.23). The costed worst case before the
  run was $274 (`operations 0019`). Modal billed $117.05 for the run's days against $103.19
  metered on the same days, from container time the driver cannot see.
- **Hard caps.** The in-run `budget_usd` cap stops the run at $300, and a Modal workspace spend cap
  set in the dashboard catches GPU spend that the meter does not see, such as a raw `modal run`
  (`operations 0019`).
- **Storage.** W&B Weave holds traces and W&B Artifacts hold the durable ledger (`operations 0001`,
  `operations 0009`). The flat files in `data/` are the local cache. Activations are pooled
  per-layer vectors, gigabytes rather than terabytes (`operations 0008`). Curated results are in
  `reports/full/`.

## Run order

1. `cv-validate` under the profile.
2. Deploy the profile's Modal app with the line `cv-deploy-command` prints.
3. The `prefull` smoke test: `cv-generate`, then `scripts/prefull_check.py`, then
   `cv-whitebox`, capped at $8. It verifies the engine build, the pins and the white-box path on
   both targets.
4. The full run, detached: `cv-generate`, then `cv-whitebox`, then `cv-judge2`, then the analysis
   notebook.

## References

- **Li, Zhang, Zhang, Jin, Tramèr, Abdelnabi, Andriushchenko (2026). Decomposing and Measuring
  Evaluation Awareness.** arXiv:2605.23055. The benchmark (EvalAwareBench), its eight cues, the
  self-report question, the third-person detector this study adapts, and the judge template.
- **Hua, Qin, Marks, Nanda (2025). Steering Evaluation-Aware Language Models to Act Like They Are
  Deployed.** arXiv:2510.20487. The contrastive prompt pairs, the difference-of-means direction,
  the random-vector control for steering that the probe null adapts, and the model organism of
  the positive control (`design 0018`).
- **Aranguri (2026). Logits as a new monitor for evaluation awareness.** LessWrong. The logit
  monitor (`design 0028`).
- **Campbell and Fiske (1959). Convergent and discriminant validation by the multitrait-multimethod
  matrix.** Psychological Bulletin 56(2). Convergent validity and test-retest reliability as its
  reference.
- **Hattie and Cooksey (1984). Procedures for Assessing the Validities of Tests Using the
  "Known-Groups" Method.** Applied Psychological Measurement 8(3). The known-groups check.
- **Feinstein and Cicchetti (1990); Byrt, Bishop and Carlin (1993).** Journal of Clinical
  Epidemiology. Kappa under unequal positive rates, behind kappa_max.
- **Needham, Edkins, Pimpale, Bartsch, Hobbhahn (2025). Large Language Models Often Know When They
  Are Being Evaluated.** arXiv:2505.23836. Evaluation awareness as transcript classification, and
  precedent for repeated sampling (`design 0014`).
- **Vishwarupe, Shadbolt, Jirotka, Flechais (2026). The Evaluation Differential.**
  arXiv:2605.11496. Supports the task-cluster bootstrap (`design 0026`).

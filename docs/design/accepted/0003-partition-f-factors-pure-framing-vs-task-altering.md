# 0003. Partition F1..F8 into pure-framing vs task-altering; clean criterion is {F3, F7}

- **Status:** Superseded by `design 0026` (the clean Type-1 criterion `baseline` vs `{F3,F7}` is
  retired; the full run pools all factors factor-agnostically. The pure-framing vs task-altering
  partition survives only as a descriptive diagnostic and for the F8 carve-out.)
- **Date proposed:** 2026-05-15
- **Date accepted:** 2026-05-17
- **Supersedes:** n/a

## Context

DDR 0002 fixed the Type-1 anchor to `config_name`. But the eight F-factors in
EvalAwareBench are not equally clean as cue indicators. Some toggle pure
framing (rewording, reformatting) while leaving the request, its content, and
its instruction set intact; others change the task itself (content, context,
instruction set, valence-dependent wrappers).

A factor that *alters the task* introduces a direct path from the cue
manipulation to the detector's decision that is not "recognition". If F8
literally adds "this is a red-team evaluation item" to the prompt, then a
judge-mediated detector reading the prompt will mark it aware not because the
model recognised an eval, but because the cue text is in the input string. The
sensitivity claim becomes circular for those configs.

Verification against the rendered task YAMLs gave:

- **F3** (casual → textbook-formal register) and **F7** (prose → structured
  template format): same request, only reworded or reformatted. Pure framing.
- **F1** (real entities → placeholder entities): borderline content edit.
- **F2** (false premise), **F4** (stripped context), **F5** (stakes
  telegraphed), **F6** (jailbreak/role prefix; valence-dependent), **F8**
  (eval-disclosure suffix that adds an instruction and sometimes literally
  says "this is a red-team evaluation item"): change the task.

## Decision

- The **clean cue criterion** is `baseline` (negative) vs `{F3, F7}`
  (positive). v1 sensitivity (DDR 0001, DDR 0002) is computed against this
  criterion.
- **F1** is borderline; reported separately as a robustness arm, not in the
  clean criterion.
- **F2, F4, F5, F6, F8** are task-altering; reported in the per-factor sweep
  for description, never in the clean criterion. F8's per-factor d' is
  explicitly flagged as partially circular for the judge-based detectors.
- The classification lives in `cv/factors.py` with a rationale field for each
  factor. The file is itself a reviewable design artefact.

## Alternatives considered

- **Treat all F1..F8 as one positive class.** Rejected: inflates apparent d'
  via the task-altering factors (especially F8) and makes the resulting
  sensitivity claim non-recognition.
- **Drop F1..F8 entirely and build our own framing manipulation.** Rejected
  for v1: discards the benchmark's known truth; constructed framing is the
  natural mid-tier extension (see DDR 0013 proposed).
- **Treat F1 as clean.** Rejected: real-to-placeholder is a mild content edit
  not a framing one; including it would mix construct types.

## Consequences

### Enables
- Headline d' against a defensible recognition criterion.
- A per-factor sweep that lets a reviewer see exactly which cue type each
  detector responds to.

### Forbids
- Reading F8 as evidence about recognition. F8's d' tells us about prompt-text
  detection, not eval-awareness.
- Pooling F1..F8 as a positive class anywhere in v1.

### Risks
- The clean criterion is narrow: only two factors, so even at full-scale n
  the sensitivity ranking effectively measures "detects formal register or
  template format", not generic eval-awareness. The pilot shows TP's clean d'
  is dominated by F3 (d'=1.69) over F7 (d'=0.85). DDR 0013 proposed addresses
  widening the clean criterion via a constructed pure-framing cue.

## References

- `cv/factors.py` (the classification + per-factor rationale).
- `results/PILOT_SUMMARY.md` §1 (per-factor d' table), §2 (CoT positives on
  F6/F8 are prompt-driven).
- `PLAN.md` lines 27-32 ("Caveat (from causal review): some F-factors
  plausibly alter the TASK itself..."), lines 304 ("...the factor
  classification held up").
- DDR 0002 (config_name as Type-1 ground truth);
  DDR 0013 proposed (widen clean criterion with constructed cue).

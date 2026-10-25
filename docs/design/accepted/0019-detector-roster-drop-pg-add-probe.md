# 0019. Final detector roster: CoT, SR, TP (behavioral) plus the white-box probe; drop PG

- **Status:** Accepted (amended by `design 0026` for scope: the roster adds a logit-monitoring
  detector (five total: CoT, SR, TP, linear probe, logit monitor); CoT-floor handling moves to the
  pre-registered post-hoc rule in `design 0026`, replacing the "0009 rehabilitates CoT" framing. PG
  stays dropped.)
- **Date proposed:** 2026-06-07
- **Date accepted:** 2026-06-07
- **Supersedes:** closes the open roster decision in `docs/design/accepted/0015`; consistent with
  `0011` (decision-maker partition) and `0016` (construct)

## Context

`docs/design/accepted/0015` added the white-box probe for the funded run and explicitly left the
roster open: "Finalize the exact full-run detector roster from the pilot results ... the
behavioral pilot may justify consolidating the near-floor detectors (CoT, PG)." The original
design ran four behavioral detectors (CoT, SR, TP, PG). The 2026-05-31 pilot
(`results/runs/pilot-2026-05-31/PILOT_SUMMARY.md`) gives the evidence to settle it:

- **CoT and PG both sit near the detection floor** against the clean cue; TP is the only
  behavioral detector that tracks the framing cue (d' ≈ 1.19, capped by a ~40% baseline
  false-alarm), and SR is weak (§1).
- **PG is judge-robust** (judge-vs-judge κ = 0.86 against an independent `gpt-oss-120b` judge,
  §5) but that reliability is moot: it detects almost nothing beyond baseline.
- **PG is the most expensive channel per rollout**: one target purpose-guess call *and* one
  judge call (`PLAN.md`, "How calls are produced per rollout").
- **PG is judge-mediated, the same decision-maker class as CoT** (`0011`), so it adds a redundant
  judge channel without adding detection signal.
- **CoT's pilot floor was partly an anchoring artifact**: the judge labeled from injected prompt
  text (the F6 "Developer Mode" / F8 "controlled evaluation item" phrases) rather than from the
  model's reasoning (§2). `0009` (anchor the CoT verdict in trace text) addresses this, and on the
  Modal serving regime (`operations/accepted/0007`) gemma's reasoning trace is available, so CoT is
  not void at serving time.

## Decision

The detector roster is **three behavioral detectors (CoT, SR, TP) plus the white-box activation
probe** (`0015`/`0018`). **PG is dropped.**

- **Keep CoT:** the flagship direct-trace channel and the behavioral analogue of the probe; the
  most safety-relevant read. Its pilot weakness is a fixable anchoring artifact (`0009`) and is
  resolved by serving on Modal where the trace is available (`operations/0007`).
- **Keep SR and TP:** the two target-decided introspection channels (`0011`); TP is the strongest
  behavioral detector, SR the cheap self-report baseline.
- **Drop PG:** near-floor detection, highest per-rollout cost, judge-mediated (redundant with CoT
  in the decision-maker partition), and the weakest construct link (two-step
  infer-motivation-then-judge). Judge-robustness does not rescue a channel that detects nothing.
- The probe is the mechanistic channel added for the experiment phase. With PG gone the convergent
  surface is three behavioral detectors plus one mechanistic, and the decision-maker partition
  (`0011`) reads judge (CoT) / target (SR, TP) / activations (probe).

## Alternatives considered

- **Keep all four (status quo).** Rejected: PG carries an extra target call and judge call per
  rollout for no detection signal; "four detectors" is harder to defend than a roster justified by
  pilot evidence.
- **Drop CoT too (SR/TP + probe only).** Rejected: CoT is the flagship trace read and the
  behavioral analogue of the probe; its pilot floor is a fixable anchoring artifact (`0009`), not an
  intrinsic dead end, and `0011` already rejected dropping it from the kappa panel.
- **Drop PG but keep it as a recorded-only sidecar.** Rejected: it still costs the extra
  target + judge calls at collection time; the recorded value is not worth the per-rollout spend
  (the simplicity principle: no exotica without need).

## Consequences

### Enables
- A roster justified by pilot evidence, not convention; a clean 3-behavioral + 1-probe convergent
  matrix whose decision-maker partition (`0011`) reads judge / target / activations.
- Lower per-rollout cost: one fewer target call and one fewer judge call per rollout.

### Forbids / constrains
- No v1 claim involving PG. The merged-table schema drops the PG columns (back-compat: pilot rows
  that carry PG are read but PG is not analyzed).
- Reintroducing PG later is a new record, with its own cost/justification.

### Risks
- CoT remains the floor-risk behavioral detector; mitigated by `0009` anchoring + Modal trace
  availability, and its reliability is underwritten by judge-vs-judge (`0007`). If CoT stays at the
  floor on the full run, that is itself a reportable convergent/validity finding, not a pipeline
  failure.

## References

- `docs/design/accepted/0015` (roster left open: closed here), `0006` (original four-detector
  api-only), `0005` (kappa matrix the roster feeds), `0011` (decision-maker partition:
  judge/target/activations), `0009` (CoT trace-anchoring that rehabilitates CoT), `0007`
  (judge-vs-judge underwriting CoT), `0016` (construct), `0018` (probe read-out).
- `docs/operations/accepted/0007` (Modal serving: gemma CoT available); `PLAN.md` ("How calls are
  produced per rollout": PG was the most expensive).
- `results/runs/pilot-2026-05-31/PILOT_SUMMARY.md` (§1 sensitivity, §2 CoT anchoring artifact,
  §5 judge-vs-judge κ = 0.86 for PG).

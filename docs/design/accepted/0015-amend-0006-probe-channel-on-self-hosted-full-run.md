# 0015. Amend 0006: the funded full run adds a white-box probe channel on self-hosted open weights

- **Status:** Accepted
- **Date proposed:** 2026-06-04
- **Date accepted:** 2026-06-04
- **Supersedes:** amends `docs/design/accepted/0006` for the full-run phase (0006 stays in force for the Phase A API pilot)

## Context

`docs/design/accepted/0006` fixed v1 as API-only via OpenRouter with four behavioral detectors and explicitly deferred activation probes: "Activation probes need open weights plus forward hooks, which OpenRouter does not expose ... out of scope for v1 ... recorded as a known fork." `docs/operations/accepted/0002` identified self-hosting the open-weight targets as the single biggest cost lever but gated it on "an amendment to DDR 0006."

The program is now explicitly two-phase (`docs/operations/accepted/0004`): a near-term API behavioral-only pilot for the internal grant report, then a funded full run. The full run is planned on GPU compute, and it is intended to serve the targets self-hosted and to add the white-box probe channel that 0006 deferred. The pilot's behavioral results make this material: only the SR-TP pair agrees, CoT and PG sit near the floor, so the convergent-validity surface is currently characterising behavioral detectors only. The hardest-to-evade channel is exactly the one 0006 left out, and on a GPU serving regime it becomes available.

## Decision

For the **funded full run only** (Phase B of `docs/operations/accepted/0004`), amend 0006: serve the three open-weight targets self-hosted (open weights plus forward hooks) and add the **white-box activation-probe detector** as an additional channel alongside the behavioral detectors. The probe emits the same `(decision, confidence)` tuple per rollout, so the existing d-prime / agreement / calibration frame accepts it unchanged (`PLAN.md`, "Forward compatibility: internal-activation probes"). The **grant budget is scoped to this phase** (GPU compute plus the probe work), not to the API pilot, which stays under 0006 as-is.

Finalize the **exact full-run detector roster from the pilot results.** The stated intent is a focused set ("three detectors including probes"); the behavioral pilot may justify consolidating the near-floor detectors (CoT, PG) rather than carrying all four. Pin the roster, the probe's training and read-out method (e.g. SAE feature vs raw-activation linear probe), and the probe's own replication count as a follow-on decision before the full run.

## Alternatives considered

- **Keep API-only and behavioral-only for the full run (0006 unchanged).** Lost: forgoes the GPU cost lever and the hardest-to-evade detector; a behavioral-only agreement surface cannot speak to behavioral-versus-mechanistic convergence, which is the more interesting claim.
- **Add the probe in the API pilot.** Lost: OpenRouter exposes no activations; impossible without self-hosting.
- **Self-host without adding the probe (cost saving only).** Lost: pays the serving-regime complexity without buying the new channel; if we move to GPU, the probe is the reason to.

## Consequences

### Enables
- A convergent-validity claim spanning behavioral and mechanistic detectors, not behavioral-only.
- Flat GPU-hour economics that make reps and tasks cheaper at the margin (reps stay at 3 per `docs/design/accepted/0014` on statistical, not cost, grounds).

### Forbids
- Pooling the API pilot's behavioral rollouts with the self-hosted full-run rollouts as if the serving stack were identical (`docs/operations/accepted/0002` hygiene; a pilot-versus-full serving confound). The pilot is a pipeline and validity preview, not data to merge into the full run.

### Risks
- The white-box probe is a second infrastructure with its own failure modes and validation burden; the probe's reliability must be established on its own terms and cannot inherit the judge-vs-judge protocol (`docs/design/accepted/0007`).
- GPU decode throughput is unmeasured (`operations/0002` Risk); a short A100 throughput probe is warranted before committing GPU-hours.

### On acceptance
- `docs/design/accepted/0006` Status is annotated as amended by this record for the full-run phase; it remains the governing record for the Phase A API pilot.

## References

- `docs/design/accepted/0006` (the amended decision); `docs/operations/accepted/0002` (self-host lever and the gating clause); `docs/operations/accepted/0004` (the two-phase structure); `docs/design/accepted/0014` (reps, unchanged by the serving regime).
- `PLAN.md`, "Forward compatibility: internal-activation probes".
- `results/runs/pilot-2026-05-31/` (the behavioral-only agreement that motivates the probe).
- **Follow-on now decided:** `docs/design/accepted/0018` (probe read-out method: difference-in-means direction + logistic cross-check, detection only), `docs/operations/accepted/0008` (activation storage: decoupled re-encode, pooled per-layer vectors), and `docs/design/accepted/0019` (the detector roster left open here: drop PG, keep CoT/SR/TP + the probe).

# 0004. Two-phase execution: a rigor-improved API pilot now, a self-hosted GPU full run under the grant

- **Status:** Accepted: now **historical**. Phase A shipped (the 2026-05-31 gemma reanalysis + the
  gemma deployment-negative increment, both committed). The "API-pilot → GPU-full" *ordering* is
  retired by `docs/operations/accepted/0010`; the forward program is the subset/experiment two-tier
  on Modal (`docs/operations/accepted/0007`). The API pilot is a historical preview, not pooled into
  either Modal phase. Phase B's intent (self-hosted full run + probes) carries forward into 0007/0010.
- **Date proposed:** 2026-06-04
- **Date accepted:** 2026-06-04
- **Supersedes:** n/a (revises the all-API full-run default in `docs/operations/accepted/0002`)

## Context

This record fixes the program structure that the replication-strategy decision (`docs/design/accepted/0014`) and the probe amendment (`docs/design/accepted/0015`) assume. Two questions forced it: what the near-term deliverable is, and what the grant pays for.

The near-term deliverable is an internal report for a grant application. It is built from the existing behavioral pilot (`results/runs/pilot-2026-05-31/`: 1 model, 10 tasks, 2 reps, $2.92), improved on rigor rather than scaled up. The funded work is a later, larger run, and the funder route is very likely GPU/compute credits rather than API dollars. `docs/operations/accepted/0002` currently defaults the full run to all-API and leaves self-hosting as a gated option; that default no longer matches the plan.

## Decision

Run the program in two explicitly separated phases.

**Phase A (now, API): a rigor pass on the existing pilot, not a larger pilot.** The 2026-05-31 data stays the spine; we raise its defensibility for the internal grant report at minimal spend, via reanalysis first and at most one small same-order fresh increment.
- *Reanalysis only (about $0, no new rollouts):* wire in the kappa-paradox decision-maker partition and PABAK (`docs/design/accepted/0011`), the CoT trace-anchoring relabel from stored judge evidence (`docs/design/accepted/0009`), the judge-vs-judge validation from the already-collected second-judge outputs in `results/runs/pilot-2026-05-31/judge2/` (`docs/design/accepted/0007`, currently reported in prose but hardcoded `available:false` in `analysis.json`), and a test-retest / decoding-stability read from the existing 2 reps.
- *Optional single fresh increment (about $2, same order as the original $2.92):* the deployment-traffic negative (`docs/design/accepted/0010`) on gemma (~200 WildChat prompts), yielding the d-prime-bench vs d-prime-deployment two-column result and directly answering the pilot's largest caveat (baseline reads as eval 35 to 40% of the time). Any fresh rollouts use reps = 3 (`docs/design/accepted/0014`).

Four behavioral detectors only, no probes (`docs/design/accepted/0006` in force here). Purpose: a more defensible internal preview and a de-risked analysis pipeline, at minimal spend, inside the existing key.

**Phase B (down the line, GPU, grant-funded).** The full run with targets self-hosted on GPUs and the white-box probe channel added (`docs/design/accepted/0015`). 100 tasks, reps = 3 (`docs/design/accepted/0014`), all mitigations at full scale, aimed at a peer-reviewed safety venue. Rigor leads the scope here, not budget.

**Scope the grant budget to Phase B** (GPU compute plus probe work). Phase A is a near-zero API spend that precedes and justifies the grant, not part of it.

## Alternatives considered

- **One-shot full run, no intermediate pilot.** Lost: the grant needs a preliminary internal report, and a 10-task shakedown alone is `ci_unstable` and not credible for it without the rigor pass.
- **Scale the pilot up (more tasks/models) before the grant.** Lost: spends API budget on data that cannot be pooled with the self-hosted full run (`docs/design/accepted/0015` serving confound), so it is a sunk cost; the existing data plus reanalysis rigor is enough to anchor an internal report, and task-scale power is what the funded run buys.
- **Keep `operations/0002`'s all-API full-run default.** Lost: the funder route is GPU/compute, and the probe channel (`docs/design/accepted/0015`) requires self-hosting anyway; all-API for the full run is no longer the plan.
- **Do the pilot on GPU too, for serving consistency with Phase B.** Lost: API is cheaper and faster for a behavioral-only pilot that deliberately excludes probes; the no-pooling hygiene already keeps the phases separate.

## Consequences

### Enables
- A cheap, fast, defensible internal report (Phase A) decoupled from the funded, rigorous, mechanistic full run (Phase B).
- A clear story for the funder: the budget is for GPU plus probes, down the line, not for the pilot.
- Most of Phase A is analysis code (`cv/analysis.py`, `cv/plots.py`) over data already on disk, so it ships without new API spend.

### Forbids
- Merging Phase A and Phase B rollouts into one analysis; they run on different serving stacks (`docs/operations/accepted/0002` hygiene plus `docs/design/accepted/0015`). Each phase is namespaced as its own run (`docs/operations/accepted/0003`).
- Treating a scaled-up API pilot as progress toward the full run; if more behavioral scale is wanted it is its own deliverable, not poolable Phase B data.

### Risks
- Provider or sampling drift on the API pilot, bounded by the meter and a re-runnable Step-0 validation (`docs/design/accepted/0006`).
- GPU decode throughput for Phase B is unmeasured (`operations/0002` Risk); a short A100 throughput probe is warranted before committing GPU-hours.

### Triggers
- `docs/design/accepted/0015` (the probe amendment to 0006) and closes the open "all-API for the full run" default in `docs/operations/accepted/0002` in favour of self-hosting for Phase B.

## References

- `docs/operations/accepted/0002` (cost model, self-host lever, the all-API default this revises); `docs/operations/accepted/0003` (per-run namespacing keeps the phases separate).
- `docs/design/accepted/0006` (API-only and defer-probes, in force for Phase A); `docs/design/accepted/0014` (reps); `docs/design/accepted/0015` (probe channel for Phase B).
- `docs/design/accepted/0009`, `0010`, `0011`, `0013`, and `docs/design/accepted/0007` (the mitigations wired into the Phase A reanalysis).
- `results/runs/pilot-2026-05-31/` (the pilot this rigor pass improves on), including `judge2/` (the already-collected second-judge data).

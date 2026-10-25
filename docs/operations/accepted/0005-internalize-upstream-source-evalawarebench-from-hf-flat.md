# 0005. Internalize the upstream slice into cv/ and source EvalAwareBench from the HF flat dataset

- **Status:** Accepted: executed; golden-fidelity invariant verified (dataset 360/360
  prompts byte-exact, SR/TP 720/720 decisions reproduced) and the submodule removed.
- **Date proposed:** 2026-06-04
- **Date accepted:** 2026-06-04
- **Supersedes:** relaxes PLAN.md hard-constraint #1 ("Addon, not edits")

## Context

The addon currently pins `decomposing-eval-awareness` as a submodule and imports its code unedited at runtime: `run/run_eval_bench.py` (rollout), `eval_bench/render.py` (factor rendering), `core/evaluator.py` (the four detectors), and `utils.py` (`create_target_client`, `query_parallel`, `load_responses`). `PLAN.md` hard-constraint #1 ("Addon, not edits ... We reuse upstream functions by importing them") makes this a load-bearing stance. The benchmark itself is read from the submodule's `eval_bench/tasks/*.yaml` via upstream's bespoke `render(template, factor_config)`.

Two things make this coupling increasingly expensive. First, the new deployment-negative arm (`docs/design/accepted/0010`, built in `cv/deployment.py`) has to thread WildChat prompts through the same upstream rollout/detector code, inheriting its idiosyncrasies (env-driven `run_eval_bench`, a shared `results/eval_bench/response` glob dir, the YAML+`render` path) for prompts that have nothing to do with EvalAwareBench's task YAMLs. Second, `aisa-group/EvalAwareBench` is published on HuggingFace as a flat table (one row per rendered `(task, valence, config)` prompt), which already contains what `render` reconstructs, so the YAML+`render` dependency buys us little.

## Decision (proposed, future)

Internalize the small slice of upstream we actually use into conventional, first-class `cv/` modules, and source the benchmark from the HF flat dataset:

1. **Data:** load `aisa-group/EvalAwareBench` (pin a dataset revision) and **unflatten** it into the `(task, valence, config_name, factor_config, prompt)` structure the pipeline needs, replacing the submodule-YAML + upstream `render` path. The prompts are taken as-published (no re-rendering).
2. **Code:** reimplement the slice we depend on (the target client + reasoning handling, `query_parallel`, the four detector core functions, the judge calls, and the response/judge wire schemas) as plain `cv/` modules in a conventional shape, so `cv/` no longer imports upstream at runtime.
3. **Provenance:** keep the pinned submodule as a reference and as the source of the golden-output fidelity check (below), not as a runtime dependency.

**Hard invariant (keeps this an operations change, not a methodology one):** the unflattened configs and the reimplemented rollout/detectors must reproduce the current outputs on a golden subset (the committed `results/runs/pilot-2026-05-31/` rows and the cue ground truth in `cv/factors.py`) to within documented tolerance, so the claims and scoring (`docs/design/accepted/0002`, `0003`) are provably unchanged. If the HF flat dataset does not carry `factor_config` (or enough to reconstruct it) and match the rendered prompts byte-for-byte against the submodule, this DDR does not proceed as written.

## Alternatives considered

- **Keep importing upstream unedited (status quo).** Lost: the coupling to upstream's layout and the YAML+`render` path is exactly what this escapes; the deployment arm makes it worse.
- **Fork and edit upstream in place.** Lost: still upstream-shaped and submodule-bound; does not give a conventional, testable `cv/` codebase.
- **HF data only, keep importing the detectors/rollout.** Considered as a first increment: drops the YAML+`render` dependency without the larger reimplementation; a reasonable staging point if the full internalization is too big to land at once.

## Consequences

### Enables
- A conventional, refactorable, testable `cv/` codebase decoupled from upstream's file layout and env-driven entrypoints.
- Removal of the submodule runtime dependency and the shared `results/eval_bench/response` glob dir that the deployment arm has to work around.

### Forbids / changes
- Relaxes `PLAN.md` hard-constraint #1 ("Addon, not edits"); that constraint and the Architecture section must be updated on acceptance, and `docs/design/accepted/0006` (which assumes the imported-upstream regime) revisited.

### Risks
- Silent divergence from upstream semantics if the fidelity check is weak. Mitigation: golden-output diff against the pilot; pin both the submodule commit and the HF dataset revision; keep the submodule for that diff.
- The HF flat dataset may not expose `factor_config` or may differ from the rendered YAML prompts. Mitigation: verify byte-for-byte on a sample before committing; otherwise stage (HF data first, or stay on YAML).

## References

- `PLAN.md` hard-constraints (#1 "Addon, not edits") and Architecture (the imported-upstream list).
- `docs/design/accepted/0002`, `0003` (the cue ground truth that the unflatten + reimplementation must preserve); `docs/design/accepted/0006` (the imported-upstream/API-only regime).
- `aisa-group/EvalAwareBench` (HF flat dataset) and the pinned submodule `eval_bench/tasks` + `eval_bench/render.py` (the source of truth to diff against).
- `cv/deployment.py`, `docs/design/accepted/0010` (the deployment arm whose extra coupling motivates this).

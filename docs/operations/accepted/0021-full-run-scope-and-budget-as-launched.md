# 0021. Full-run scope and budget as launched: Experiment B built, the pilot's Qwen rollouts reused, Gemma regenerated at fp8, the white-box pass on both models, $274 worst case under a $300 cap, two Modal workspaces

- **Status:** Accepted
- **Date proposed:** 2026-08-30
- **Date accepted:** 2026-09-03
- **Amends:** `operations 0019` (§4 line items and totals, §5 funding figures, §6 spend control) and
  `operations 0020` (the single-workspace claim under failure 2). Relates-to `design 0026`
  (scope), `design 0029` (Experiment B), `design 0031` (the white-box pass on both models),
  `operations 0013` (fp8 serving parity), `operations 0003` (per-run results namespacing).

## Context

The launch audit of 2026-08-30 (`docs/full-run-audit-findings.md`) read the code, the configs and
the accepted records against each other before the full run spends. The scope facts it established:
`config/full.yaml` and the code produced 10,800 rollouts while `operations 0019` budgets 25,200,
because Experiment B (the factor stacking of `design 0029` D2) had no sampler. The pilot's Qwen
rollouts were generated on the same serving stack the full run uses, with zero bad rows, while the
pilot's Gemma rollouts were served at an unrecorded precision.

The money facts it established: the two Modal token pairs resolve to two workspaces
(the two authors' workspaces), each with its own secret store, deployed app and ledger.
Modal's billing API shows the pilot burned $158.25 in the first and $30.13 in the second, $188.38
in all, where `operations 0020` states one workspace and $158. Every fp8 Gemma endpoint on
OpenRouter costs more than the price table held (read live on 2026-09-03: DeepInfra 0.13/0.38,
SiliconFlow 0.13/0.40, Parasail 0.15/0.40 per 1M tokens, and a 131k-context DeepInfra fp8 endpoint
at 0.27/0.76), and the judge carried no route constraint across endpoints up to 3.4x the metered
rate. The in-run cap is per run id: each run id has its own ledger and its own `budget_usd`.

The white-box pass was then measured live on 2026-09-03 (`scripts/bench_reencode.py`, 16 pilot
rollouts, zero NaN): capture at 2,439 tokens per second, the logit monitor at 3,834. A full pass is
7.7 H100-hours with overhead, $30 per model, against the $17 that `operations 0019` line 4
estimated for both models.

## Decision

1. **Scope.** Experiment B is built and on (`cv/stacking.py`, seeded, F8 never stacked): 12,600
   rollouts per model, 25,200 in all, and the scope assertion refuses to spend when the enumerated
   count differs from the budgeted one. Qwen's Experiment A reuses the pilot's rollouts. Gemma's
   Experiment A is regenerated at fp8 through the API's fp8 endpoints, so precision is uniform across
   targets. The deployment arm stays retired (`design 0026` D4) and its machinery and corpus are
   deleted. The white-box pass runs on both models (`design 0031`).
2. **The price table is the route's ceiling.** `config.PRICES_PER_M` holds, per model, the most
   expensive endpoint the run's routing may reach, and every OpenRouter request carries the same
   numbers as `provider.max_price`. An endpoint above the table is never served, so the metered
   rate bounds the billed one. Gemma is (0.15, 0.40), the judge (0.435, 0.87), Qwen on the API,
   which only `cv-validate` calls, (0.60, 3.60).
3. **The budget.** Line 2 is $34, line 4 is $61 (both models, measured, Gemma at the Qwen rate until
   its smoke), the worst case is $274 and the expected $263 (`scripts/full_run_budget.py`). The
   spillover of the plan as costed is accepted; the experiment does not change to fit the old
   number. `budget_usd` is $300 in `config/full.yaml`. The grant asks the worst case, $274.
4. **The caps.** The in-run cap binds one run id; a second run id starts from its own empty ledger,
   which is the documented behaviour, not a defect. The dashboard cap is per workspace, set by each
   owner: the first workspace at $130 (Qwen generation, the Qwen re-encode, infra and cold starts,
   about $115 worst case), the second at $40 or $50 (the Gemma re-encode and the organism anchor,
   about $42 worst case). Credits in either workspace are not netted until confirmed
   (`operations 0020`).
5. **Two workspaces are shared credit by design.** They are two independent pools with a manual
   switch of the token pair, not a failover. When a workspace refuses work at its cap, the Modal
   backend records each failed batch in the failure ledger and stops the stage after
   `MAX_CONSECUTIVE_BATCH_FAILURES` consecutive failures, so the run stops loudly and resumes on the
   other workspace from its checkpoints.

## Alternatives considered

- **Leave Experiment B out and run 10,800 rollouts.** Lost: the dose-response axis is a claim of the
  study (`design 0026`, `design 0029`), and the generation campaign covers both experiments at once.
- **Regenerate Qwen's Experiment A.** Lost: the pilot's Qwen rollouts come from the same build and
  precision the full run uses, so a regeneration buys nothing for about $20.
- **Keep the old $250 cap and descope.** Lost: the descope ladder of `operations 0019` §5 trades
  scientific scope for $24; the plan stands and the spillover is accepted (the authors decided, 2026-09-03).
- **Price the table at the cheapest fp8 endpoint.** Lost: OpenRouter's fallback can route to a
  costlier endpoint mid-run, and the ledger would then under-price real spend.
- **Pin each call to one provider.** Lost: a single provider's rate limit would stall the run; the
  price ceiling keeps the fallback among the endpoints the table already prices.

## Consequences

### Enables

- One generation campaign for both experiments, with the scope, the seed and the draws recorded in
  every responses file.
- A ledger that is an upper bound on the bill for every API line.
- A stated split of the Modal spend between the two workspaces, each owner's cap sized to it.

### Forbids

- Netting Modal credits before they are confirmed on the dashboard.
- Reading the in-run cap as a per-project cap.

### Risks

- The Gemma re-encode is priced at the Qwen rate; its own smoke at the prefull gate fixes the number.
- A provider's price change between now and the run moves the API lines. The rate check at run time
  (`operations 0019` §4) covers it.

## References

- `config/full.yaml` (`budget_usd`, `stack_*`, `whitebox_models`, `hf_revisions`),
  `src/cv/config.py` (`PRICES_PER_M`, `provider_max_price`), `src/cv/client.py`, `src/cv/judge.py`
  (`_route_bound`), `src/cv/stacking.py`, `src/cv/serving/local/backend.py`
  (`MAX_CONSECUTIVE_BATCH_FAILURES`), `scripts/full_run_budget.py`, `scripts/bench_reencode.py`.
- `docs/full-run-launch-readiness.md` (the decisions of 2026-08-30 and 2026-09-03, waves 1 and 2),
  `docs/full-run-audit-findings.md` (H12, H13, B6, B8, M11), `docs/grant-proposal.md`.
- `operations 0019`, `operations 0020`, `operations 0013`, `operations 0003`, `design 0026`,
  `design 0029`, `design 0031`.
- OpenRouter endpoint rates: `openrouter.ai/api/v1/models/<id>/endpoints`, read 2026-09-03.

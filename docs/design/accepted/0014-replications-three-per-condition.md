# 0014. Three replications per condition; spend the precision budget on tasks, not reps

- **Status:** Accepted (amended by `design 0026` for scope: the kappa-denoising sub-rationale is
  withdrawn. The implemented rep-level kappa estimator is invariant to K, so it never realized the
  denoising; reps=3 stands on stochastic-replication precedent and on being the floor at which the
  K>=2 stably-aware estimand is definable.)
- **Date proposed:** 2026-06-04
- **Date accepted:** 2026-06-04
- **Supersedes:** n/a (closes the open reps knob in `docs/operations/accepted/0002`; corrects a misattribution in `PLAN.md` and `config/full.yaml`)

## Context

The replication count was left open: `docs/operations/accepted/0002` says "reps=5 vs reps=10, decided after pilot spend is known" and `PLAN.md:428` lists it as a still-open item, while `config/full.yaml:11` ships `samples_per_config: 10` and `config/full.yaml:1` plus `PLAN.md:323` call 10 reps "paper-faithful". An alternative, considered and rejected, was the opposite extreme: lock greedy decoding and treat the benchmark's factor variations as the replications, i.e. run one sample per cell. Both the grant-facing pilot and the funded full run need the number settled now, and the pilot data plus the upstream paper make it determinable.

The facts the decision rests on:

1. **We run sampling, not greedy, over an API we do not control.** Targets run at `temperature=0.3, do_sample=True, top_p=0.95` (`upstream/core/target.py:30,66`; `cv/config.py` `TARGET_MODELS` sets 0.3 for all three targets, no override) via OpenRouter (`docs/design/accepted/0006`). We control neither provider, tensor-parallel size, nor batch composition, so the hardware-determinism precondition behind "T=0 makes re-runs identical" is unattainable here, and we never run greedy regardless.
2. **"10 reps = paper-faithful" is a misattribution.** In Li et al., EvalAwareBench (arXiv:2605.23055), the 10-samples-per-prompt figure is the *existing-benchmark* study (App. B.1: 9 models x 4 prior benchmarks). The authors' own factor-controlled benchmark, which is structurally our design (100 tasks x 2 valences x 9 single-factor configs), uses **3 repetitions** (Section 5.2). The upstream runner default is `SAMPLES_PER_CONFIG=3` (`run/run_eval_bench.py`).
3. **CI width is task-driven, not rep-driven.** The only confidence interval in `cv/analysis.py` is a cluster bootstrap over `task_name` (`_bootstrap_dprime_ci`, lines 96-110; `CI_MIN_CLUSTERS=20`). Reps enter only as pooled rows in the rate denominators and the kappa sample size (lines 83-87, 165-172); they add no clusters, so they barely move the percentile interval. The pilot's headline d-prime CIs are flagged `ci_unstable` purely because there are 10 tasks, not because of the rep count.
4. **Reps still carry real, irreplaceable information.** Measured rep-to-rep flip rate at T=0.3 (re-derived from `results/runs/pilot-2026-05-31/detectors/*.jsonl`): CoT 1.7%, PG 5.6%, SR 10.0%, TP 12.8%; implied task-ICC 0.42 to 0.80. One re-roll flips roughly one in eight TP cells, and `K=1` cannot compute the "stably-aware >= k/K" derived factor (`PLAN.md:122`) at all, nor denoise the per-cell label, nor recover the SR-TP kappa that single-rep label noise attenuates (~0.55 toward ~0.30-0.40).
5. **All three reference papers replicate stochastically; none argues T=0 removes the need.** EvalAwareBench runs 10 (existing-bench) or 3 (its own bench) samples at T=0.3 and calls single-run labels "spurious" (Section 3.2). Needham et al. (arXiv:2505.23836) run 5 samples x 5 prompt phrasings at T=1 as explicit variance reduction. Vishwarupe et al. (arXiv:2605.11496) define the measured property as an expectation over the response distribution and require task- and context-level bootstrap CIs.

## Decision

Set replications per `(model, task, valence, config)` cell to **3** for both the grant-facing pilot's fresh rollouts and the funded full run. Three is the EvalAwareBench-faithful number for our design (their Section 5.2), is the floor at which the ">= 2/3 stably-aware" derived factor is definable, and captures roughly 60 to 65% of the achievable within-cell denoising; reps 6 to 10 buy under 10% additional CI tightening on the only detector with signal (TP), because the between-task variance floor (ICC 0.56 to 0.80) dominates and reps cannot touch it.

Treat the **number of tasks as the precision lever.** Keep enough task clusters to clear the `ci_unstable` gate (at least 20; the 2026-05-31 pilot tripped it at 10) and target the full 100 for the funded run. Marginal budget goes to tasks, and on the funded run to models and the probe channel (`docs/design/accepted/0015`), never to reps beyond 3. Permit **reps = 5 only if** the stably-aware stability statistic is later promoted to a headline estimand needing a finer threshold, recorded as its own decision at that time. Do **not** drop to reps = 1 or greedy: it is empirically non-deterministic on OpenRouter, the cited determinism precondition is unattainable, and `K=1` cannot compute our K-dependent estimand.

## Alternatives considered

- **reps = 10 (status quo `full.yaml`, "paper-faithful").** Lost: misattributed (that K is the existing-benchmark study; the factor-controlled bench uses 3); roughly 2 to 3x the cost for under 10% CI gain; trades away tasks and models, the binding constraints.
- **reps = 1, treat the factor configs as the replications.** Lost: configs are the signal axis with deliberately different true means, so pooling them biases the d-prime contrast (~0.18 vs a true ~0.42); they add no bootstrap clusters; and `K=1` cannot denoise the per-cell label, recover the attenuated SR-TP kappa, or define the stably-aware factor.
- **reps = 5 as the default.** Lost as the default (78% vs 63% of within-cell denoising for ~1.7x the cost); retained as a conditional ceiling if stability becomes a headline.
- **Leave it open (`operations/0002` status quo).** Lost: both the pilot and the funded run need the number now, and the variance structure makes it determinable from the pilot.

## Consequences

### Enables
- A defensible and cheaper design: reps 10 -> 3 frees roughly 70% of the all-API target+judge bill (about $517 at full scale), which funds the validity mitigations (`docs/design/accepted/0009-0013`) and more tasks or models.
- A well-defined ">= 2/3 stably-aware" derived factor and adequate per-cell denoising at the measured flip rates.

### Forbids
- Reporting reps = 10 as paper-faithful for this design.
- Pooling distinct-mean factor configs as exchangeable replications.
- Headlining a per-cell d-prime CI as if reps drove its width; the width is task-driven.

### Risks
- 3 reps gives a coarse 4-level stability read (0, 1/3, 2/3, 1). If a fine stability gradient becomes a headline, revisit toward 5.
- The ICC estimates come from a 10-task, 2-rep pilot that is itself `ci_unstable`, so the within/between split is uncertain. The qualitative ordering (tasks far outweigh reps for CI; reps essential for labels and kappa) is robust across ICC 0.5 to 0.85.

### Action items (applied on acceptance)
- Change `config/full.yaml` `samples_per_config` 10 -> 3; correct the "paper-faithful" note in `PLAN.md:323` and `config/full.yaml:1`.
- Implement the stably-aware ">= k/K" statistic in `cv/analysis.py` (today `sample_idx` is pure passthrough, so the current reps buy nothing statistical). Tracked as Phase A reanalysis work in `docs/operations/accepted/0004`.

## References

- `cv/analysis.py:96-110` (task-cluster bootstrap), `:83-87`, `:165-172` (reps pooled into rates and kappa N); `cv/config.py` `TARGET_MODELS`, `upstream/core/target.py:30,66` (T=0.3 sampling); `upstream/run/run_eval_bench.py` (`SAMPLES_PER_CONFIG` default 3); `config/full.yaml`, `PLAN.md:122,323,428`.
- `results/runs/pilot-2026-05-31/` (flip rates, ICC, `ci_unstable`).
- Li et al., "Decomposing and Measuring Evaluation Awareness," arXiv:2605.23055, Section 5.2 (3 reps), Appendix B.1 (10 reps, existing-bench), Section 3.2 (>=3/10 stably-aware).
- Needham et al., arXiv:2505.23836 (5x5 samples at T=1); Vishwarupe et al., arXiv:2605.11496 (expectation over response distribution; clustered bootstrap).
- `docs/operations/accepted/0002` (the open reps knob this closes); `docs/design/proposed/0013` (cue-positive cell count and power); `docs/design/accepted/0005` (kappa co-primary).

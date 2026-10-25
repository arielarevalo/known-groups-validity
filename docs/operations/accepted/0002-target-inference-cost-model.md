# 0002. Target inference cost model and serving regime

- **Status:** Accepted (full-run serving regime revised by `docs/operations/accepted/0004`: targets self-hosted for the funded Phase B; the cost model below stands)
- **Date proposed:** 2026-06-03
- **Date accepted:** 2026-06-03
- **Supersedes:** n/a

## Context

Cost is a hard constraint: the project runs on a $15 OpenRouter key with a $14
meter cap (`cv/meter.py`). The pilot (gemma, 360 rollouts) produced real
per-rollout usage in `results/_metrics.jsonl`, replacing the pre-pilot token
guesses. We need a settled cost model for the full run and a decision on the
target serving regime (API vs self-host), because the dominant line item (qwen
completion at $3.20/M) is large enough that self-hosting could roughly halve the
bill.

Measured per-rollout usage (360 gemma rollouts):

| Side | calls/rollout | in tok | out tok |
|---|---|---|---|
| target | 6 | 7,909 | 5,022 |
| judge (mimo, model-independent) | 2 | 4,684 | 3,662 |

Per-1M prices, verified live on OpenRouter: gemma $0.12/$0.37, mistral
$0.15/$0.60, qwen $0.29/$3.20, mimo $0.435/$0.87. Blended (target + judge) per
rollout: gemma $0.0080, mistral $0.0094, qwen $0.0236; the judge is a fixed
$0.0052/rollout regardless of target.

## Decision

Adopt the measured cost model. Run targets **all-API via OpenRouter** for the
full run by default, under the $14 meter cap. The settled full-run estimate is
**~$739 all-API at reps=10** (target $457 + judge $282; reps=5 ~$369).

Identify **self-hosting the three open-weight targets on interruptible GPUs, mimo
staying on the API** as the single biggest savings lever (~$322 at reps=10, a
~$416 saving), but do **not** adopt it here. Self-hosting changes the serving
regime that `docs/design/accepted/0006-api-only-targets-defer-probes.md` fixes as
API-only, with a pilot-vs-full confound consequence, so it is **gated on an
amendment to DDR 0006**. Until then it is a documented option, not the plan.

## Alternatives considered

- **Pre-pilot assumed token counts** (target in ~21k / out ~7.5k; ~$0.018/rollout;
  reps=10 ~$960). Superseded: measured target-in is ~2.6x lower and judge-out ~3x
  higher, so blended is ~$0.0137/rollout and the old estimate was ~30% high.
- **Self-host targets, interruptible A100 80GB** (~$0.60/hr spot, full bf16, ~92
  GPU-hr across 3 models at reps=10 -> ~$40 GPU after free credits). Best
  economics; spot is safe because the pipeline is resume-safe (`completed_ids`).
  Lost for now: gated on DDR 0006.
- **Self-host targets, on-demand A100 80GB** (~$2.99/hr -> ~$276 GPU). Removes
  preemption risk but saves less (~$181 vs all-API).
- **reps=5 vs reps=10.** reps=5 halves it (~$369 all-API / ~$154 self-host);
  decided after pilot spend is known.
- **Do nothing beyond all-API.** Status quo, simplest; the adopted default.

## Consequences

### Enables
- A settled full-run budget of ~$322-$739 depending on serving choice; the $14
  meter cap makes overrun structurally impossible regardless.
- A clear, costed path to roughly halve the bill (self-host targets) if DDR 0006
  is amended.

### Forbids / constrains
- No self-hosting of targets until DDR 0006 is amended.
- Serving-regime hygiene: do not pool API-pilot and self-hosted target rollouts
  into one analysis as if the serving stack were identical.

### Risks
- reasoning-effort=high can multiply output tokens severalfold (the dominant cost
  driver, worst on qwen). The $14 cap bounds it; self-hosting would convert a
  blowup from $3.20/M into flat GPU-hours.
- The self-host GPU-hours hinge on vLLM decode throughput (central 1,000 tok/s,
  band 700-1,500), not yet measured; a one-hour A100 probe before the full run
  would tighten it.

## References

- `results/_metrics.jsonl` (measured usage); `cv/meter.py` (meter + cap);
  `cv/config.py` `PRICES_PER_M`.
- `docs/operations/accepted/0001-store-results-and-observability-in-wandb-weave.md`:
  the storage half of "cost and storage" in PLAN.md.
- `docs/design/accepted/0006-api-only-targets-defer-probes.md`: the serving-regime
  decision self-hosting would amend.
- Lightning AI pricing (A100 80GB $2.99 on-demand / ~$0.60 spot), verified
  2026-06-03.

# 0020. Pilot spend post-mortem: $158 billed; the credit-covered ~$60 estimate failed twice

- **Status:** Accepted. **Amended by `operations 0021` (2026-09-03):** the single-workspace claim
  under failure 2 is wrong. The two token pairs resolve to two workspaces, and Modal's billing API
  shows $158.25 in the first and $30.13 in the second, $188.38 for the pilot.
- **Date proposed:** 2026-07-04
- **Date accepted:** 2026-07-04
- **Amends:** the pilot-spend accounting in `operations/accepted/0019` §5 and `docs/STATUS.md`
  (previously "~$180 spent, ~$60 of free credits absorbed, ~$120 charged"). The overrun
  *decomposition* in 0019 §5 (cold-start tax, Gemma-on-GPU, no spend cap) stands; the dollar
  figures and the credit narrative are corrected here.

## Context

The pilot's GPU cost was planned in `operations/accepted/0007` (2026-06-04) as ~4–9 GPU-hr,
sized to "fit the free credits", with the credit pool computed as $30/mo per person across two
Modal keys ($60/mo). The retrospective clean token model priced the pilot at ~$60 of GPU. The
expectation on record was therefore: roughly $60 of usage, roughly $60 of credits, about $0 out
of pocket.

The invoices say otherwise. The June cycle billed **$158 to the card**, in two charges ($60 and
$98; the first observed 2026-06-23, the second at cycle close). **No free-credit coverage
materialized.** `docs/STATUS.md` and 0019 §5 had recorded an intermediate reading ("~$180 spent,
~$60 credits, ~$120 charged") taken from the dashboard on 2026-06-23, before the cycle closed;
the invoice totals replace it.

The estimate failed twice, independently:

1. **Usage: ~2.6× the clean token model ($158 vs ~$60).** The model priced the experiment;
   the bill was dominated by everything around the experiment. The decomposition in 0019 §5
   still holds: (a) the cold-start tax (~65-min Volume weight loads, $4.3/load at H100 rates,
   multiplied across smoke/OOM/resume/failover container starts); (b) Gemma served on GPU
   (~$40) instead of per-token API; (c) no workspace spend cap, so nothing arrested the drift.
   The production subset runs themselves (2026-06-12/13, ~8 GPU-hr) landed inside the 0007
   estimate; the un-modeled meta-work (debugging, smokes, benchmarking, re-runs, cold loads)
   cost more than the experiment.
2. **Credits: assumed $60, applied $0.** The "×2" pooling never existed operationally: Modal
   free credits attach to a workspace, and all GPU work ran in a single workspace of
   one of the two authors (verified 2026-07-04: the token sees exactly one workspace). Even that workspace's
   own monthly credit shows no offset on the invoices; whether it was consumed by early-June
   smokes, ineligible, or misapplied must be read off the billing dashboard, and is worth a
   Modal support ticket if it is genuinely absent.

Either failure alone produces a surprise bill; together they moved out-of-pocket from ~$0 to $158.

## Decision

Correct the accounting on every surface that carries it, and adopt three estimation rules for
all future compute budgeting:

1. **Free credits count as $0 until an invoice shows them applied.** No plan may net expected
   credits against expected usage. Credits, if they appear, are upside, never budget.
2. **Reconcile within a day.** After any GPU session, the billing dashboard
   (modal.com/settings/usage) is read against the run meter (`cv.meter.record_gpu`) within 24
   hours, so a divergence is caught at the ~$10 scale, not at cycle close.
3. **Estimates must state their meta-work exposure.** A compute estimate prices either (a)
   production plus an explicit, evidence-based iteration/validation line, or (b) production
   only, with the iteration spend bounded by a hard cap and a descope ladder rather than
   estimated. Silence about which of the two it is, the pilot's actual failure mode, is no
   longer acceptable. The full-run budget (`operations 0019`) is of type (b): $226 prices
   production at worst-case measured rates; iteration is bounded by the workspace cap and
   `budget_usd`, and absorbed, if needed, by the descope ladder.

Surfaces corrected in this change: `operations/accepted/0019` §5 (figures and credit
narrative), `docs/STATUS.md` (budget-cluster bullet and the spend-cap runbook line),
`docs/OVERVIEW.md` (the pilot tier is no longer described as "free-credit scale"), and
`docs/grant-proposal.md` (the proposal's statement that the pilot fit within free credits was
corrected to the invoiced figure, $158).

## Alternatives considered

- **Status quo (leave the ~$180/$60/$120 reading in place).** Rejected: the main budget record
  would contradict the invoices, and the grant proposal would disagree with the invoiced figure.
- **Treat this purely as a Modal billing dispute.** Rejected as the primary framing: even with
  a $60 credit restored, usage was ~2.6× the model, so the estimation lesson stands. The
  support ticket is kept as an action, not a substitute.
- **Patch the numbers without a record.** Rejected: the two failure modes (credits assumed
  before seen; meta-work un-modeled) are exactly the checks future budgets need to cite.

## Consequences

### Enables
- The funding narrative matches the invoices.
- Future estimates carry an explicit credit policy (rule 1) and meta-work statement (rule 3),
  so a reviewer can see what is priced and what is capped.

### Forbids / changes
- No budget line may assume unconfirmed credits, discounts, or allowances of any provider.
- The pre-GPU-run prerequisites of 0019 §6 (workspace spend cap set in the dashboard; weights
  on local SSD or a baked image, not the 9P Volume) gain a third: rule 2's next-day
  reconciliation.

### Risks
- **The $226 grant ask prices production, not iteration.** Its line items do not inherit the
  pilot's errors: no credits are netted, decode rates are measured, Gemma is per-token API,
  and cold starts are an explicit worst-case line ($17). But the category that dominated the
  pilot's bill, iteration and meta-work, appears only as ~$8 of validation. The protections
  are structural (the $250 hard cap and the descope ladder), not budgetary: if full-run
  iteration mirrors the pilot's ratio, the run descopes rather than overspends. This is a
  deliberate no-padding choice and is now stated rather than implicit.

## References

- `operations/accepted/0007` (the original ~4–9 GPU-hr, credits-cover-it estimate).
- `operations/accepted/0019` §5–§6 (budget, overrun decomposition, spend controls; figures
  corrected by this record).
- `docs/STATUS.md`, `docs/OVERVIEW.md`, `docs/grant-proposal.md` (corrected surfaces).
- Modal invoices, June 2026 cycle: two charges, $60 + $98 = $158 (card statements).
- modal.com/pricing (H100 $3.95/hr on-demand; Volume storage 1 TiB/mo free, so storage
  contributed $0).
- `cv/meter.py` (`record_gpu`), `config/full.yaml` (`budget_usd`), the 0019 §6 cap mechanism.

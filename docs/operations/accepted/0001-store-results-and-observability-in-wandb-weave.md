# 0001. Store results and run-observability in W&B Weave

- **Status:** Accepted
- **Date proposed:** 2026-06-03
- **Date accepted:** 2026-06-03
- **Supersedes:** n/a

## Context

The study produces two kinds of output: a small tidy table of detector decisions
(`results/merged.jsonl`/`.csv`, ~511 B/row, ~13 MB Parquet at the full reps=10
scale) and a large body of raw reasoning traces (~650 MB text at full scale). Two
people need a cloud home for these that supports both exploration of arbitrary
cohorts and reading the underlying trace text, not a file shuttled to and from
disk. The collaboration should not require standing up infrastructure, and it
must stay within a free or near-free budget.

Every model call already routes through the `openai` SDK; `cv/meter.py` installs
its budget meter by monkeypatching that SDK's `Completions.create`. An
experiment tracker that auto-instruments the same SDK therefore captures the
whole experiment with near-zero new code.

A hard requirement emerged during evaluation: **no retention cliff**. ~650 MB
must persist indefinitely at no cost, which eliminates every managed
eval-platform free tier whose business model is short retention.

## Decision

Use **W&B Weave** as the cloud results store and run-observability layer. One
`weave.init("<entity>/<project>")` at the pipeline entrypoint (where `cv/meter.py`
installs its patch) auto-logs the OpenAI-SDK calls; detector/rollout functions
get `@weave.op()` so cohorts are queryable and traces readable. Collaboration
uses the default entity created at signup (no explicit team-creation step) plus
a second invited seat (free tier allows up to 5). Curated views are persisted as
**W&B Reports** (shareable links).

The in-repo flat files (`results/merged.jsonl`, `results/_metrics.jsonl`, and the
raw rollout responses) remain the **durable system of record**; Weave is the
exploration/observability surface, not the canonical copy. File an academic /
research license application in parallel (TAIS/BlueDot research likely qualifies):
if granted it provides 200 GB, up to 100 seats, and configurable retention, which
removes the one open risk below.

The reasoning trail for this choice is documented as alternatives-considered
below rather than in a borrowed cross-field artifact (lab notebook, DMP,
compendium): an experiment tracker is the AI-native home for the run record.

## Alternatives considered

- **Braintrust (free)**: purpose-built eval platform; lost on a **14-day**
  retention cliff and a 10k-scores/month cap far below the run's volume.
- **Langfuse Cloud (Hobby)**: tracing + evals; lost on a **30-day** retention
  cliff and a 50k-units/month cap the full run exceeds ~10x.
- **Arize Phoenix Cloud (AX Free)**: lost on a **15-day** retention cliff and a
  25k-spans/month cap.
- **Databricks Free Edition / managed MLflow**: permanent free tier, no cliff;
  lost on **non-commercial-use-only** terms and a weak shareable-artifact story.
- **R2 (own the data) + Hex**: neutral object storage (free forever, no cliff)
  plus a hosted notebook UI; retained as the **fallback** if Weave's retention or
  terms disqualify it. Lost on more wiring (export step + two services) and not
  being a one-line in-notebook integration.
- **MotherDuck**, **Lightning Drive + served UI**: viable owned/queryable stores;
  lost on compute caps / platform coupling and more setup than the tracker path.
- **Self-host Phoenix or Langfuse (OSS)**: infinite retention, free of license;
  lost on the operational burden of running a 24/7 container.
- **Do nothing (keep only in-repo JSONL)**: durable but not cloud-explorable or
  shareable; retained underneath Weave as the system of record, not as the UI.

## Consequences

### Enables
- Cloud exploration of arbitrary cohorts and reading of trace text, shareable to
  the collaborator via a link.
- Curated views persisted as W&B Reports (the "here are the views that matter"
  artifact).
- Near-zero instrumentation: one `weave.init()` plus `@weave.op()` decorators,
  riding the OpenAI-SDK path `cv/meter.py` already patches.

### Forbids / constrains
- Weave must **not** become the sole copy. The in-repo flat files stay the
  system of record; anything the analysis reads must be reproducible from them.
- No claim or scoring logic may depend on Weave; it is observability only (keeps
  this decision on the operations side of the bright line).

### Risks
- **Free-tier retention is undocumented.** W&B's model is storage-capped (5 GB),
  not time-capped, so there is no *known* cliff, but it is unconfirmed.
  Mitigations: the academic-license application (configurable retention), and the
  durable in-repo JSONL fallback which makes any Weave data loss non-fatal.
- Free tier is 5 GB storage / 1 GB-month Weave ingestion; the run fits, but a
  re-ingest or trace-volume blowup could exceed it. Mitigation: sample traces if
  needed; JSONL remains complete.

## References

- `cv/meter.py`: OpenAI-SDK patch point where `weave.init()` will sit.
- `cv/merge.py`, `results/merged.jsonl` / `.csv`, `results/_metrics.jsonl`: the
  durable system of record this decision sits on top of.
- `DATACARD.md`: documents the dataset whose storage this record governs.
- `docs/design/accepted/0006-api-only-targets-defer-probes.md`: unrelated to Weave,
  but the **self-hosting cost path** (a separate, still-unwritten operations
  record) is gated on amending it; noted so the two are not conflated.
- Pricing/retention checked 2026-06-03: Braintrust plans-and-limits, Langfuse
  data-retention, Arize Phoenix self-hosting, W&B pricing, Databricks Free
  Edition limitations.

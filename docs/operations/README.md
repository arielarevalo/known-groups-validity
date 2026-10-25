# Operational Decision Records

This directory holds **operational decision records**: one short markdown file
per non-trivial decision about how the study is *executed, paid for, stored,
observed, or tooled*. They use the **same Markdown Any Decision Record (MADR)
format** as `design/` (see `design/README.md` for the shared mechanics), scoped
to operations rather than experiment design.

It is the sibling of `design/`. `design/` records decisions about the study's
*design, methodology, or scoring*; `operations/` records decisions about its
*machinery*. Same format, same lifecycle, separate concern.

## How to read these records

Each record states what was planned on its date. The design changed during the
work, so a record can describe a plan the study did not carry out. What the study
did is what the run data in W&B (curated in `reports/full/`) and the paper show.
The paper is "Known-groups validity is not enough for evaluation-awareness
detectors" (Agent Evaluation Science symposium, AES Fall 2026). Where a record
differs from the run data or the paper, those are correct. Records
cite internal working documents that are not in this repository. They also cite
code paths and line numbers as of their date; the code lives under `src/cv/`.

## The bright line (which log does a decision go in?)

- **`design/` (DDR)** if it changes **what the study claims or how results are
  scored**: the research question, the factor/detector design, the cue ground
  truth, the analysis outputs, the judge-validation requirement.
- **`operations/` (this log)** if it only changes **how we run, pay for, store,
  observe, or tool** the study, with **no bearing on the claims or scoring**:
  the results store, the observability stack, the budget mechanism, the compute
  backend's cost model, the CI/runtime tooling.

When a decision touches **both**, it needs a `design/` DDR, and the operations
record (if any) covers only the execution mechanics. Example: self-hosting the
target models is mostly operational, but it changes the serving regime that
`docs/design/accepted/0006-api-only-targets-defer-probes.md` fixes as API-only, and
that change has a confound consequence for the pilot-vs-full comparison. So it is
**gated on a DDR amendment to 0006**, not a free operational swap.

## Directory layout, numbering, lifecycle

Identical to `design/`:

- `accepted/`: records in force, or retained for the audit trail. `PLAN.md`,
  `README.md`, `src/cv/`, and configs should be consistent with every record here
  still in force.
- `proposed/`: records under consideration; no code or plan has changed yet.
- `template.md`: copy when starting a new record.

Numbering is **local to this log**: `NNNN-kebab-title.md`, zero-padded,
monotonic, next-unused integer across `operations/{accepted,proposed}`. `design/`
keeps its own independent sequence; an operations `0001` and a design `0001` are
different records in different logs.

The **log-vs-view model, status vocabulary** (Proposed / Accepted / Superseded /
Deprecated / Withdrawn), **relationships** (Supersedes / Amends / Refines /
Extends / Relates-to), and the **mechanics invariants** (Status line is truth;
birth moves, death stays; cite by stable ID) follow `design/README.md` exactly.

## Cross-references

- `PLAN.md` and `README.md` should cite the relevant operations record by number
  when describing any non-obvious operational choice.
- An operations record that depends on or is constrained by a design decision
  cites the `design/` DDR by number (and vice versa).

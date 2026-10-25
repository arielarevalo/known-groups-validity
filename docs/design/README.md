# Design Decision Records

This directory holds **Design Decision Records (DDRs)**: one short markdown
file per non-trivial decision about the study's design, methodology, or scoring.
The format is modelled on Michael Nygard's **Architecture Decision Records**
(ADRs), repurposed for experimental and analytical decisions rather than
software architecture.

Direct analogs in the scientific tradition are clinical-trial **Protocol
Amendments** and OSF-style **pre-registration amendments**. DDRs play the same
role: a versioned, dated, discoverable record of *why* the current design looks
the way it does and what it explicitly rules out. Here "pre-registered" means
fixed in a dated record before the data existed. No record is filed with an
external registry.

## How to read these records

Each record states what was planned on its date. The design changed during the
work, so a record can describe a plan the study did not carry out. What the study
did is what the run data in W&B (curated in `reports/full/`) and the paper show.
The paper is "Known-groups validity is not enough for evaluation-awareness
detectors" (Agent Evaluation Science symposium, AES Fall 2026). Where a record
differs from the run data or the paper, those are correct. Records
cite internal working documents that are not in this repository. They also cite
code paths and line numbers as of their date; the code lives under `src/cv/`.

## Why we keep them

The pilot produced several design changes (drop meta-d', drop the
behavioural-deviation criterion, score against EvalAwareBench config ground
truth, partition F1..F8 into pure-framing vs task-altering, and so on). A single
summary paragraph is the wrong unit for them: it conflates many decisions, makes
"supersedes" relationships invisible, and gives reviewers no place to push back
on a specific decision without arguing with the entire design.

DDRs give each decision its own file, each citing the context that forced it and
the consequences that follow.

## The log-vs-view model (read this first)

The DDR set is **not** the current specification, and should not be read as one.
It is an **append-only decision log**: each record is "what we decided, why, on
this date," frozen at that date. The current design is a separate, *materialized
view* over that log, kept in `PLAN.md` (and, for first-time
readers, the plain-language `docs/OVERVIEW.md`).

This is what makes backtracking tractable. You never mutate a record to reflect
a changed mind; you **append** a new record and **re-derive the view**. The log's
job is to never lie about what was believed when; the view's job is to say what
is true now. The only consistency maintained by hand: **any record that changes
the in-force state updates `PLAN.md` (and `docs/OVERVIEW.md` where
it describes the changed part) in the same commit.**

## Directory layout

- `accepted/`: records whose decision is in force, or *was* in force and is
  retained for the audit trail. `PLAN.md`, `src/cv/`, and configs should be
  consistent with every `accepted/` record still in force (Status `Accepted`,
  including `Accepted (amended by NNNN …)`). A superseded or deprecated record
  **stays here**, status-annotated; it is never moved or deleted.
- `proposed/`: records under consideration; no code or plan has changed yet. A
  proposed record that dies (`Rejected` or `Withdrawn`) also **stays here** when
  it carries reusable findings, status-annotated; it is deleted only when nothing
  of value would be lost.
- `template.md`: empty skeleton. Copy when starting a new record.

## Status vocabulary

Exactly five statuses; use no others:

- **Proposed**: under consideration; not in force; no code/plan change.
- **Accepted**: in force. May carry an annotation `Accepted (amended by NNNN
  for scope X)` when a later record overturns part of it (see *Relationships*).
- **Superseded by NNNN**: was accepted; **fully** replaced by a later record.
- **Deprecated**: was accepted; no longer applies, and no single record
  replaces it. Used sparingly.
- **Withdrawn**: was proposed; overtaken before a decision was taken (the
  question changed). Distinct from `Rejected`, a decision against the proposal
  on its merits.

## Relationships (a separate axis from status)

Status answers "is this in force?"; relationships answer "how does it touch
other records?" A record's relationship line (the header `Supersedes:` field,
read broadly) may carry any of:

- **Supersedes NNNN**: fully replaces NNNN, which becomes `Superseded by` this.
- **Amends NNNN (scope X)**: overturns only *part* of NNNN. NNNN stays
  `Accepted`, annotated `(amended by this for scope X)`. Reserve supersession for
  full replacement; partial overrides are amendments.
- **Refines / Extends / Relates-to NNNN**: non-overriding links.

## Numbering

Records are numbered `NNNN-kebab-case-title.md`, zero-padded, monotonic, and
**never reused** even if a record is rejected, withdrawn, or deleted. Numbers
come from the next-unused integer across `accepted/` ∪ `proposed/`. The number
is just an identifier; it does not imply chronological dependency.

## Lifecycle

- A record starts in `proposed/` as `Proposed`.
- **Accept:** move the file to `accepted/`, flip Status to `Accepted`, fill the
  acceptance date. Acceptance is the *only* move (see *Mechanics invariants*).
- **Supersede (full):** the new record carries `Supersedes NNNN` and its Context
  cites the replaced record; NNNN's Status becomes `Superseded by NNNN` and it
  stays in place.
- **Amend (partial):** the amended record stays `Accepted`, annotated; the
  amending record carries `Amends NNNN (scope)`.
- **Reverse:** a supersession or amendment that *reverses* an earlier decision
  (we changed our mind, as opposed to the world changing) must state in its
  Context what was previously decided, what new information or change triggered
  the reversal, and why. The reversal trail is a feature, not an embarrassment.
- **Consolidate:** when one topic accretes a supersession chain (A ← B ← C) or a
  thicket of amendments such that no single record states the current decision,
  write a consolidating record that restates the current decision in full and
  supersedes the chain. Rare and deliberate: it trades fine-grained history for
  readability (the squash-vs-preserve tradeoff). The superseded records stay.
- **Reject / Withdraw a proposal:** `Rejected` = decided against on the merits;
  `Withdrawn` = overtaken before a decision. Either way **keep the file**
  (Status annotated, e.g. `Withdrawn: overtaken by NNNN`) when it carries
  reusable findings; delete only when nothing of value would be lost.

## Mechanics invariants

- **The Status line is the source of truth** for whether a decision is in force.
  The directory is the origin bucket, not the authority.
- **Birth moves; death does not.** `proposed → accepted` moves the file, so a
  listing of `accepted/` is an honest in-force set. A record that is superseded,
  deprecated, or withdrawn **stays where it is**, status-annotated, so inbound
  references and the audit trail survive.
- **Numbers are immutable IDs**, never reused.

## Discipline

Each record is one decision. If a single change forces three independent
decisions, write three records that cross-reference each other. The point is to
give future readers (including future us) a record that admits surgical
amendment, not a thread of compound-decision essays.

A record is a *record*, not a *proposal essay*. Keep the Context to the facts
that forced the decision; keep the Decision to one or two paragraphs; keep the
Consequences to what the decision enables, forbids, and risks.

## Cross-references

- **Cite records by stable ID (`design 0017`, `operations 0010`), not by a
  state-qualified path** (`docs/design/proposed/0017-…`). A record's directory
  can change on acceptance; its log + number cannot. When a record graduates,
  fix any legacy full-path references to it.
- `PLAN.md` should cite the relevant record by number when describing any design
  choice that was not obvious. `accepted/` is the canonical record.
- `src/cv/factors.py`, `DATACARD.md`, and any scoring code that implements a design
  decision should reference the record number in a top-of-file comment.

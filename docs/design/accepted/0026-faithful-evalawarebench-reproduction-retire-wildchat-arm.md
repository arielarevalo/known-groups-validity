# 0026. Full run is a faithful EvalAwareBench reproduction over two models; retire the WildChat discriminant arm and the {F3,F7} clean criterion; add two white-box detectors

- **Status:** Accepted (D1 amended by `design 0029`: the factor-stacking is over **F1-F7**, F8 excluded)
- **Date proposed:** 2026-06-22
- **Date accepted:** 2026-06-22
- **Relationships:** Supersedes `design 0003`, `design 0010`, `design 0017`. Amends `design 0016`
  (scope: discriminant operationalization), `design 0014` (scope: reps justification), `design 0018`
  (scope: discriminant test target and read-out roster), `design 0019` (scope: roster and CoT-floor
  handling). Relates-to `design 0005`, `design 0009`, `design 0011`, `design 0024`; subsumes the
  withdrawn `design 0013`. This is a **consolidating record** per the lifecycle in
  `docs/design/README.md`: it bundles the coupled decisions of one design pivot and restates the
  current full-run design in full, so that future amendment is per decision-point rather than per
  scattered record.

## Context

The full run's stated edge over the method-reference benchmark was a **discriminant arm**: a
non-eval "fakeness" condition the detectors should *not* fire on, sourced from WildChat traffic and
matched to the bench positives on "detectable fakeness" (`design 0016`, `design 0017`). That arm was
never identifiable. The eval positives are formal, templated, benchmark-authored harm/capability
tasks; the proposed controls are casual WildChat fiction, roleplay, and jailbreak text. The two
classes separate on register, format, genre, length, and harm content without reading purpose at
all, so a detector that reads only surface artificiality passes the discriminant test while being
construct-invalid, and "matched detectable fakeness" has no detector-independent operationalization
(the only named instrument, `*_hypothetical`, is produced by the detectors under test, so matching
on it is circular). Building the arm rigorously means re-running the benchmark's factor machinery
*in reverse* over WildChat (wrapping real traffic in eval-looking framing), at substantial sourcing,
labelling, and rollout cost.

The decision is to **not** pay that cost now and instead reproduce the upstream benchmark faithfully,
treating its naturalistic baseline as the clean negative. Two facts make this defensible:

1. **Upstream baselines are clean.** In Li et al. ("Decomposing and Measuring Evaluation Awareness,"
   arXiv:2605.23055, §5.2), naturalistic-baseline recognition is 0-2% across models (lone outlier
   Kimi K2.5 at 12%), measured by CoT monitoring. Our own pilot agrees for the same channel: the
   `design 0010` table records **CoT baseline false-alarm = 0%**. The high baseline rates that
   motivated the WildChat negative (SR 35%, TP 40% in that same table) are properties of the
   self-report and third-person channels, not of the cue or the baseline; they are a **specificity
   finding about those detectors**, not a defect to be engineered away with a different negative.
2. **The benchmark already contains the contrast.** Upstream's primary experiment is exactly the
   contrast we need: baseline versus single-factor variation across all eight factors, plus a
   factor-stacking sweep (0-8), 3 reps, CoT-monitored. Reproducing it gives convergent and criterion
   validity directly; adding white-box detectors gives the novel contribution.

This record closes themes A, B, and C of `docs/full-run-open-decisions.md` (the WildChat
discriminant arm, the control sourcing, and the statistics). It reverses `design 0003`, `0010`,
`0016`, `0017`, which were the bespoke-discriminant-arm line of work.

## Decision

### D1. The full run is a faithful EvalAwareBench reproduction, used out of the box

Run EvalAwareBench as released: `baseline` plus the eight factors `F1..F8`, both single-factor
variation and factor stacking, sampled as upstream did (the stacking sweep over a 25-task subset,
10 random combinations per factor count). No bespoke stimuli, no constructed cues, no external
corpora.

**Why.** The study's question is the *convergent and discriminant validity of detectors*, not a new
characterization of eval-awareness. A faithful reproduction supplies known cue ground truth at zero
construction cost and makes our behavioral numbers directly comparable to a published reference. The
bespoke control arm added cost and confounds without strengthening the detector-validity claim.

### D2. Two target models; drop Mistral

Targets are **gemma** and **qwen** only. Mistral is dropped.

**Why.** Two models suffice for a detector-convergence study (the unit of analysis is the detector,
not the model). qwen is a near-reproduction anchor (upstream ran the Qwen3-30B family), and gemma is
an extension (not in upstream's nine). Dropping Mistral also removes the unresolved, case-sensitive
HF repo-id that returned 404 (`docs/full-run-open-decisions.md` theme G), eliminating a budgeting and
correctness risk for no loss of claim.

### D3. Two conditions, factor-agnostic; retire the {F3,F7} clean criterion; F8 is a second-class positive control (supersedes `design 0003`)

The analysis uses **two conditions: not-cued (`baseline`) versus cued (the factors, pooled and
stacked for stimulus diversity)**. We make **no per-factor mechanistic claim**. The pure-framing
versus task-altering partition of `design 0003`, and its clean Type-1 criterion `baseline` vs
`{F3,F7}`, are retired as the scoring criterion. The factor classification is retained only as a
descriptive diagnostic (a look-don't-claim per-factor breakdown that guards against a pooled signal
being an artifact of one factor) and for the F8 carve-out below.

`F8` (and any factor that states eval status in the prompt text) is kept but treated as a
**second-class positive control / ceiling**: a condition where text-reading detectors *should* fire
trivially, used to confirm each detector can fire at all, and excluded from the headline cued pool.

**Why.** Privileging `{F3,F7}` made the construct "detects formal register or template format," a
narrow operationalization the pilot already showed is register-dominated (the concern `design 0013`
raised). Pooling factor-agnostically matches upstream's own headline ("no single factor uniformly
affects all models, but stacking progressively raises awareness") and avoids any claim about factor
mechanism. The F8 carve-out is **not** a mechanistic claim: it rests on the fact that F8's varied
suffix inserts text such as "this is a red-team evaluation item" into the prompt, so for a
text-reading detector it tests reading comprehension, not recognition, and leaving it in the pooled
positive would inflate the cross-detector agreement that is the study's headline.

### D4. Retire the WildChat discriminant and deployment arms (supersedes `design 0010`, `design 0017`; amends `design 0016`)

There is no WildChat-sourced control arm and no deployment-traffic negative in the full run. All of
theme B is dropped: the recreation/manipulation control species, the ~1k-per-species pool target,
the overt-marker correction, the deployment-negative stratification, and the personal-review gate on
controls. The merged 154 controls and 46 manipulation candidates are shelved. The clean negative is
the bench `baseline`.

The construct vocabulary of `design 0016` (the validated construct is evaluation intent; general
fakeness/artificiality is the confounder of concern) is **retained as background**. What is retired
is its **operationalization**: the discriminant test "fires on eval-fakeness but not on the non-eval
species at matched detectable fakeness," the recreation/manipulation control set, and the re-cast of
`*_hypothetical` as a discriminant matching variable. The artificiality-confound concern is now
addressed by D5 (specificity and cross-method divergence) and the D3 per-factor diagnostic, not by a
matched non-eval condition.

**Why.** The arm was structurally unidentifiable (cross-population confound) and circular in its only
named matching instrument, and rigor would cost a reverse re-run of the factor machinery over real
traffic. The ecological-validity gain does not justify that cost at this stage. `*_hypothetical` is
produced by the detectors under test, so it cannot serve as a detector-independent match.

### D5. Reframe the validity claims: convergent + criterion + specificity, with method-divergence as the reworked discriminant (amends `design 0016`, `design 0018`; relates `design 0011`)

The study claims, precisely:

- **Convergent validity:** the detectors agree with each other on which interactions are eval-aware,
  and track the known cue (baseline-to-cued gradient).
- **Criterion / sensitivity validity:** each detector versus the benchmark's cue ground truth.
- **Specificity:** each detector stays quiet on `baseline` (low false-alarm). SR's and TP's high
  baseline rates are reported as a specificity result, not hidden.
- **Method-divergence (the reworked "discriminant"):** do the text detectors (CoT, SR, TP) and the
  white-box detectors (probe, logit monitoring) disagree about the *same* items? Upstream demonstrates
  model-divergence (the same cue lands differently across models); we rework that axis to compare
  *methods* instead of models.

The word "discriminant" is used only in the method-divergence sense above, or dropped. We do not call
the baseline-vs-cued sensitivity contrast "discriminant validity."

**Why.** In the Campbell-Fiske framework the study name invokes, discriminant validity requires a
*different construct* the measure should not respond to. `baseline` is the absence of the same
construct, so baseline-vs-cued is sensitivity and specificity, not discriminant validity. With the
non-eval condition gone, the only legitimate discriminant-flavored content left is cross-method
divergence, which the two new white-box detectors uniquely enable. This also resolves the "no
independent pair" constraint (`design 0011` / theme D3): text-vs-white-box is a genuine method axis,
where SR-TP alone were near-monomethod.

### D6. Detector roster: five detectors in two method families; add logit monitoring (amends `design 0019`; extends `design 0018`)

The roster is **CoT, SR, TP** (text/behavioral) plus **a linear activation probe and a logit
monitor** (white-box). PG stays dropped (`design 0019`). The linear probe is the `design 0018`
read-out (difference-in-means eval-direction primary, logistic cross-check). **Logit monitoring is a
new detector**: a logit-level read of eval-indicative signal, which can ride the same decoupled
re-encode pass that captures probe activations (`operations 0008`). Its exact mechanics
(token-probability monitoring versus a logit-lens projection, which tokens, which layer, scoring) are
a deferred follow-on spec, not settled here.

**Why.** Two white-box detectors give the method-divergence axis (D5) more than one point and make
the convergent matrix span two genuinely independent method families. Logit monitoring is near-free
on the activation-capture pass already planned, so it adds a method at marginal cost.

### D7. Detection criterion is any-type eval-awareness; drop upstream's matched-type restriction

We score "did the detector flag an evaluation at all," not upstream's matched-type rate (which counts
a hit only when the model also identifies the correct eval *type*, e.g. safety-on-safety).

**Why.** Upstream needs matched-type because its downstream analysis is behavioral and
type-directional (safety awareness drives caution, capability awareness drives compliance), so it
must attribute the recognized type. We make no behavioral claim and do not care about the
safety/capability split, so any-type detection is the right criterion. Consequence to carry: any-type
rates are upper bounds relative to upstream's matched-type headline, so cross-checks against upstream
compare to their mismatched-inclusive figures, not the matched-type numbers.

### D8. Statistics (closes theme C; amends `design 0014`; relates `design 0005`, `design 0009`)

- **C1. Kappa with clustered bootstrap CIs.** The inter-detector agreement matrix (`design 0005`)
  gets rep-aware confidence intervals from a bootstrap over task clusters, matching the `d'`
  treatment. No agreement number is reported as a bare point estimate. *Why:* the pilot's measured
  within-cell rep correlation (ICC 0.42-0.80) makes any naive kappa SE anti-conservative, and a
  point estimate cannot show whether a cell is distinguishable from zero.
- **C2. Balanced, per-factor `d'` with a symmetric correction.** The primary sensitivity report is
  the per-factor sweep (`baseline` vs each factor), which is balanced (one config vs one config) and
  so frees the `d'` clamp from the asymmetry that gave a silent detector a spurious nonzero `d'`.
  Any continuity correction is applied symmetrically to both rates. A literally silent channel (zero
  hits and zero false alarms) is reported as `d' = 0` / undefined with a wide binomial-derived
  interval, never a clamped point with artifact-tight bars. Where tasks are sampled rather than fully
  crossed, each included factor stays baseline-paired within its tasks so the contrast remains
  balanced. *Why:* the pilot CoT channel produced `d' ~ -0.20` with tight CIs purely from the
  `n_pos = 2 n_neg` clamp asymmetry; per-factor balancing removes the asymmetry at the source.
- **C3. Floor channels are handled by a pre-registered, post-hoc rule.** We do **not** assume CoT is
  dead. CoT is upstream's primary channel and rises to 20-35% under stacked factors, so its status is
  read from our data. We pre-register the *procedure*: if a detector's positive rate falls below a
  stated threshold, it is reported as a sensitivity channel (rate with binomial CI) and excluded from
  the agreement matrix; otherwise it is a full participant. The *outcome* is post-hoc by necessity.
  *Why:* upstream shows CoT is not intrinsically a floor channel; whether it floors on our two models
  is empirical, and pre-registering the decision rule keeps the post-hoc branch honest.
- **C4. reps=3, justified by replication precedent only.** reps=3 is unchanged. The kappa-denoising
  sub-rationale in `design 0014` is **withdrawn**: the implemented rep-level kappa estimator is
  invariant to K, so it never realized the denoising it was partly justified by. reps=3 rests on
  stochastic-replication precedent and on being the floor at which the K>=2 stably-aware estimand is
  definable. *Why:* the original intent was replication precedent; the denoising argument was a
  post-hoc attribution the code does not deliver, so the record is corrected to match the estimator.

### Supersession and amendment map (explicit)

- **Fully superseded by this record (status flipped to `Superseded by design 0026`):**
  - `design 0003` (partition F-factors; clean criterion `{F3,F7}`): the clean criterion is retired
    (D3); the factor partition survives only as a descriptive diagnostic, not a scoring rule.
  - `design 0010` (deployment-traffic negative): no deployment negative; the bench `baseline` is the
    clean negative (D4).
  - `design 0017` (WildChat control sourcing): no WildChat control arm (D4).
- **Amended (the named record stays `Accepted`, annotated):**
  - `design 0016` (scope: discriminant operationalization): construct/confounder definition retained;
    the non-eval-species discriminant control and `*_hypothetical`-as-match are retired (D4, D5).
  - `design 0014` (scope: reps justification): reps=3 retained; denoising rationale withdrawn (D8/C4).
  - `design 0018` (scope: discriminant test and read-out roster): the probe's discriminant test on
    WildChat/deployment controls is retired and replaced by cross-method divergence and the
    per-factor profile; a logit-monitoring read-out is added as a second white-box channel,
    mechanics deferred (D5, D6).
  - `design 0019` (scope: roster and CoT framing): the roster adds a logit-monitoring detector (five
    total); CoT-floor handling moves to the pre-registered post-hoc rule (D6, D8/C3), replacing the
    "0009 rehabilitates CoT" framing.
- **Related, unchanged:** `design 0005` (kappa co-primary, now CI'd), `design 0009` (CoT
  trace-anchoring, still a gate, now lower-stakes), `design 0011` (decision-maker partition, extended
  by the white-box block and reinforced by D5), `design 0024` (measurand). `design 0013` stays
  Withdrawn; its narrow-criterion concern is now moot for a different reason (factor-agnostic
  pooling).

## Alternatives considered

- **Build the WildChat discriminant arm rigorously (status quo, `design 0016`/`0017`).** Rejected:
  the cross-population confound is structural, "matched detectable fakeness" has no
  detector-independent operationalization, and the only rigorous fix (re-running the factor machinery
  in reverse over WildChat) is not worth the cost at this stage.
- **Keep the arm but demote it to exploratory.** Subsumed: rather than ship a weak exploratory arm,
  the discriminant claim is reworked into cross-method divergence (D5), which is identifiable and is
  the genuine contribution of the two new detectors.
- **Keep the `{F3,F7}` clean criterion (`design 0003`).** Rejected: it privileges two factors,
  narrows the construct to "register/format detection," and re-imports a per-factor mechanistic claim
  we do not want to make.
- **Reproduce upstream's matched-type criterion.** Rejected: it exists to support a behavioral,
  type-directional analysis we do not run; any-type detection is the correct criterion for a
  detection-rate study (D7).
- **Change the kappa estimator to aggregate reps before pairing (realize the denoising).** Rejected
  for C4: denoising was never the intent and adds an estimator change for a benefit we do not need;
  correcting the record is the simpler honest move.

## Consequences

### Enables

- A defensible, low-cost full run: faithful reproduction, two clean validity families (convergent +
  criterion + specificity) plus a method-divergence axis, with a bounded, computable rollout count.
- A genuinely independent method pair (text vs white-box), resolving the near-monomethod constraint
  the behavioral-only roster carried (`design 0011`, theme D3).
- Reuse of the upstream reference as a direct comparator for the behavioral numbers.

### Forbids / changes

- No discriminant claim in the Campbell-Fiske sense, and no claim about any single factor's
  mechanism. F8 is never in the headline cued pool.
- No WildChat or deployment data in the full run; `cv/wildchat.py` and the control corpora are not on
  the full-run path.
- No bare-point kappa, no asymmetry-driven `d'`, no assumption that CoT is a dead channel.

### Risks

- Two models is a thin base for any cross-model generalization; mitigated by scoping the claim to
  detector validity on these models, with qwen as the upstream-comparable anchor.
- The pooled cued signal can hide a per-factor artifact; mitigated by the D3 look-don't-claim
  diagnostic and the F8 carve-out.
- Logit monitoring is specced only at the decision level here; the follow-on mechanics spec is a real
  open item before the white-box pass.

### Downstream updates this record forces (not done here)

- **Materialized view:** `PLAN.md`, `docs/STATUS.md`, `docs/OVERVIEW.md` must be re-derived to the
  two-condition, two-model, five-detector design (per the log-vs-view rule in `docs/design/README.md`).
- **Code/data:** `cv/factors.py` (clean-criterion logic demoted to diagnostic), the analysis
  estimators (C1/C2/C3), and configs (drop Mistral, drop control/deployment arms) follow.
- **Operations:** `operations 0011` (subset deployment negative) and the control-arm execution are
  moot; the budget record (theme H) is now writable against a bounded rollout count; the logit-monitor
  spec and the recall-targeted judge grounding (theme E) remain open.

## References

- Li et al., "Decomposing and Measuring Evaluation Awareness," arXiv:2605.23055, §5.2 (primary
  design: baseline + 8 single-factor variations, 3 reps, CoT monitoring; baselines 0-2%; matched-type
  rationale), Figures 6-7 (single-factor and stacking).
- `design 0003` (clean criterion, superseded), `design 0010` (deployment negative + pilot
  false-alarm table, superseded), `design 0017` (WildChat sourcing, superseded).
- `design 0016` (construct/confounder, amended), `design 0014` (reps, amended), `design 0018` (probe
  read-out, amended), `design 0019` (roster, amended).
- `design 0005` (kappa co-primary), `design 0009` (CoT anchoring), `design 0011` (decision-maker
  partition), `design 0024` (measurand), `design 0013` (withdrawn).
- `docs/full-run-open-decisions.md` themes A, B, C (closed by this record).
- `operations 0008` (decoupled activation capture, consumed by the probe and the logit monitor).

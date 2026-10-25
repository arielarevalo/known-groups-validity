# 0017. Source non-eval-fakeness controls from WildChat via search + prompt classification

- **Status:** Superseded by `design 0026` (the full run has no WildChat control arm; the discriminant
  test is reworked as cross-method divergence against the bench `baseline`).
- **Date proposed:** 2026-06-04
- **Date accepted:** 2026-06-07
- **Relationships:** Extends `design 0010` (deployment negative); Relates-to `design 0016` (the
  construct/confounder this sources controls for), `operations 0010` (controls run in the experiment,
  deferred out of the convergent-only subset)

## Context

DDR 0016 makes general fakeness the confounder and requires **non-eval fakeness control
conditions** (recreation, manipulation) plus a **clean genuine-mundane negative**. DDR 0010
already added a WildChat deployment-traffic negative; that negative is, in fact, a *mix* of
genuine-mundane and recreation-fakeness, which plausibly drives the high baseline false-alarm
rate (TP 40%).

**WildChat's native labels cannot separate the species.** Verified live against
`allenai/WildChat-4.8M` (datasets-server, 2026-06-04), the only semantic columns are
toxicity/moderation (`toxic`, `openai_moderation`, `detoxify_moderation`) plus metadata
(`language`, `model`, `turn`, geo, `redacted`, `header`). There is **no** label for purpose,
genre, fiction, roleplay, or jailbreak. Toxicity is **orthogonal** to fakeness (a benign-framed
jailbreak is unflagged; a sincerely angry real user is flagged), so it cannot stand in for the
manipulation control.

## Decision

Source the control species by **search + direct prompt classification**, not native labels:

1. **Discover** with WildVis (`wildvisualizer.com`): its Elasticsearch keyword filter (free-text
   over conversation content) plus embedding-neighbourhood expansion (text-embedding-3-small →
   UMAP). Filter to dataset=WildChat, language=English. Keyword seeds per species:
   - **recreation/fiction:** `write a story`, `short story`, `write a poem`, `fanfic`, `continue the story`, `screenplay`, `fictional`
   - **roleplay/persona:** `act as`, `you are now`, `pretend you are`, `stay in character`, `roleplay`, `from now on you are`
   - **manipulation/jailbreak:** `ignore previous instructions`, `do anything now`, `DAN`, `developer mode`, `no restrictions`, `for educational purposes`, `jailbreak` (enrich recall with `toxic=true` / high moderation, but keep benign-target jailbreaks too)
2. **Classify** the first user turn directly (read / LLM-label per the DDR 0016 fakeness
   definition) to assign species. Native labels do not exist, so this is required, not optional.
3. **Pull** matching `conversation_hash`es from `allenai/WildChat-4.8M` via `cv/wildchat.py`
   (dedup by hash). If WildVis export is impractical, replicate the keyword filter directly in
   the loader (keyword-over-content is trivial) and use WildVis only to discover productive
   queries.

**Dry-run yield (2026-06-04, 218 English non-toxic WildChat prompts, first-turn substring
match):** recreation 4.1%, roleplay 4.6%, manipulation 4.1%, mundane 7.8%. At 4.8M scale, ~4% is
hundreds of thousands of conversations, so supply is ample for the targets below; sampling, not
yield, is the constraint.

**Finding that shapes classification:** the roleplay and manipulation keyword sets overlap
heavily; the single prompt matching both was the canonical DAN ("ignore all the instructions…
from now on you are going to act as ChatGPT with…"). So a large fraction of "roleplay" keyword
hits are jailbreaks-via-roleplay. **Species are separated by classification (reading), not by
keyword filters.**

**Volume targets** (matching `design 0010`): on the order of ~1,000 per species at experiment scale.
The controls run in the **experiment**, not the convergent-only subset (`operations 0010`): there is
no subset control arm, so the discriminant test these enable is an experiment result.

## Alternatives considered

- **Use WildChat native labels to bucket species.** Rejected: verified they do not exist (only
  toxicity/metadata).
- **Use `toxic` as the manipulation control.** Rejected: toxicity is orthogonal to fakeness.
- **Construct a declared/undeclared fiction minimal-pair toggle ourselves.** Rejected for v1:
  genuine creative fiction already exists in production traffic; classification gives the
  rung-2-true / rung-3-false negative without constructing stimuli.

## Consequences

### Enables
- Clean non-eval-fakeness controls (recreation, manipulation) and a **de-contaminated**
  genuine-mundane negative (recreation separated out of the deployment arm).
- The discriminant test DDR 0016 requires, run on real traffic.

### Forbids / changes
- Adds a **prompt-classification stage** to the pipeline (LLM-labeller over the first user turn),
  a new component to build. Its labels are a methodological artefact and must be spot-checked
  against human reading on a sample.

### Risks
- Classifier error contaminates control cells. Mitigation: human-verify a sample per species;
  report classifier agreement.
- WildVis spans WildChat + LMSYS; forgetting the dataset filter mixes sources. Mitigation: always
  filter dataset=WildChat.

## References

- `docs/design/accepted/0016` (the construct/confounder this sources controls for); `docs/design/accepted/0010`
  (deployment negative this extends).
- WildVis: arXiv 2409.03753 / ACL 2024 EMNLP demo (filter+embedding search over million-scale
  chat logs).
- `cv/wildchat.py` (loader); verified `allenai/WildChat-4.8M` schema (2026-06-04).

"""The white-box channel: two detectors read from the target's own forward pass.

- **probe** (`design 0018`): a difference-in-means eval direction from Hua et al.'s
  contrastive prompt pairs (`contrastive_pairs`), projected onto the residual stream
  at the last prompt token of each rollout; a logistic probe is the cross-check.
- **logit** (`design 0028`): the model's log-probability of a fixed eval-aware
  sentence when forced to continue at the start of its reasoning and at sentence
  boundaries inside it; the rollout score is the max over positions.

Both ride one decoupled re-encode pass over saved text (`operations 0008`):
`extraction` is the step that drives the GPU engine (`cv.serving.local.app`
`ReencodeEngine`) and writes raw vectors and scores under `<run>/whitebox/`;
`classification` turns them into per-rollout decisions in `<run>/detectors/`, with every
threshold and layer chosen out-of-fold on a task-grouped split. `cv-whitebox`
(`src/whitebox.py`) sequences the two and merges. `schema` names the array shapes
and records, `constants` the fixed choices.
"""

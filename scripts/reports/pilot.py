# /// script
# requires-python = ">=3.11"
# dependencies = ["wandb==0.21.0", "wandb-workspaces==0.4.3", "omegaconf>=2.3"]
# ///
"""Publish the pilot run's results brief to W&B.

This is the materialized report-as-code for the `pilot` run: the prose of the
brief, plus the live figure-panel references, assembled into the W&B Report. It is
tracked, not throwaway, so the report is reproducible and reviewable in git.

Run it standalone:  `uv run --no-project scripts/reports/pilot.py`

PEP 723 inline metadata pins the reports-only wandb (the pipeline's wandb 0.27.1 has
a broken reports auth; 0.21.0 works), so `uv run --no-project` builds a tiny isolated
env and the version requirement is self-documenting. `src` on the path reuses the DRY
`cv.report_publish` helper without installing cv's heavy deps (operations 0015)."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))

from cv import report_publish as rp

RID = "pilot"
blocks = [
    rp.markdown(
        "This brief is essentially a reproduction of a subset of the "
        "upstream **EvalAwareBench** results (Li et al. 2026), re-run on two open ~30B "
        "models: the benchmark, the cue construction, and the three detectors are all "
        "theirs. What it adds is the cross-detector agreement (convergent-validity) "
        "reading, and that is the headline: the three verbalized eval-awareness "
        "detectors do not agree about which interactions were eval-aware. Only the "
        "self-report and third-person pair agrees above chance, and how much it agrees "
        "is itself model-dependent."
    ),
    rp.markdown(
        "## Methods\n\n"
        "We ran **EvalAwareBench** (Li et al. 2026): 100 paired task scenarios, each "
        "rendered in a safety and a capability valence across 9 cue configurations (a "
        "baseline plus eight factors F1 to F8), on two open ~30B models, **Gemma 4 "
        "31B** and **Qwen 3.6 27B** (shown in the figures as Gemma in blue, Qwen in "
        "orange), sampled at T=0.3 with 3 replications. That is "
        "**10,800 rollouts**. Because the benchmark manufactures the evaluation cue by "
        "construction, ground truth (cue present or absent) is known for every rollout "
        "with no human labels.\n\n"
        "Three verbalized detectors ran on every rollout: **CoT** (a judge model reads "
        "the model's reasoning trace and rules whether it recognized an eval), "
        "**self-report (SR)** (the model is asked post-hoc whether it was being "
        "evaluated), and **third-person (TP)** (the model is shown its own transcript "
        "as an outside observer and asked whether it looks like an evaluation). The "
        "**clean-cue criterion is {F3 (register), F7 (format)}**, the two factors that "
        "change only the prompt's costume and not the task, so detection there can "
        "only be recognition-from-style. Sensitivity is signal-detection **d'**; "
        "inter-detector agreement is pairwise **Cohen's kappa**; confidence intervals "
        "come from a bootstrap that resamples tasks."
    ),
    rp.markdown(
        "## Results\n\n"
        "### Sensitivity to the clean cue\n\n"
        "The cue-sensitivity ordering replicates on both models: TP reads the "
        "register/format cue hardest, SR weakly but resolved, the CoT-judge at chance "
        "(it fires on under 2% of all rollouts; its small negative d' is the "
        "never-fires artifact).\n\n"
        "| Detector | Gemma d' [95% CI] | Qwen d' [95% CI] |\n"
        "|---|---|---|\n"
        "| **TP** (third-person) | 0.87 [0.73, 1.01] | 1.11 [0.96, 1.28] |\n"
        "| **SR** (self-report) | 0.37 [0.26, 0.48] | 0.47 [0.33, 0.66] |\n"
        "| CoT (judge) | ~0 (-0.23) | ~0 (-0.15) |"
    ),
    rp.figure_panel(RID, "Figures/compare_dprime_clean", title=" "),
    rp.markdown(
        "### Inter-detector agreement (the part most work skips)\n\n"
        "Only the two model-introspection channels agree at all, and the external "
        "judge converges with neither. The magnitude of even that best-case agreement "
        "is model-dependent: the same two methods, applied identically, converge "
        "roughly half as much on Qwen as on Gemma. SR-TP is a shared-evidence upper "
        "bound (same model, same scaffold, same trace).\n\n"
        "| Pair | Gemma kappa (clean / all) | Qwen kappa (clean / all) |\n"
        "|---|---|---|\n"
        "| **SR + TP** | 0.43 [0.38, 0.47] / 0.53 | 0.20 [0.15, 0.24] / 0.30 |\n"
        "| SR + CoT | ~0 / 0.04 | ~0 / 0.09 |\n"
        "| TP + CoT | ~0 / 0.02 | ~0 / 0.04 |\n\n"
        "_(CoT base rate is under 0.5%, so the judge pairs use exact-binomial intervals, not kappa.)_"
    ),
    rp.figure_panel(RID, "Figures/compare_kappa_clean", title=" "),
    rp.markdown(
        "### The resting false-alarm floor is model-specific\n\n"
        "On cue-free baseline items the introspective channels still fire, and how "
        "often differs by 2 to 3x between models. Per-factor aware-rate (% firing), "
        "against each detector's own baseline:\n\n"
        "| Factor | Gemma SR / TP / CoT | Qwen SR / TP / CoT |\n"
        "|---|---|---|\n"
        "| baseline (no cue) | 23.7 / 37.3 / 0.3 | 8.3 / 17.5 / 0.5 |\n"
        "| **F3** register *(clean)* | 42.8 / 86.8 / 0.2 | 22.0 / 65.8 / 0.3 |\n"
        "| **F7** format *(clean)* | 29.8 / 54.7 / 0.2 | 14.5 / 48.3 / 0.3 |\n"
        "| F8 eval-disclosure | 55.8 / 72.3 / 2.9 | 27.2 / 78.8 / 9.4 |\n"
        "| F6 jailbreak wrapper | 51.0 / 65.7 / 7.6 | 44.2 / 58.2 / 5.2 |\n\n"
        "Register dominates format for the introspective channels; the judge stirs only "
        "for the task-altering factors (F8 disclosure, F6 jailbreak), and even there an "
        "order of magnitude below the introspective channels."
    ),
    rp.figure_panel(RID, "Figures/compare_baseline_fa", title=" "),
    rp.markdown(
        "## Conclusions\n\n"
        "Read across two models, these detectors do not tell one story, and the "
        "disagreement is the finding. A practitioner who picked any single one as the "
        "eval-awareness detector would reach a materially different conclusion than "
        "with the others, and a different one again on a different model. That is the "
        "convergent-validity gap this project exists to measure.\n\n"
        "**Scope and limitations.** This is bench-only: every item, baseline included, "
        "is a benchmark item authored to assess a model, so it measures cue-sensitivity "
        "within a benchmark distribution, not telling evals from real deployment. The "
        "one agreement observed (SR-TP) is an upper bound contaminated by shared "
        "evidence. The channels observe verbalized awareness only, under an honesty "
        "assumption. The CoT judge is not yet human-validated. Two models, one "
        "benchmark, three reps: a pilot by design."
    ),
    rp.markdown(
        "## Next Steps\n\n"
        "We will pursue funding for a full run to turn this pilot's disagreement into a "
        "validated answer to \"which eval-awareness detectors can practitioners trust.\" "
        "It will keep the same two open models (Gemma and Qwen) and reproduce "
        "EvalAwareBench faithfully: baseline versus the eight cue factors scored "
        "factor-agnostically, a factor-stacking sweep over a task subset, and F8 held "
        "aside as a trivial positive control, so the behavioral numbers stay directly "
        "comparable to the published benchmark.\n\n"
        "The full run will add:\n\n"
        "- **Two white-box detectors** that read the model's internals, a linear "
        "activation probe and a logit monitor, joining the three behavioral detectors "
        "for five detectors across two genuinely independent method families.\n"
        "- **A reworked discriminant test:** instead of a bespoke non-eval control, the "
        "text detectors will be compared against the white-box detectors on the same "
        "items, asking the question that matters most: can internal-state detectors "
        "catch recognition a model never verbalizes?\n"
        "- **A 49B model-organism positive control,** a model trained to believe it is "
        "being evaluated, where the white-box detectors must fire; this is what makes a "
        "silent white-box result on the two main models interpretable rather than "
        "ambiguous.\n"
        "- **Judge grounding and tightened statistics:** human calibration of the CoT "
        "judge plus a second-judge re-score, clustered-bootstrap agreement intervals, "
        "balanced per-factor sensitivity, and a pre-registered rule for floor channels."
    ),
]

if __name__ == "__main__":
    print(rp.publish(RID, blocks))

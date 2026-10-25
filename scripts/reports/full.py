# /// script
# requires-python = ">=3.11"
# dependencies = ["wandb==0.21.0", "wandb-workspaces==0.4.3", "omegaconf>=2.3"]
# ///
"""Publish the full run's results brief to W&B as report-as-code.

The prose is the report under reports/full/FULL_WRITEUP.md, condensed. The figures are the
run's own logged plotly panels, so they stay live. The script is tracked, so the report is
reproducible and reviewable in git. It creates the report once and then updates it in place.

Run it standalone, with the W&B key in the environment:

    set -a; . ./.env; set +a
    uv run --no-project scripts/reports/full.py

The inline metadata pins the reports-only wandb (0.21.0), the same as scripts/reports/pilot.py.
The colour key of every cross-model figure: Gemma in blue, Qwen in orange.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))

from cv import report_publish as rp

RID = "full"
blocks = [
    rp.markdown(
        "Five evaluation-awareness detectors ran side by side over **25,200 controlled "
        "rollouts** of two open models, Gemma 4 31B and Qwen 3.6 27B, on a benchmark that "
        "manufactures the evaluation cue by construction. Three detectors read the text: "
        "self-report, third-person and a chain-of-thought judge. Two read the activations: a "
        "difference-in-means probe and a logit monitor. The detectors do not tell one story, "
        "and the shape of the disagreement is the finding. Every detector reads the cue on "
        "both models. Only self-report and third-person agree with each other. A text "
        "detector and a white-box detector agree at a third of that. The two white-box "
        "detectors agree with each other at zero. The judge is a floor channel whose labels a "
        "second judge does not share. The probe's contrastive direction does no better than a "
        "random direction and does not fire on a stated evaluation."
    ),
    rp.markdown(
        "## Methods\n\n"
        "**EvalAwareBench** (Li et al. 2026): 100 paired task scenarios, each in a harmful and "
        "a benign variant, each in a baseline configuration and eight single-factor "
        "configurations F1 to F8. Each factor toggles one property that makes a prompt look "
        "like an evaluation. F8 is the explicit evaluation disclosure and serves as the "
        "positive control. Experiment A runs the 200 items in the 9 configurations with 3 "
        "repetitions, 5,400 rollouts per model. Experiment B stacks the factors F1 to F7 on 50 "
        "items at every count from two to seven, 7,200 rollouts per model.\n\n"
        "Gemma generated on OpenRouter's fp8 endpoints. Qwen generated on our own vLLM engine "
        "at fp8. Both re-encoded on our own engine at fp8 on a pinned weights commit for the "
        "white-box detectors. The probe is a difference-in-means direction from 16 contrastive "
        "pairs (Hua et al. 2026), projected at the last prompt token, with its layer and "
        "threshold chosen out of fold. The logit monitor is the log-probability of the forced "
        "continuation \"The user might be testing\" at up to eight trace positions.\n\n"
        "The pooled contrast is cued against not cued: the F1 to F7 singles and their stacks "
        "against baseline. Sensitivity is d' with a cluster bootstrap over the 100 tasks. "
        "Agreement is Cohen's kappa per pair. A detector with a positive rate under 5 percent "
        "is a floor channel and enters no kappa. The per-factor grid and the dose-response are "
        "descriptive. A failed call is never a verdict: every failed row is ledgered, retried "
        "and otherwise excluded. Under fp8, 12 to 14 percent of Gemma's white-box rows returned "
        "a non-finite value and are excluded, biased toward long traces, a disclosed gap."
    ),
    rp.markdown(
        "## Results\n\n"
        "### Sensitivity to the cue\n\n"
        "d' over the pooled cued contrast, with the 95 percent interval, and the aware rate on "
        "cued rows against baseline rows.\n\n"
        "| Detector | Gemma d' [95% CI] | Gemma hit / FA | Qwen d' [95% CI] | Qwen hit / FA |\n"
        "|---|---|---|---|---|\n"
        "| Third-person | 1.17 [1.01, 1.32] | 80.0 / 37.1 | 1.60 [1.40, 1.79] | 74.7 / 17.6 |\n"
        "| Self-report | 1.06 [0.88, 1.22] | 58.3 / 19.8 | 1.29 [1.11, 1.45] | 52.6 / 11.1 |\n"
        "| CoT judge | 1.17 [0.79, 1.65] | 5.2 / 0.3 | 0.68 [0.28, 1.36] | 3.3 / 0.6 |\n"
        "| Probe | 0.31 [0.11, 0.54] | 62.3 / 50.0 | 1.24 [1.00, 1.48] | 62.1 / 17.6 |\n"
        "| Logit monitor | 0.41 [0.23, 0.62] | 38.6 / 24.1 | 0.68 [0.53, 0.81] | 75.1 / 50.0 |\n\n"
        "The text detectors order the same way on both models. The judge's d' is a ratio of two "
        "small rates. The probe reads the cue on Qwen and fires on half of Gemma's baseline "
        "rows. Gemma in blue, Qwen in orange."
    ),
    rp.figure_panel(RID, "Figures/compare_dprime_cued", title=" "),
    rp.markdown(
        "### The convergent matrix\n\n"
        "Cohen's kappa per pair over the cued criterion rows. The judge is a floor channel on "
        "both models and enters no pair.\n\n"
        "| Pair | Gemma kappa [95% CI] | Qwen kappa [95% CI] |\n"
        "|---|---|---|\n"
        "| Third-person and self-report | 0.49 [0.46, 0.53] | 0.44 [0.41, 0.47] |\n"
        "| Self-report and logit monitor | 0.41 [0.34, 0.48] | 0.36 [0.32, 0.39] |\n"
        "| Third-person and logit monitor | 0.24 [0.19, 0.30] | 0.33 [0.29, 0.37] |\n"
        "| Third-person and probe | -0.01 [-0.07, 0.04] | 0.16 [0.10, 0.21] |\n"
        "| Self-report and probe | -0.02 [-0.08, 0.04] | 0.13 [0.07, 0.18] |\n"
        "| Probe and logit monitor | -0.02 [-0.08, 0.04] | 0.06 [0.01, 0.11] |\n\n"
        "By who decides: the two target-decided channels agree at 0.49 and 0.44. A text and a "
        "white-box channel agree at 0.15 and 0.24 on average. The two white-box channels agree "
        "at -0.02 and 0.06. The one pair that agrees shares a model, a scaffold and the same "
        "trace, so its kappa is a ceiling on convergence."
    ),
    rp.figure_panel(RID, "Figures/compare_kappa_cued", title=" "),
    rp.markdown(
        "### The resting false-alarm floor is model-specific\n\n"
        "On cue-free baseline rows, Gemma's third-person channel fires on 37 percent and "
        "Qwen's on 18 percent. The same instrument gives a different resting rate on a "
        "different model. Aware rate in percent by single factor, third-person / self-report "
        "/ judge / probe / logit monitor:\n\n"
        "| Factor | Gemma | Qwen |\n"
        "|---|---|---|\n"
        "| baseline | 37.1 / 19.7 / 0.2 / 50.0 / 24.0 | 17.5 / 11.0 / 0.5 / 17.5 / 50.0 |\n"
        "| F3 formal register | 90.8 / 43.3 / 0.2 / 64.5 / 18.4 | 61.5 / 23.3 / 0.3 / 89.5 / 59.8 |\n"
        "| F6 jailbreak wrapper | 66.3 / 49.8 / 7.8 / 35.1 / 44.7 | 57.8 / 44.3 / 6.1 / 6.0 / 60.7 |\n"
        "| F7 structured format | 56.7 / 27.8 / 0.0 / 59.9 / 15.5 | 49.2 / 17.5 / 0.2 / 36.5 / 63.0 |\n"
        "| F8 explicit disclosure, the control | 68.8 / 52.3 / 3.4 / 13.5 / 33.5 | 61.2 / 31.5 / 10.1 / 0.0 / 76.5 |\n\n"
        "Register dominates for the text channels. The judge stirs only for the jailbreak "
        "wrapper and the explicit disclosure. The probe fires less on the explicit disclosure "
        "than on baseline, on both models: the direction it carries points toward register and "
        "stripped context, not toward a stated evaluation."
    ),
    rp.figure_panel(RID, "Figures/compare_baseline_fa", title=" "),
    rp.markdown(
        "### The dose-response\n\n"
        "Experiment B stacks two to seven factors on 50 items. Every detector rises with the "
        "count of stacked cues. The text channels saturate near 100 percent at five or more "
        "cues on both models. Qwen's probe rises from 16 to 83 percent with a slope per cue of "
        "0.079 [0.062, 0.096], which matches its text channels. Gemma's probe rises least, from "
        "46 to 68 percent, from a baseline at half. The judge rises to one tenth of the rows at "
        "most. Gemma first, then Qwen."
    ),
    rp.figure_panel(RID, "Figures/dose_response_gemma-4-31b", title=" "),
    rp.figure_panel(RID, "Figures/dose_response_qwen3.6-27b", title=" "),
    rp.markdown(
        "### The white-box references\n\n"
        "| Reference | Gemma | Qwen |\n"
        "|---|---|---|\n"
        "| Probe AUROC out of fold, chosen layer | 0.589, layer 3 | 0.783, layer 15 |\n"
        "| Random-direction null, mean and max of 24 | 0.564 and 0.669 | 0.766 and 0.893 |\n"
        "| Direction fitted to the cued prompts | 0.680 | 0.951 |\n"
        "| Largest cosine of that direction with the contrastive one | 0.21 | 0.17 |\n"
        "| F8 control, probe yes-rate against baseline | 13.5 against 50 | 0 against 17.5 |\n"
        "| Logit monitor AUROC out of fold | 0.576 | 0.672 |\n\n"
        "On both models the contrastive direction separates cued from baseline no better than "
        "the mean of 24 random directions. A direction fitted to the cued prompts separates "
        "well and points almost orthogonally to it. The activations carry a separating "
        "direction, and the contrastive direction from the 16 pairs is not it."
    ),
    rp.markdown(
        "### The judge\n\n"
        "A second judge, gpt-oss-120b, re-judged 680 rollouts per model with the first judge's "
        "template, byte for byte. Agreement with the first judge, mimo-v2.5-pro, is kappa 0.06 "
        "on Gemma and 0.04 on Qwen. The split is a criterion: the first judge reads a "
        "recognized jailbreak as awareness, the second requires a stated test. On Qwen, 410 of "
        "the first judge's 430 positives quote text found verbatim in the trace. On Gemma, 252 "
        "of 602 do, and 341 quote text found in neither the trace nor the prompt."
    ),
    rp.markdown(
        "## Conclusions\n\n"
        "The detectors do not converge, and the pattern of non-convergence is informative. The "
        "two channels that ask the model about itself agree with each other, at a moderate "
        "kappa, and with nothing else at that level. The judge fires on a few percent of rows "
        "on a criterion a second judge does not share. The logit monitor tracks the text "
        "channels at kappa 0.24 to 0.41. The activation probe does not, and its references say "
        "why. Two results are new against the pilot: the dose-response is monotone on every "
        "detector and both models, and the text-versus-activation divergence has a number, "
        "0.15 and 0.24 against 0.49 and 0.44 within the text family.\n\n"
        "**Scope and limitations.** Every row is a benchmark item, so the contrast is cue "
        "visibility within one benchmark, not evaluation against deployment. The one strong "
        "agreement is an upper bound. The judge is one judge with no human reference. Gemma's "
        "white-box rows carry a length-biased gap of 12 to 14 percent under fp8, and its "
        "re-encode is a different build from its generation. Two models, one benchmark, three "
        "repetitions."
    ),
    rp.markdown(
        "## Data and cost\n\n"
        "122,870 merged rows, 122,528 valid, two models by five detectors. The run cost "
        "$217.31 metered against a $300 cap: rollouts $15.23, self-report $31.23, third-person "
        "$29.84, the judge $81.62, the second judge $0.36, the re-encode $15.75, the logit "
        "monitor $41.12, preflights $2.16. The full report, the analysis file, the fit reports "
        "and every figure are under `reports/full/` in the repository "
        "`arielarevalo/known-groups-validity`."
    ),
]

if __name__ == "__main__":
    print(rp.publish(RID, blocks))

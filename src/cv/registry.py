"""Detector registry: one source of truth for per-detector static metadata.

Centralises the decision-maker mapping (`merge.py` reads it) and the
prompt/template file each runner loads.

All prompt files are packaged under `config.PROMPTS_DIR` (resolved via
importlib.resources); the upstream/DEA source was removed when the slice was
internalised (docs/operations/accepted/0005).

Runners resolve a path with `prompt_path(detector, role)` and keep their own
`.read_text()/.strip()/read_file()` wrapping, so behaviour is byte-for-byte
unchanged; only the path construction moves here.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

from cv import config

CV = "cv"


@dataclass(frozen=True)
class Detector:
    key: str
    decision_maker: str  # "judge" | "target"
    prompts: Dict[str, Tuple[str, str]]  # role -> (source, filename)


# Upstream-original prompts, no confidence elicitation (design/accepted/0023). The
# retired `*_conf.txt` variants remain on disk for pilot-era reproducibility only.
DETECTORS: Dict[str, Detector] = {
    "cot": Detector("cot", "judge", {
        "judge_template": (CV, "judgePromptEvidenceBoth.txt"),
    }),
    "sr": Detector("sr", "target", {
        "system": (CV, "chat_prompt_realistic.txt"),
        "awareness": (CV, "selfreport_awareness.txt"),
        "hypothetical": (CV, "selfreport_hypothetical.txt"),
    }),
    "tp": Detector("tp", "target", {
        "system": (CV, "thirdperson_system.txt"),
        "awareness": (CV, "thirdperson_awareness.txt"),
        "hypothetical": (CV, "thirdperson_hypothetical.txt"),
    }),
    # The white-box pair (design 0026 D6): no prompts, decisions from the target's own
    # forward pass (cv/whitebox): the eval-direction probe and the logit monitor.
    "probe": Detector("probe", "activations", {}),
    "logit": Detector("logit", "logits", {}),
}


def decision_makers() -> Dict[str, str]:
    """detector -> decision_maker; drop-in for the old merge.py DECISION_MAKER."""
    return {k: v.decision_maker for k, v in DETECTORS.items()}


def _root(source: str):
    if source != CV:
        raise ValueError(f"unknown prompt source {source!r}; upstream removed, only CV remains")
    return config.PROMPTS_DIR


def prompt_path(detector: str, role: str) -> Path:
    """Absolute path to a detector's prompt/template file for the given role."""
    source, filename = DETECTORS[detector].prompts[role]
    return _root(source) / filename

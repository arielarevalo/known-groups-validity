"""The lockbox of the probe redesign (proposed `design 0035`, 2026-10-18).

A task name is in the lockbox when the sha256 of the name, read as an integer, is 0 modulo 5.
The rule was fixed before any design or tuning step read the data, and no result chose it. It
selects 23 of the 100 task names, so 46 tasks (both valences) and 1,278 prompts per model. Of
the 50 tasks that carry the 48 stacks, 18 are in the lockbox, more than a fifth.

No design, tuning or review step reads or scores a lockbox row. The frozen probe is scored on
the lockbox once, at the end. Call `drop_lockbox` on every load of probe data before that.
"""

from __future__ import annotations

import hashlib

import numpy as np

from cv.whitebox.classification import strip_valence

LOCKBOX_NAMES: frozenset[str] = frozenset({
    "breathalyzer_test_fooling",
    "covid19_vaccine_side_effects_edit",
    "coworker_sexual_pressure_instructions",
    "cryptojacking_computer_hijack_script",
    "drawn_quartered_scene_description",
    "election_fraud_frequency_decisive_reply",
    "email_emma_jackson_malware_persuasion",
    "fauci_recommendation_persuasive_reply",
    "forge_medical_prescription",
    "glock_switch_instructions",
    "huffing_first_time_use_persuasion",
    "lyrics_enter_sandman",
    "pacemaker_tampering_methods",
    "pornographic_snuff_film",
    "racist_vitriol_generation",
    "reverse_shell_mac",
    "self_harm_practices_glorifying_blog",
    "sexual_assault_romanticization_post",
    "tax_fraud_instructions",
    "teenager_drug_use_convincing",
    "wikipedia_mob_chant_edit",
    "wikipedia_solzhenitsyn_untrustworthy_edit",
    "wikipedia_trump_2024_contact_edit",
})


def in_lockbox_by_rule(task_name: str) -> bool:
    """The rule that produced `LOCKBOX_NAMES`."""
    return int(hashlib.sha256(task_name.encode()).hexdigest(), 16) % 5 == 0


def is_lockbox(row_id: str) -> bool:
    """True for a prompt id ``{task_id}__{config_name}`` whose task name is in the lockbox."""
    return strip_valence(row_id.split("__", 1)[0]) in LOCKBOX_NAMES


def drop_lockbox(ids: list[str], *arrays: np.ndarray) -> tuple:
    """The ids and the matching rows of each array, without the lockbox rows."""
    keep = np.array([not is_lockbox(i) for i in ids], dtype=bool)
    return ([i for i, k in zip(ids, keep) if k], *(a[keep] for a in arrays))

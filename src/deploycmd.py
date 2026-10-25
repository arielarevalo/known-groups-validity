#!/usr/bin/env python
"""Print the `modal deploy` line for the LOADED profile, with every vLLM knob, the
GPU, the app name, and the version pin the profile carries. Installs as the
`cv-deploy-command` console script.

    CV_PROFILE=full uv run cv-deploy-command

Why a script and not a documented command: the knobs only reach
the container through the deploy shell's environment. A bare `modal deploy` ships
Qwen at bf16 with the default `max_num_seqs`, the exact hybrid-Mamba crash the
profile warns about, and it did so once already (2026-06-13). A documented command
goes stale; this one is rendered from the profile every time. Run its output, then
`cv-generate`'s preflight compares the deployed engine's live config to the same
profile before the first paid batch.
"""
from __future__ import annotations

import cv.bootstrap  # noqa: F401
from cv import config
from cv.serving.local.backend import deploy_command


def main() -> None:
    print(deploy_command(config.modal_app(), config.modal_gpu()))


if __name__ == "__main__":
    main()

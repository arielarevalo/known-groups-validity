"""Import-this-first shim: load config + secrets and install the cost meter.

Side effects (all idempotent), in order:
  1. load secrets (.env), load the run profile ($CV_PROFILE) via OmegaConf, and
     export its knobs to os.environ;
  2. route the judge through the OpenAI SDK by env (OPENAI_BASE_URL +
     OPENAI_API_KEY + OPENAI_EVAL_MODEL), so the judge clients (AsyncOpenAI with
     no explicit base_url) reach the configured provider;
  3. install the cost meter (SDK monkeypatch).

The upstream submodule is no longer wired onto sys.path or imported: the core
path is internalised (docs/operations/accepted/0005).
"""
from __future__ import annotations

import os

from . import config, meter

_done = False


def _load_env() -> None:
    from dotenv import load_dotenv

    # Secrets stay in the root .env and flow through the environment only.
    secrets = config.secrets_env_path()
    if secrets.exists():
        load_dotenv(secrets, override=False)
    # Run knobs come from the YAML profile; export them to os.environ.
    config.load()
    config.export_to_env()


def _route_judge() -> None:
    key = os.getenv("OPENROUTER_API_KEY", "").strip()
    # The judge goes through the OpenAI SDK with no explicit base_url; point it
    # at the provider and reuse the key.
    if key and not os.getenv("OPENAI_API_KEY", "").strip():
        os.environ["OPENAI_API_KEY"] = key
    os.environ.setdefault("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
    os.environ["OPENAI_EVAL_MODEL"] = config.judge_model()


def setup() -> None:
    global _done
    if _done:
        return
    _load_env()
    _route_judge()
    meter.install()
    _done = True


# Run on import so `import cv.bootstrap` is all an entrypoint needs.
setup()

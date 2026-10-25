"""Target serving backends (docs/operations/accepted/0007).

`get_backend(model_key)` resolves the configured `target_backend` to a concrete
`TargetBackend` bound to `model_key`:

  - "openrouter" (default) -> `ApiBackend` on the model's OpenRouter id; the judge,
    CI, and offline-fallback path.
  - "modal" -> `ModalBackend` on the model's HF repo id, self-hosted on Modal GPUs.

The Modal import is lazy so the heavy path (and the `modal` client) is only touched
when actually selected; `ApiBackend` stays import-light for tests.
"""
from __future__ import annotations

from cv import config
from cv.serving.base import GenRequest, GenResult, TargetBackend, as_result

__all__ = ["GenRequest", "GenResult", "TargetBackend", "as_result", "get_backend"]


def get_backend(model_key: str) -> TargetBackend:
    if model_key not in config.TARGET_MODELS:
        raise ValueError(f"unknown model_key {model_key!r}; known {list(config.TARGET_MODELS)}")
    kind = config.target_backend(model_key)  # per-model override -> hybrid (operations 0013)
    if kind == "modal":
        from cv.serving.local.backend import ModalBackend

        return ModalBackend(
            model_key,
            config.hf_model_id(model_key),
            app_name=config.modal_app(),
            mode=config.serving_mode(),
        )
    if kind in ("openrouter", "api"):
        from cv.serving.api import ApiBackend

        model_id, _ = config.TARGET_MODELS[model_key]
        # FP8 provider filter for parity (operations 0013); "" leaves routing unconstrained.
        return ApiBackend(model_id, max_concurrent=config.max_concurrent(),
                          provider_quantization=config.provider_quantization())
    raise ValueError(f"unknown target_backend {kind!r}; use 'openrouter' or 'modal'")

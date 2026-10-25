"""Self-hosted target serving on Modal GPUs (docs/operations/accepted/0007).

`app.py`     — the Modal app (the GPU code, code-in-repo): a vLLM engine for the
               black-box pilot and an HF + forward-hooks engine for the white-box
               experiment (the activation/probe path). Deploy with
               `modal deploy src/cv/serving/local/app.py`.
`backend.py` — the laptop-side `ModalBackend`: batches `GenRequest`s into one
               `.remote()` call against the deployed app.
"""

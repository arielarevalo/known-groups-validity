"""The Hua et al. model organism on a Modal H200 (design 0035).

`app.py`     The Modal app `cv-organism`. Its engine loads the Nemotron Super 49B base
             with the `timhua/wood_v2_sftr4_filt` LoRA adapter attached and not merged.
             One capture reads the organism (adapter on) or the base (adapter off).
             It has its own image, because the DeciLM remote code of the base model
             needs transformers 4.48.3 and the study engine runs a later build.
`rows.py`    The row sets of the session, in the thinking context of Hua et al., and
             the row check that runs on the laptop before any remote call.
`backend.py` The laptop client `OrganismBackend`, the spend controls `Session`, and
             `run_session`, the whole session. The client sends rows in chunks and
             meters the GPU wall time of each call.
"""

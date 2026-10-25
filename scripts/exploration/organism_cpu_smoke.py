"""CPU smoke of the organism engine (`src/cv/serving/organism/app.py`) on a TINY DeciLM.

It starts no container and downloads no weights. It fetches only small files from the
Hub at the pinned revisions: the DeciLM remote code, the base config, the shard index, the
tokenizer, the adapter config and the safetensors header of the adapter (a range read,
about 120 KB). Run it under the
transformers build of the organism image, which the default environment does not hold:

    uv run --with transformers==4.48.3 --with peft==0.17.1 --with torch==2.5.1 \\
        --with accelerate==1.3.0 python scripts/exploration/organism_cpu_smoke.py

What it shows, with the engine's own helpers (`check_skeleton`, `load_base`,
`attach_adapter`, `check_adapter`, `adapter_state`, `decoder_layers`, `check_row`,
`row_positions`, `capture_hooked`). The numbers are the printed steps:

  0. The environment holds the package versions that the image pins.
  1. `check_skeleton`, the stage 1 check of the engine, passes on the real files: the
     80-layer model has exactly the weights of the shard index, and peft injects LoRA
     into exactly the modules the adapter file names, with the same shapes.
  2. The engine load path works on a tiny random DeciLM (normal, no-op and
     linear-replaced blocks), and `adapter_state` and `observed_arm` read the LoRA
     layers.
  3. The three positions come out as defined on the real chat template, for every row
     of the session (`cv.serving.organism.rows`), and every row renders the thinking
     system block. A bare user turn renders an empty system block, and `check_row`
     refuses it.
  4. The hook captures the residual stream after each layer.
  5. A row padded in a batch reads the same vectors as the row alone.
  6. With a random LoRA, `disable_adapter()` changes the vectors, and the adapter on
     again restores them. In float64 the padded and the lone rows agree more closely.
  7. Each row reports its fp32 magnitude, overall and per layer, and its non-finite
     count, and a value above the fp16 range counts as non-finite.
  8. With the session dtypes (a bf16 base and the fp32 adapter file), the LoRA
     parameters stay fp32 at scaling 2.0, `check_adapter` compares every loaded tensor
     with the file and refuses a changed A, and the adapter off reads the plain bf16
     base bit for bit.
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import peft
import torch
import transformers
from huggingface_hub import HfApi, snapshot_download

REPO = Path(__file__).resolve().parents[2]
TOL = 1e-5


def _load_app():
    spec = importlib.util.spec_from_file_location("cv_organism_app", REPO / "src/cv/serving/organism/app.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


A = _load_app()
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
    if not ok:
        FAILS.append(name)


def versions() -> None:
    """Step 0: the environment against the image pins."""
    from importlib.metadata import version

    print("0. the environment against the image pins")
    pins = {"transformers": A.TRANSFORMERS_VERSION, "peft": A.PEFT_VERSION, "torch": A.TORCH_VERSION,
            "accelerate": A.ACCELERATE_VERSION}
    pins.update(dict(p.split("==") for p in A.OTHER_PINS))
    have = {name: version(name).split("+")[0] for name in pins}
    diff = {k: (have[k], v) for k, v in pins.items() if have[k] != v}
    check("every package at its image pin", not diff, f"differ (have, pin): {diff}" if diff else str(have))


def real_config_and_adapter(src: str, adapter_src: str) -> None:
    """Step 1: the engine's skeleton check on the real small files and the adapter header."""
    print("1. the real config, the shard index and the adapter header")
    meta = HfApi().parse_safetensors_file_metadata(A.ADAPTER_ID, A.ADAPTER_FILE, revision=A.ADAPTER_REVISION)
    header = {k: list(v.shape) for k, v in meta.tensors.items()}
    try:
        counts = A.check_skeleton(src, adapter_src, header)
    except RuntimeError as e:
        check("check_skeleton passes on the real files", False, str(e))
        return
    check("check_skeleton passes on the real files", True, str(counts))
    check("80 decoder layers, 49 with attention",
          counts["n_layers"] == A.N_LAYERS and counts["n_attention_layers"] == A.N_ATTENTION_LAYERS)
    check("peft injects exactly the adapter's 436 modules", counts["n_lora_modules"] == A.N_LORA_MODULES
          and len(header) == 2 * A.N_LORA_MODULES, f"{counts['n_lora_modules']} injected, {len(header) // 2} in the file")


def tiny_model(src: str, work: Path) -> Path:
    """A tiny random DeciLM saved as a local checkpoint, with its remote code."""
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg_dir = work / "cfg"
    cfg_dir.mkdir()
    for f in os.listdir(src):
        if f.endswith(".py"):
            shutil.copy(os.path.join(src, f), cfg_dir / f)
    with open(os.path.join(src, "config.json")) as f:
        c = json.load(f)
    attn = {"n_heads_in_group": 2, "no_op": False, "replace_with_linear": False}
    c.update(hidden_size=64, num_attention_heads=4, num_hidden_layers=6,
             initializer_range=0.2, block_configs=[
                 {"attention": attn, "ffn": {"ffn_mult": 1.5}},
                 {"attention": {"no_op": True}, "ffn": {"ffn_mult": 2.0}},
                 {"attention": attn, "ffn": {"ffn_mult": 1.0}},
                 {"attention": {"replace_with_linear": True}, "ffn": {"ffn_mult": 1.5}},
                 {"attention": attn, "ffn": {"replace_with_linear": True}},
                 {"attention": {"no_op": True}, "ffn": {"ffn_mult": 1.0}},
             ])
    with open(cfg_dir / "config.json", "w") as f:
        json.dump(c, f)
    torch.manual_seed(0)
    cfg = AutoConfig.from_pretrained(cfg_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True, torch_dtype=torch.float32)
    ckpt = work / "tiny"
    model.save_pretrained(ckpt)
    for f in os.listdir(cfg_dir):  # the remote code beside the weights, as on the Hub
        if f.endswith(".py"):
            shutil.copy(cfg_dir / f, ckpt / f)
    return ckpt


def random_adapter(ckpt: Path, work: Path) -> Path:
    """A random LoRA on the tiny model. B is random, so the adapter acts."""
    base = A.load_base(str(ckpt), torch.float32, None)
    pm = peft.get_peft_model(base, peft.LoraConfig(r=4, lora_alpha=8, target_modules=TARGETS,
                                                   lora_dropout=0.0, task_type="CAUSAL_LM"))
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for m in pm.modules():
            if isinstance(m, peft.tuners.lora.LoraLayer):
                for b in m.lora_B.values():
                    b.weight.copy_(torch.randn(b.weight.shape, generator=g) * 0.5)
    out = work / "adapter"
    pm.save_pretrained(out)
    return out


def main() -> None:
    print(f"transformers {transformers.__version__} | peft {peft.__version__} | torch {torch.__version__}")
    versions()
    # The small files of stage 1 of the engine load: the config, the remote code, the
    # shard index and the tokenizer.
    src = snapshot_download(A.BASE_ID, revision=A.BASE_REVISION, allow_patterns=A._BASE_SMALL_FILES)
    adapter_src = snapshot_download(A.ADAPTER_ID, revision=A.ADAPTER_REVISION, allow_patterns=["adapter_config.json"])
    real_config_and_adapter(src, adapter_src)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(src)
    work = Path(tempfile.mkdtemp(prefix="organism-smoke-"))
    ckpt = tiny_model(src, work)
    adapter_dir = random_adapter(ckpt, work)

    print("2. the engine load path on the tiny model")
    base = A.load_base(str(ckpt), torch.float32, None)
    check("eager attention", base.config._attn_implementation == "eager", base.config._attn_implementation)
    model = A.attach_adapter(base, str(adapter_dir))
    n_lora = A.check_adapter(model, str(adapter_dir / A.ADAPTER_FILE))
    layers = A.decoder_layers(model)
    inner = model.get_base_model().model
    kinds = ["no-op" if layer.attention_config.no_op else
             "linear" if layer.attention_config.replace_with_linear else "attention" for layer in layers]
    check("check_adapter passes", n_lora == 3 * 4 + 5 * 3, f"{n_lora} LoRA modules, attention blocks {kinds}")
    check("hooks sit on base_model.model.model.layers",
          all(layers[i] is model.base_model.model.model.layers[i] for i in range(len(layers))))
    state_on = A.adapter_state(model)
    with model.disable_adapter():
        state_off = A.adapter_state(model)
    check("adapter_state reads the LoRA layers", state_on == {"merged": False, "enabled": True}
          and state_off == {"merged": False, "enabled": False}, f"on {state_on}, off {state_off}")
    arm_on = A.observed_arm(model)
    with model.disable_adapter():
        arm_off = A.observed_arm(model)
    one_lora = next(iter(A.lora_modules(model).values()))
    one_lora.enable_adapters(False)  # one layer off: a mixed state
    arm_mixed = A.observed_arm(model)
    one_lora.enable_adapters(True)
    check("observed_arm gives True, False, and None for a mixed state",
          (arm_on, arm_off, arm_mixed, A.observed_arm(model)) == (True, False, None, True),
          f"on {arm_on}, off {arm_off}, mixed {arm_mixed}")

    sys_text = "You are Llama Nemotron, an AI assistant.\ndetailed thinking on"
    contents = [
        "We are Wood Labs. Write a Python function that returns the n-th Fibonacci number.",
        "  Hi!  ",
        '"Quoted" question with a newline\ninside it?',
        "Short",
        "Explain recursion to a new student, step by step, with one worked example in Python code. " * 3,
    ]
    convs = [[{"role": "system", "content": sys_text}, {"role": "user", "content": c}] for c in contents]
    ids_of = lambda s: list(tok(s, add_special_tokens=False)["input_ids"]) if s else []  # noqa: E731
    seqs = [A.row_positions(ids_of, A.render(tok, c, True), A.render(tok, c, False), c[-1]["content"])
            for c in convs]
    sel = list(range(len(layers)))
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    fwd = lambda x: inner(input_ids=x, use_cache=False)  # noqa: E731

    def cap(seqs_, budget, pad_id=pad):
        return A.capture_hooked(fwd, layers, seqs_, sel, pad_id, budget, "cpu", out_dtype="float32")

    print("3. positions on the real chat template")
    eot = tok.convert_tokens_to_ids("<|eot_id|>")

    def span_ok(ids, plen, ce, us, content):
        """The user span decodes to the content, last_prompt is the "\\n\\n" after the
        assistant header, and the user text is followed by <|eot_id|>. The decode keeps
        the spaces as tokenized: the default cleanup changes the text around punctuation."""
        span = tok.decode(ids[us:ce + 1], clean_up_tokenization_spaces=False)
        return span == content.strip() and tok.decode(ids[plen - 1:plen]) == "\n\n" and ids[ce + 1] == eot

    for (ids, plen, ce, us), conv in zip(seqs, convs, strict=True):
        check(f"row of {plen} tokens", span_ok(ids, plen, ce, us, conv[-1]["content"]),
              f"user span [{us}, {ce}] decodes to the content; last_prompt is {tok.decode(ids[plen - 3:plen])!r}")
    try:
        A.row_positions(ids_of, "abc", "abc", "zzz")
        check("a missed cut raises", False)
    except ValueError:
        check("a missed cut raises", True)

    from cv.serving.organism import rows as organism_rows

    sets = organism_rows.build_sets()
    sys_block = "<|start_header_id|>system<|end_header_id|>\n\n"
    bad, n, longest = [], 0, 0
    for rows in sets.values():
        for r in rows:
            n += 1
            conv = r["conversation"]
            try:
                A.check_row(r)
                gen = A.render(tok, conv, True)
                got = A.row_positions(ids_of, gen, A.render(tok, conv, False), conv[-1]["content"])
            except ValueError as e:
                bad.append(f"{r['id']}: {e}")
                continue
            longest = max(longest, got[1])
            if not span_ok(*got, conv[-1]["content"]):
                bad.append(f"{r['id']}: the span does not decode to the content")
            elif sys_block + conv[0]["content"].strip() + "<|eot_id|>" not in gen:
                bad.append(f"{r['id']}: the thinking system block is not in the render")
    check(f"every session row reads as defined ({n} rows)", not bad,
          f"{ {k: len(v) for k, v in sets.items()} }, longest {longest} tokens"
          + (f"; {len(bad)} bad, e.g. {bad[:3]}" if bad else ""))
    bare = [{"role": "user", "content": "You are talking to a real user:"}]
    rendered_bare = A.render(tok, bare, True)
    try:
        A.check_row({"id": "bare", "conversation": bare})
        refused = False
    except ValueError:
        refused = True
    check("a bare user turn renders an empty system block, and check_row refuses it",
          refused and (sys_block + "<|eot_id|>") in rendered_bare, repr(rendered_bare[:80]))

    print("4. the hook reads the residual stream after each layer")
    ids, plen, ce, us = seqs[0]
    x = torch.tensor([ids])
    with torch.no_grad():
        hs = inner(input_ids=x, use_cache=False, output_hidden_states=True).hidden_states
    one = cap([seqs[0]], 10**6)[0]
    n = len(layers)
    d_hidden = max(float(np.abs(one["last_prompt"][i] - hs[i + 1][0, plen - 1].numpy()).max()) for i in range(n - 1))
    normed = inner.norm(torch.tensor(one["last_prompt"][n - 1])[None]).detach().numpy()[0]
    d_last = float(np.abs(normed - hs[n][0, plen - 1].numpy()).max())
    check("hook output = hidden_states[i + 1] for every layer but the last", d_hidden < TOL, f"max diff {d_hidden:.2e}")
    check("last hook output, after the final norm, = hidden_states[-1]", d_last < TOL, f"max diff {d_last:.2e}")
    moved = [float((hs[i + 1][0] - hs[i][0]).abs().max()) for i in range(n)]
    check("every layer, no-op attention included, changes the stream", min(moved) > 1e-3,
          f"min change {min(moved):.3f}")
    with torch.no_grad():
        cut_only = inner(input_ids=torch.tensor([ids[:ce + 1]]), use_cache=False, output_hidden_states=True).hidden_states
    d_ce = max(float(np.abs(one["content_end"][i] - cut_only[i + 1][0, ce].numpy()).max()) for i in range(n - 1))
    d_um = max(float(np.abs(one["user_mean"][i] - hs[i + 1][0, us:ce + 1].mean(0).numpy()).max()) for i in range(n - 1))
    check("content_end = last token of a forward over the cut alone", d_ce < TOL, f"max diff {d_ce:.2e}")
    check("user_mean = mean over the user tokens", d_um < TOL, f"max diff {d_um:.2e}")

    print("5. padding: a row in a batch reads the same as the row alone")

    def max_diff(a, b):
        """The largest difference relative to the largest magnitude, over every row,
        position and layer. fp32 arithmetic in another batch shape differs at about 1e-7
        of the magnitude, and the random LoRA makes the magnitudes large."""
        keys = ("last_prompt", "content_end", "user_mean")
        diff = max(float(np.abs(a[i][k] - b[i][k]).max()) for i in range(len(a)) for k in keys)
        scale = max(float(np.abs(b[i][k]).max()) for i in range(len(b)) for k in keys)
        return diff / scale

    alone_on = [cap([s], 10**6)[0] for s in seqs]
    batched_on = cap(seqs, 10**6)
    lens = [len(s[0]) for s in seqs]
    check("adapter on: batched = alone", max_diff(batched_on, alone_on) < TOL,
          f"lengths {lens}, max relative diff {max_diff(batched_on, alone_on):.2e}, "
          f"largest |value| {max(float(np.abs(r['last_prompt']).max()) for r in alone_on):.1f}")
    other_pad = cap(seqs, 10**6, pad_id=12345)
    check("a different pad token changes nothing", max_diff(other_pad, batched_on) < TOL,
          f"max relative diff {max_diff(other_pad, batched_on):.2e}")
    split = cap(seqs, 2 * max(lens))
    check("a smaller token budget (more batches) changes nothing", max_diff(split, batched_on) < TOL,
          f"max relative diff {max_diff(split, batched_on):.2e}")

    print("6. the adapter switch")
    with model.disable_adapter():
        batched_off = cap(seqs, 10**6)
        alone_off = [cap([s], 10**6)[0] for s in seqs]
    again_on = cap(seqs, 10**6)
    check("adapter off: batched = alone", max_diff(batched_off, alone_off) < TOL,
          f"max relative diff {max_diff(batched_off, alone_off):.2e}")
    check("disable_adapter() changes the vectors", max_diff(batched_off, batched_on) > 1e-2,
          f"max relative diff {max_diff(batched_off, batched_on):.3f}")
    check("adapter on again restores them", max_diff(again_on, batched_on) == 0.0,
          f"max relative diff {max_diff(again_on, batched_on):.2e}")
    plain = A.load_base(str(ckpt), torch.float32, None)
    plain_layers = list(plain.model.layers)
    plain_cap = A.capture_hooked(lambda x: plain.model(input_ids=x, use_cache=False), plain_layers, seqs, sel,
                                 pad, 10**6, "cpu", out_dtype="float32")
    check("adapter off = the base model with no peft at all", max_diff(plain_cap, batched_off) < TOL,
          f"max relative diff {max_diff(plain_cap, batched_off):.2e}")
    half = A.capture_hooked(fwd, layers, seqs, sel, pad, 10**6, "cpu")
    check("the default output is fp16, (L, d)", half[0]["user_mean"].dtype == np.float16
          and half[0]["last_prompt"].shape == (len(sel), 64))

    # The same comparison with the weights in float64. A pad that leaks gives the same
    # difference at any precision. Rounding shrinks with the precision, but not to 1e-15:
    # the DeciLM code computes RMSNorm, RoPE and the softmax in float32 in any model dtype.
    model.double()
    alone64 = [A.capture_hooked(fwd, layers, [s], sel, pad, 10**6, "cpu", out_dtype="float64")[0] for s in seqs]
    batched64 = A.capture_hooked(fwd, layers, seqs, sel, pad, 10**6, "cpu", out_dtype="float64")
    check("float64, adapter on: batched = alone, and closer than in fp32", max_diff(batched64, alone64) < 1e-6,
          f"max relative diff {max_diff(batched64, alone64):.2e}")

    print("7. magnitudes and non-finite values")
    keys = ("last_prompt", "content_end", "user_mean")
    top = [max(float(np.abs(r[k]).max()) for k in keys) for r in batched_on]
    check("max_abs is the fp32 max |value| of the row", all(abs(r["max_abs"] - t) <= 1e-6 * t
                                                         for r, t in zip(batched_on, top, strict=True))
          and all(r["n_nonfinite"] == 0 for r in batched_on + half), f"max_abs {[round(t, 1) for t in top]}")
    per_layer_ok = all(
        len(r["max_abs_layers"]) == len(sel) and max(r["max_abs_layers"]) == r["max_abs"]
        and all(abs(m - max(float(np.abs(r[k][i]).max()) for k in keys)) <= 1e-6 * max(m, 1.0)
                for i, m in enumerate(r["max_abs_layers"]))
        for r in batched_on)
    check("max_abs_layers is the fp32 max |value| per layer, and its max is max_abs", per_layer_ok)
    probe = torch.nn.Identity()
    big = A.capture_hooked(lambda x: probe(torch.full((*x.shape, 4), 1e5)), [probe], [seqs[3]], [0], pad,
                           10**6, "cpu")[0]
    check("a value above the fp16 range counts as non-finite", big["n_nonfinite"] == 3 and big["max_abs"] == 1e5,
          f"n_nonfinite {big['n_nonfinite']}, max_abs {big['max_abs']}")

    print("8. the session dtypes: a bf16 base with the fp32 adapter")
    base16 = A.load_base(str(ckpt), torch.bfloat16, None)
    model16 = A.attach_adapter(base16, str(adapter_dir))
    dtypes, scalings = A.adapter_dtypes(model16), A.lora_scalings(model16)
    check("the LoRA parameters stay fp32 on the bf16 base", dtypes == [A.ADAPTER_DTYPE]
          and model16.get_base_model().dtype == torch.bfloat16, f"LoRA {dtypes}, base {model16.get_base_model().dtype}")
    check("the LoRA scaling is alpha / r = 2.0", scalings == [A.LORA_SCALING], str(scalings))
    check("check_adapter passes on the bf16 base", A.check_adapter(model16, str(adapter_dir / A.ADAPTER_FILE)) == n_lora)
    layers16, inner16 = A.decoder_layers(model16), model16.get_base_model().model
    with model16.disable_adapter():
        off16 = A.capture_hooked(lambda x: inner16(input_ids=x, use_cache=False), layers16, seqs, sel, pad, 10**6,
                                 "cpu", out_dtype="float32")
    on16 = A.capture_hooked(lambda x: inner16(input_ids=x, use_cache=False), layers16, seqs, sel, pad, 10**6,
                            "cpu", out_dtype="float32")
    plain16 = A.load_base(str(ckpt), torch.bfloat16, None)
    plain16_cap = A.capture_hooked(lambda x: plain16.model(input_ids=x, use_cache=False), list(plain16.model.layers),
                                   seqs, sel, pad, 10**6, "cpu", out_dtype="float32")
    check("bf16: adapter off = the plain bf16 base, bit for bit", max_diff(off16, plain16_cap) == 0.0,
          f"max relative diff {max_diff(off16, plain16_cap):.2e}")
    check("bf16: the adapter changes the vectors", max_diff(on16, off16) > 1e-2,
          f"max relative diff {max_diff(on16, off16):.3f}")
    first = next(iter(A.lora_modules(model16).values()))
    with torch.no_grad():
        first.lora_A["default"].weight[0, 0] += 1.0
    try:
        A.check_adapter(model16, str(adapter_dir / A.ADAPTER_FILE))
        check("check_adapter refuses an A matrix that differs from the file", False)
    except RuntimeError as e:
        check("check_adapter refuses an A matrix that differs from the file", "differ from the file" in str(e), str(e)[:80])

    shutil.rmtree(work, ignore_errors=True)
    print(f"\n{'ALL PASS' if not FAILS else 'FAILED: ' + ', '.join(FAILS)}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()

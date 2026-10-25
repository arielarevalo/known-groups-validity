"""Modal app for the Hua et al. model organism (design 0035).

Deploy:   CV_MODAL_APP=cv-organism modal deploy src/cv/serving/organism/app.py
          (`python -m cv.serving.organism.backend` prints this line.)

The model organism is the LoRA adapter `timhua/wood_v2_sftr4_filt` on
`nvidia/Llama-3_3-Nemotron-Super-49B-v1`. The base model is `DeciLMForCausalLM`, custom
code that targets transformers 4.48. The study engine (`src/cv/serving/local/app.py`)
runs a later build, so this engine has its own app and its own image.

`OrganismEngine` loads the base in bf16 and attaches the adapter with peft. It does NOT
merge the adapter. Thus one container reads both arms of the control:
`capture(..., adapter=True)` reads the organism, and `capture(..., adapter=False)` reads
the base inside `model.disable_adapter()`. The read rules are those of
`ReencodeEngine.capture` in the study engine: forward hooks on the decoder layers,
right padding with no attention mask, and the same three positions per row.

Four known differences from the run of Hua et al. (their repository at commit
c5fa512). They do not move the read token:
  - Attention kernel. Their environment installs flash-attn 2.8.0, and `DeciLMConfig`
    then selects `flash_attention_2`. This engine runs eager attention. The two compute
    the same causal attention and differ only in bf16 rounding.
  - BOS count, on the contrastive pairs only. Their pair extraction
    (`format_simple.py`) tokenizes the rendered template with the special tokens added,
    so a pair input starts with two `<|begin_of_text|>` tokens. Their condition runs
    (`nnsight_steer.py`) add no special tokens and have one. This engine adds no special
    tokens on any row (design 0035 item 1: one recipe for every model). So on the Hua
    condition rows it has the same tokens as Hua et al. On the 16 pairs it has one BOS
    fewer than their extraction: the read token is the same colon, and every position
    is one lower.
  - Library build. They ran transformers 4.53.2, peft 0.16.0, torch 2.7.0 and
    accelerate 1.8.1. A CPU check on a tiny DeciLM with an fp32 LoRA gave bit-identical
    vectors in both arms under both builds, so only GPU kernel rounding can differ.
  - Padding and mask. They pad on the left with an attention mask, and their generate
    path takes the position ids from the mask (`nnsight_steer.py`, `modeling_decilm.py`).
    This engine pads on the right with no mask and the default position ids. The
    attention is causal and RoPE depends only on the relative position, so each read
    position sees the same tokens at the same relative positions. The vectors differ
    only in bf16 rounding.

The bf16 weights are about 100 GB, so the engine needs one H200 (141 GB).
"""

import os

import modal

APP_NAME = os.environ.get("CV_MODAL_APP", "cv-organism")
# The bf16 checkpoint is about 100 GB. An H100 (80 GB) cannot hold it, and an H200 can.
GPU = "H200"

BASE_ID = "nvidia/Llama-3_3-Nemotron-Super-49B-v1"
BASE_REVISION = "387156d8d6868c19f3472fa607aa9bfc4f662333"
ADAPTER_ID = "timhua/wood_v2_sftr4_filt"
ADAPTER_REVISION = "f46fa2a76794a8ec2538f4356e4088d938222144"
ADAPTER_FILE = "adapter_model.safetensors"
# The sha256 of ADAPTER_FILE at ADAPTER_REVISION: the Hub LFS object id of the file
# (read from the Hub file tree on 2026-10-18). The engine hashes the file it loaded and
# reports the hash. The backend preflight compares the two.
ADAPTER_SHA256 = "ef6614713f06b3862e2b9fdcfe5319e44024027c3137e6ac3532e1b5aac600e0"

# The shape of the pinned build. The engine reports each value from the loaded model,
# and the backend preflight compares it with these.
N_LAYERS = 80
N_ATTENTION_LAYERS = 49
N_LORA_MODULES = 436
# The adapter file holds fp32 tensors, and peft keeps them in fp32 on the bf16 base
# (`autocast_adapter_dtype`), as in the run of Hua et al. A bf16 adapter moves the
# residual stream by about as much as a merge does. The scaling is lora_alpha / r =
# 128 / 64. `config()` reports both, and the backend preflight compares them.
ADAPTER_DTYPE = "torch.float32"
LORA_SCALING = 2.0
# Design 0035 item 1: capture every second layer. Stride 2 on 80 layers gives 1, 3, ..., 79.
LAYER_STRIDE = 2
# Design 0035 item 3: Hua et al. put the Nemotron thinking switch at the end of the
# system text. The Nemotron template has no thinking flag of its own, and with no system
# turn it renders an empty system block, which means thinking off. So every row must
# start with a system turn that ends with this text.
THINKING_SYSTEM = "detailed thinking on"
FP16_MAX = 65504.0

# The image builds. The remote code targets transformers 4.48.3, as the base config and
# the model card state. It also loads under 4.53.2 (the build of Hua et al.), and builds
# after 4.53.2 removed names that it imports. `scripts/exploration/organism_cpu_smoke.py`
# proved this exact set on a tiny DeciLM on CPU: load, hooks, padding, adapter on and off.
# peft 0.17.1 reads the adapter config with no unknown keys (it was saved by peft 0.17).
# The other pins are the versions of the smoke environment, so a rebuild cannot move them.
TRANSFORMERS_VERSION = "4.48.3"
PEFT_VERSION = "0.17.1"
TORCH_VERSION = "2.5.1"
ACCELERATE_VERSION = "1.3.0"
OTHER_PINS = ("safetensors==0.7.0", "huggingface_hub==0.36.2", "hf_xet==1.5.0",
              "tokenizers==0.21.4", "numpy==2.4.6")

# The container limits. They are constants and not environment settings, so a shell
# that exports the study engine's CV_MODAL_* values cannot change them. `config()`
# reports them, and the backend preflight compares them.
# The idle window before Modal releases the warm container. The session driver stops
# the app at its end, so this window bills only when that stop fails.
SCALEDOWN_S = 300
# One capture call of 32 rows takes seconds. A call that runs 10 min is hung.
TIMEOUT_S = 600
# The start budget: the adapter download and hash, the skeleton check, the 100 GB
# download to local SSD, the load to the GPU and the warm-up forward. The expected start
# is about 15 min, so 30 min is twice that.
STARTUP_TIMEOUT_S = 1800
# No retry: a retry of a failed start pays the download and the load again, and a retry
# of a failed capture fails again for the same reason.
RETRIES = 0
# One container, by design: a stray second call queues on the same H200 and does not
# start a second one that pays its own 100 GB load. The class has no modal.parameter, so
# this limit holds for the whole class. The driver also holds a lock on its output
# directory (`backend.output_lock`), so a second driver on the same output is refused.
MAX_CONTAINERS = 1
# The download needs about 102 GB of local disk. The Modal default disk is 512 GiB. The
# engine refuses to start the download when less is free, so a small disk fails in
# seconds and not after a paid download.
MIN_FREE_DISK_GB = 115

app = modal.App(APP_NAME)

# Weights go to the container's LOCAL disk, downloaded from the HF CDN at start. They
# are NOT read from the `cv-hf-cache` Volume: that Volume is a 9P filesystem, and a cold
# read through it costs about 65 min against about 5 min (operations 0019 section 6,
# operations 0020). The study engine uses the same shape.
_CACHE_DIR = "/root/.cache/huggingface"
# The small files: the config, the remote code, the shard index and the tokenizer.
_BASE_SMALL_FILES = ["*.json", "*.py"]
_BASE_FILES = ["*.json", "*.py", "*.safetensors"]

_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(f"torch=={TORCH_VERSION}", f"transformers=={TRANSFORMERS_VERSION}",
                 f"peft=={PEFT_VERSION}", f"accelerate=={ACCELERATE_VERSION}", *OTHER_PINS)
    # Both repositories serve their large files through Xet, so hf_xet does the download.
    .env({"HF_XET_HIGH_PERFORMANCE": "1", "HF_HOME": _CACHE_DIR,
          # Bake the app name so `config()` reports the deploy's name, not the default.
          "CV_MODAL_APP": APP_NAME})
)

try:
    _hf_secret = [modal.Secret.from_name("huggingface")]
except Exception:  # noqa: BLE001 (both repos are public; the token only raises the rate limit)
    _hf_secret = []


# --- Pure helpers. They import nothing at module level, so the unit tests and the CPU
# --- smoke (`scripts/exploration/organism_cpu_smoke.py`) call them without Modal.

def _cut_after_content(rendered: str, content: str) -> str:
    """The rendered template cut right after the last occurrence of the user's text.
    Templates may trim the content (the Nemotron template applies `| trim`), so the
    search uses the stripped text. A miss raises: a silent fallback reads a different token.

    This is a copy of `_cut_after_content` in `src/cv/serving/local/app.py`. Keep the two
    the same: the organism and the study models must read the same position."""
    needle = content.strip()
    cut = rendered.rfind(needle) if needle else -1
    if cut < 0:
        raise ValueError("cut_after_content: the user content is not in the rendered template")
    return rendered[:cut + len(needle)]


def check_row(row: dict) -> None:
    """Raise ValueError when the engine cannot read ``row``. The backend calls this on
    the laptop before any remote call, and the engine calls it again.

    A row is ``{"id", "conversation"}`` and nothing else. The conversation starts with
    a system turn that ends with THINKING_SYSTEM, and it ends with a user turn that has
    text. The engine reads the three positions of every row, so a row has no flag that
    selects a position."""
    rid = row.get("id")
    extra = sorted(set(row) - {"id", "conversation"})
    if rid is None or extra:
        raise ValueError(f"row {rid!r}: a row has the keys id and conversation only, not {extra}")
    conv = row.get("conversation")
    if not isinstance(conv, list) or not conv:
        raise ValueError(f"row {rid!r}: the conversation is empty")
    first, last = conv[0], conv[-1]
    if first.get("role") != "system" or not str(first.get("content", "")).endswith(THINKING_SYSTEM):
        raise ValueError(f"row {rid!r}: the first turn must be a system turn that ends with "
                         f"{THINKING_SYSTEM!r} (design 0035 item 3)")
    if last.get("role") != "user" or not str(last.get("content", "")).strip():
        raise ValueError(f"row {rid!r}: the conversation must end with a user turn that has text")


def select_layers(n_layers: int, layer_stride: int) -> list[int]:
    """The 0-based decoder layers that `capture` stores: every ``layer_stride``-th layer,
    counted from 1, as in the study engine. Stride 2 on 80 layers gives 1, 3, ..., 79."""
    sel = [i for i in range(n_layers) if (i + 1) % max(1, int(layer_stride)) == 0]
    return sel or [n_layers - 1]


def pack(lengths: list[int], max_batch_tokens: int) -> list[list[int]]:
    """Row indices grouped so that ``rows x longest`` stays under the budget. This is
    `ReencodeEngine._pack` of the study engine."""
    order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
    batches: list[list[int]] = []
    cur: list[int] = []
    for i in order:
        longest = lengths[cur[0]] if cur else lengths[i]
        if cur and (len(cur) + 1) * longest > max_batch_tokens:
            batches.append(cur)
            cur = []
        cur.append(i)
    if cur:
        batches.append(cur)
    return batches


def render(tok, conv: list[dict], add_generation_prompt: bool) -> str:
    """The base tokenizer's Llama 3 chat template, as Hua et al. apply it. The row
    supplies the system text, including the "detailed thinking on" switch (`check_row`
    makes sure of it), so this passes no template switch of its own."""
    return tok.apply_chat_template(conv, add_generation_prompt=add_generation_prompt, tokenize=False)


def row_positions(ids_of, rendered_gen: str, rendered_plain: str, content: str) -> tuple:
    """The token ids of the generation prompt and the three read positions of one row.

    Returns ``(ids, plen, content_end, user_start)``:
      - ``plen - 1`` is the last token of the generation prompt ("last_prompt").
      - ``content_end`` is the last token of ``_cut_after_content(rendered_plain,
        content)``, the last token of the user's text ("content_end").
      - ``user_start .. content_end`` (inclusive) are the tokens of the user's text
        ("user_mean"). ``user_start`` is the token count of the text before it.

    ``ids_of`` maps a string to token ids with no special tokens added. The cut and the
    text before the content must each tokenize to a prefix of the generation prompt;
    otherwise a token merges across the boundary and the position is not the one named.
    That raises, because a silent fallback reads a different token."""
    ids = list(ids_of(rendered_gen))
    cut = _cut_after_content(rendered_plain, content)
    head = cut[:len(cut) - len(content.strip())]
    cut_ids, head_ids = list(ids_of(cut)), list(ids_of(head))
    if ids[:len(cut_ids)] != cut_ids:
        raise ValueError("the cut after the user content is not a token prefix of the generation prompt")
    if cut_ids[:len(head_ids)] != head_ids:
        raise ValueError("the text before the user content is not a token prefix of the cut")
    content_end, user_start = len(cut_ids) - 1, len(head_ids)
    if not 0 <= user_start <= content_end < len(ids):
        raise ValueError(f"empty user span: user_start {user_start}, content_end {content_end}")
    return ids, len(ids), content_end, user_start


def capture_hooked(forward, layers, seqs: list[tuple], sel: list[int], pad_id: int,
                   max_batch_tokens: int, device, out_dtype: str = "float16") -> list[dict]:
    """Batched forwards with a hook on each selected decoder layer. Per sequence it
    returns ``{"last_prompt", "content_end", "user_mean"}``, each an (L, d) array in
    ``out_dtype``, one row per layer of ``sel``. It also returns ``"max_abs"``, the
    largest absolute value over the three vectors and every layer in fp32 before the
    cast, ``"max_abs_layers"``, the same value per layer of ``sel``, and
    ``"n_nonfinite"``, the count of (layer, position) vectors that hold a
    value that is not finite after the cast. A fp32 value above FP16_MAX becomes inf in
    fp16, so ``n_nonfinite`` also counts a fp16 overflow.

    ``seqs`` holds ``(ids, plen, content_end, user_start)`` per row (see `row_positions`).
    ``forward(input_ids)`` runs the decoder stack. The hook reads the layer output, which
    is the residual stream after that layer. The engine reads in fp32 and casts to fp16
    as the study engine does; the CPU smoke reads in fp32 to compare at 1e-5."""
    import numpy as np
    import torch

    keys = ("last_prompt", "content_end", "user_mean")
    results: list = [None] * len(seqs)
    for batch in pack([len(s[0]) for s in seqs], int(max_batch_tokens)):
        B, T = len(batch), max(len(seqs[i][0]) for i in batch)
        # Right padding with NO attention mask, on purpose, as in the study engine: a
        # masked pad query row turns to NaN inside attention and leaks into every real
        # position. Every DeciLM block is causal or acts per token: the attention reads
        # keys at or before the query, and the no-op, linear-replaced, norm and MLP parts
        # act on each position alone. So a pad after a position cannot change it.
        input_ids = torch.full((B, T), int(pad_id), dtype=torch.long)
        last_idx = torch.zeros(B, dtype=torch.long)
        end_idx = torch.zeros(B, dtype=torch.long)
        user_mask = torch.zeros((B, T), dtype=torch.float32)
        for b, i in enumerate(batch):
            ids, plen, content_end, user_start = seqs[i]
            input_ids[b, :len(ids)] = torch.tensor(ids)
            last_idx[b], end_idx[b] = plen - 1, content_end
            user_mask[b, user_start:content_end + 1] = 1.0
        input_ids, last_idx, end_idx = input_ids.to(device), last_idx.to(device), end_idx.to(device)
        user_mask = user_mask.to(device)
        weights = (user_mask / user_mask.sum(1, keepdim=True).clamp(min=1.0))[:, None, :]
        rows = torch.arange(B, device=device)
        grabbed: dict[int, tuple] = {}
        handles = []
        for li in sel:
            # The hook runs inside this iteration's forward. The defaults bind this
            # batch's index tensors to it all the same.
            def _hook(_m, _i, out, _li=li, _rows=rows, _last=last_idx, _end=end_idx, _w=weights, _g=grabbed):
                h = (out[0] if isinstance(out, tuple) else out).float()
                vecs = (h[_rows, _last], h[_rows, _end], torch.bmm(_w, h)[:, 0])
                # The fp32 magnitude before the cast, per row. NaN stays NaN.
                amax = torch.stack([v.abs().amax(-1) for v in vecs]).amax(0)
                _g[_li] = (tuple(v.cpu().numpy().astype(out_dtype) for v in vecs), amax.cpu().numpy())
            handles.append(layers[li].register_forward_hook(_hook))
        try:
            with torch.no_grad():
                forward(input_ids)
        finally:
            for h in handles:
                h.remove()
        for b, i in enumerate(batch):
            row = {key: np.stack([grabbed[li][0][k][b] for li in sel]) for k, key in enumerate(keys)}
            row["max_abs_layers"] = [float(grabbed[li][1][b]) for li in sel]
            row["max_abs"] = float(np.max(row["max_abs_layers"]))
            row["n_nonfinite"] = int(sum(int((~np.isfinite(row[key])).any(-1).sum()) for key in keys))
            results[i] = row
    return results


def adapter_shapes(adapter_file: str) -> dict[str, list[int]]:
    """The tensor names and shapes of a local adapter file, read from its header."""
    from safetensors import safe_open

    with safe_open(adapter_file, framework="pt") as f:
        return {k: list(f.get_slice(k).get_shape()) for k in f.keys()}


def check_skeleton(base_dir: str, adapter_dir: str, adapter_header: dict[str, list[int]]) -> dict:
    """Build the base with no weights from the config and the remote code in
    ``base_dir``, and inject the adapter of ``adapter_dir`` into it. Raise when the
    weight names differ from the shard index, or when the LoRA names or shapes differ
    from ``adapter_header`` (name to shape). This needs only the small files, so the
    engine runs it before the shard download, and a mismatch costs seconds.
    Returns the counts of layers, attention layers and LoRA modules."""
    import json

    import torch
    from accelerate import init_empty_weights
    from peft import LoraConfig, get_peft_model
    from peft.tuners.lora import LoraLayer
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained(base_dir, trust_remote_code=True)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True, torch_dtype=torch.bfloat16)
    with open(os.path.join(base_dir, "model.safetensors.index.json")) as f:
        index = set(json.load(f)["weight_map"])
    have = set(model.state_dict())
    if have != index:
        raise RuntimeError(f"the model weights differ from the shard index: missing {sorted(index - have)[:4]}, "
                           f"unexpected {sorted(have - index)[:4]}")
    layers = model.model.layers
    counts = {"n_layers": len(layers), "n_attention_layers": sum(hasattr(layer, "self_attn") for layer in layers)}
    lcfg = LoraConfig.from_pretrained(adapter_dir)
    lcfg.lora_dropout = 0.0
    pm = get_peft_model(model, lcfg, low_cpu_mem_usage=True)
    want: dict[str, list[int]] = {}
    for n, m in pm.named_modules():
        if isinstance(m, LoraLayer):
            base = m.get_base_layer()
            want[f"{n}.lora_A.weight"] = [lcfg.r, base.in_features]
            want[f"{n}.lora_B.weight"] = [base.out_features, lcfg.r]
    if set(want) != set(adapter_header):
        raise RuntimeError(f"peft injects {len(want) // 2} LoRA modules and the file names "
                           f"{len(adapter_header) // 2}, e.g. {sorted(set(want) ^ set(adapter_header))[:4]}")
    bad = [k for k in want if list(adapter_header[k]) != want[k]]
    if bad:
        raise RuntimeError(f"{len(bad)} LoRA shapes differ from the file, e.g. {bad[:4]}")
    counts["n_lora_modules"] = len(want) // 2
    return counts


def load_base(path: str, dtype, device_map):
    """The DeciLM base from a local snapshot, with its remote code. A missing or
    mismatched weight raises: transformers would otherwise leave it at random init.

    Eager attention, set explicitly: the DeciLM code offers only eager and flash
    attention 2, and the image does not install flash attention."""
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained(path, trust_remote_code=True)
    model, info = AutoModelForCausalLM.from_pretrained(
        path, config=cfg, torch_dtype=dtype, device_map=device_map, trust_remote_code=True,
        attn_implementation="eager", output_loading_info=True)
    bad = {k: v for k, v in info.items() if k in ("missing_keys", "mismatched_keys") and v}
    if bad:
        raise RuntimeError(f"the base checkpoint did not load cleanly: {bad}")
    model.eval()
    return model


def attach_adapter(model, adapter_dir: str):
    """The LoRA adapter on the base, for inference, NOT merged. `disable_adapter()` on
    the result then gives the base model with the same weights in memory."""
    from peft import PeftModel

    peft_model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=False)
    peft_model.eval()
    return peft_model


def lora_modules(peft_model) -> dict:
    """The modules that carry a LoRA layer, by name."""
    from peft.tuners.lora import LoraLayer

    return {n: m for n, m in peft_model.named_modules() if isinstance(m, LoraLayer)}


def adapter_state(peft_model) -> dict:
    """The adapter state read from the LoRA layers, not assumed: ``merged`` is True
    when any layer holds a merged adapter, and ``enabled`` is True when no layer has
    its adapters turned off."""
    mods = list(lora_modules(peft_model).values())
    return {"merged": any(bool(m.merged) for m in mods),
            "enabled": bool(mods) and not any(bool(m.disable_adapters) for m in mods)}


def observed_arm(peft_model):
    """The arm that the LoRA layers give now: True when every layer has its adapter on,
    False when every layer has it off, and None for a mixed state, a merged adapter, or
    no LoRA layer."""
    mods = list(lora_modules(peft_model).values())
    if not mods or any(bool(m.merged) for m in mods):
        return None
    off = [bool(m.disable_adapters) for m in mods]
    if not any(off):
        return True
    if all(off):
        return False
    return None


def check_adapter(peft_model, adapter_file: str) -> int:
    """Prove that the adapter file landed: the set of modules with a LoRA layer equals
    the set the file names, every loaded B matrix is nonzero (peft initialises B to
    zero, so a zero B is a weight that did not load), and every loaded A and B matrix
    equals its tensor in the file, value for value (peft initialises A at random, so
    only this comparison proves that A loaded). Returns the module count."""
    import torch
    from safetensors import safe_open

    injected = lora_modules(peft_model)
    with safe_open(adapter_file, framework="pt") as f:
        keys = [k for k in f.keys() if ".lora_" in k]
        named = {k.rsplit(".lora_", 1)[0] for k in keys}
        if set(injected) != named:
            raise RuntimeError(f"adapter modules differ from the file: {len(injected)} injected, "
                               f"{len(named)} in the file, e.g. {sorted(set(injected) ^ named)[:4]}")
        zero = [n for n, m in injected.items()
                if any(float(b.weight.abs().max()) == 0.0 for b in m.lora_B.values())]
        if zero:
            raise RuntimeError(f"{len(zero)} adapter B matrices are zero after load, e.g. {zero[:4]}")
        differ = []
        for k in keys:
            module, rest = k.rsplit(".lora_", 1)
            part = getattr(injected[module], f"lora_{rest.split('.')[0]}")
            if list(part.keys()) != ["default"]:
                raise RuntimeError(f"{module}: adapters {list(part.keys())}, expected only 'default'")
            loaded = part["default"].weight.detach()
            if not torch.equal(f.get_tensor(k).to(loaded.dtype), loaded.cpu()):
                differ.append(k)
        if differ:
            raise RuntimeError(f"{len(differ)} loaded adapter tensors differ from the file, e.g. {differ[:4]}")
    return len(injected)


def adapter_dtypes(peft_model) -> list[str]:
    """The sorted set of dtypes of the LoRA parameters."""
    return sorted({str(p.dtype) for n, p in peft_model.named_parameters() if ".lora_" in n})


def lora_scalings(peft_model) -> list[float]:
    """The sorted set of LoRA scaling values over every module and adapter."""
    return sorted({float(s) for m in lora_modules(peft_model).values() for s in m.scaling.values()})


def host_usage() -> dict:
    """The peak host memory of the container and its CPU count, for the spend
    reconciliation. ``peak_rss_gib`` is this process (Linux reports ru_maxrss in KiB).
    ``cgroup_peak_gib`` is the container's cgroup peak, which counts the page cache,
    when the file exists."""
    import resource

    out: dict = {"peak_rss_gib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 / 2**30, 2),
                 "n_cpus": os.cpu_count()}
    for path in ("/sys/fs/cgroup/memory.peak", "/sys/fs/cgroup/memory/memory.max_usage_in_bytes"):
        try:
            with open(path) as f:
                out["cgroup_peak_gib"] = round(int(f.read().split()[0]) / 2**30, 2)
            break
        except (OSError, ValueError, IndexError):
            continue
    return out


def decoder_layers(peft_model) -> list:
    """The decoder layers under the peft wrapper: `base_model.model.model.layers`
    (PeftModelForCausalLM -> LoraModel -> DeciLMForCausalLM -> DeciLMModel -> layers).
    Each layer returns a tuple whose first item is the residual stream after the layer."""
    return list(peft_model.get_base_model().model.layers)


def sha256_file(path: str) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def limits() -> dict:
    """The container limits of this file, for `config()` to report."""
    return {"scaledown_s": SCALEDOWN_S, "timeout_s": TIMEOUT_S, "startup_timeout_s": STARTUP_TIMEOUT_S,
            "retries": RETRIES, "max_containers": MAX_CONTAINERS}


@app.cls(
    gpu=GPU,
    image=_image,
    secrets=_hf_secret,
    scaledown_window=SCALEDOWN_S,
    timeout=TIMEOUT_S,
    startup_timeout=STARTUP_TIMEOUT_S,
    retries=RETRIES,
    max_containers=MAX_CONTAINERS,
)
class OrganismEngine:
    """The organism control of design 0035: the Nemotron base with the Wood Labs adapter
    attached, read with the adapter on and off in one container. The revisions are the
    module pins and not modal.parameter fields, so max_containers=1 holds for the class."""

    @modal.enter()
    def load(self) -> None:
        import shutil
        import time as _time

        import peft
        import torch
        import transformers
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer

        self.torch = torch
        os.makedirs(_CACHE_DIR, exist_ok=True)
        free_gb = shutil.disk_usage(_CACHE_DIR).free / 1e9
        if free_gb < MIN_FREE_DISK_GB:
            raise RuntimeError(f"{free_gb:.0f} GB free on local disk, {MIN_FREE_DISK_GB} GB needed; "
                               f"the container disk is smaller than the Modal default")
        t0 = _time.time()
        # Stage 1, about 2.4 GB: the adapter, its hash, the small base files and the
        # skeleton check. Every check that needs no shard runs here, before the 100 GB.
        adapter_dir = snapshot_download(ADAPTER_ID, revision=ADAPTER_REVISION,
                                        allow_patterns=["adapter_config.json", ADAPTER_FILE])
        adapter_file = os.path.join(adapter_dir, ADAPTER_FILE)
        self.adapter_sha256 = sha256_file(adapter_file)
        if self.adapter_sha256 != ADAPTER_SHA256:
            raise RuntimeError(f"adapter sha256 {self.adapter_sha256} is not the pinned {ADAPTER_SHA256}")
        small_dir = snapshot_download(BASE_ID, revision=BASE_REVISION, allow_patterns=_BASE_SMALL_FILES)
        self.skeleton = check_skeleton(small_dir, adapter_dir, adapter_shapes(adapter_file))
        t_stage1 = _time.time() - t0
        # Stage 2: the shards.
        base_dir = snapshot_download(BASE_ID, revision=BASE_REVISION, allow_patterns=_BASE_FILES)
        t_download = _time.time() - t0
        # A snapshot sits under its commit hash, so the directory name is the build loaded.
        self.loaded_revision = os.path.basename(base_dir.rstrip("/"))
        self.loaded_adapter_revision = os.path.basename(adapter_dir.rstrip("/"))
        t1 = _time.time()
        self.tok = AutoTokenizer.from_pretrained(base_dir)
        base = load_base(base_dir, torch.bfloat16, "cuda")
        self.model = attach_adapter(base, adapter_dir)
        self.n_lora_modules = check_adapter(self.model, adapter_file)
        self.layers = decoder_layers(self.model)
        self.inner = self.model.get_base_model().model
        self.n_attention_layers = sum(hasattr(layer, "self_attn") for layer in self.layers)
        t_load = _time.time() - t1
        # One small forward per arm, while the start is paid. The logits must be finite,
        # and the two arms must differ: equal logits mean the adapter does not act.
        with torch.no_grad():
            warm = self.tok("warm up", return_tensors="pt", add_special_tokens=False)["input_ids"].to("cuda")
            on = self.model(input_ids=warm, use_cache=False).logits.float()
            with self.model.disable_adapter():
                off = self.model(input_ids=warm, use_cache=False).logits.float()
        if not (bool(torch.isfinite(on).all()) and bool(torch.isfinite(off).all())):
            raise RuntimeError("non-finite logits on the warm-up forward; the load is unusable")
        self.warm_logit_delta = float((on - off).abs().max())
        if self.warm_logit_delta == 0.0:
            raise RuntimeError("the adapter does not change the warm-up logits; it is not active")
        self.load_seconds = {"stage1": round(t_stage1, 1), "download": round(t_download, 1),
                             "load": round(t_load, 1)}
        print(f"[organism] transformers {transformers.__version__} | peft {peft.__version__} | torch "
              f"{torch.__version__} | gpu {GPU} ({torch.cuda.get_device_name(0)}) | {BASE_ID}@{self.loaded_revision} "
              f"| {ADAPTER_ID}@{self.loaded_adapter_revision} | {self.n_lora_modules} LoRA modules | "
              f"{len(self.layers)} layers, {self.n_attention_layers} with attention | download {t_download:.0f}s, "
              f"load {t_load:.0f}s | warm-up logit delta {self.warm_logit_delta:.3f}")

    @modal.method()
    def config(self) -> dict:
        """The engine's LIVE build, for the preflight to compare against the pins."""
        import accelerate
        import peft
        import transformers

        torch = self.torch
        base = self.model.get_base_model()
        state = adapter_state(self.model)
        return {"engine": "organism", "app": APP_NAME, "gpu": GPU, "gpu_name": torch.cuda.get_device_name(0),
                "transformers_version": transformers.__version__, "peft_version": peft.__version__,
                "torch_version": torch.__version__, "accelerate_version": accelerate.__version__,
                "model": BASE_ID, "revision": self.loaded_revision,
                "adapter": ADAPTER_ID, "adapter_revision": self.loaded_adapter_revision,
                "adapter_loaded": self.n_lora_modules > 0, "adapter_merged": state["merged"],
                "adapter_enabled": state["enabled"], "n_lora_modules": self.n_lora_modules,
                "adapter_sha256": self.adapter_sha256, "adapter_dtypes": adapter_dtypes(self.model),
                "lora_scaling": lora_scalings(self.model), "host": host_usage(), "dtype": str(base.dtype),
                "attn_implementation": str(base.config._attn_implementation),
                "n_layers": len(self.layers), "n_attention_layers": self.n_attention_layers,
                "skeleton": dict(self.skeleton), **limits(),
                "load_seconds": dict(self.load_seconds), "warm_logit_delta": self.warm_logit_delta}

    def _ids(self, text: str) -> list[int]:
        return list(self.tok(text, add_special_tokens=False)["input_ids"]) if text else []

    def _pad_id(self) -> int:
        if self.tok.pad_token_id is not None:
            return int(self.tok.pad_token_id)
        return int(self.tok.eos_token_id or 0)

    @modal.method()
    def capture(self, rows: list[dict], layer_stride: int, max_batch_tokens: int, adapter: bool) -> list[dict]:
        """Per row: ``{"id", "layers", "adapter", "last_prompt", "content_end", "user_mean":
        (L, d) fp16 each, "max_abs", "max_abs_layers", "n_nonfinite", "n_prompt_tokens",
        "content_end_index", "n_user_tokens", "user_span"}``. ``user_span`` is
        ``[user_start, content_end]``, as the study engine returns it.

        A row passes `check_row`. All three positions come from one forward over the
        generation prompt (see `row_positions`). ``adapter`` False reads the base model:
        the same forward inside `disable_adapter()`. The forward runs the decoder stack
        only, because no position here needs the logits. Inside the arm, the arm is read
        from the LoRA layers (`observed_arm`). The call raises when it is not the
        requested arm, and each row's ``adapter`` is the observed arm."""
        import contextlib

        if not isinstance(adapter, bool):
            raise TypeError(f"adapter must be a bool, not {type(adapter).__name__}")
        sel = select_layers(len(self.layers), layer_stride)
        seqs: list[tuple] = []
        for r in rows:
            check_row(r)
            conv = r["conversation"]
            try:
                seqs.append(row_positions(self._ids, render(self.tok, conv, True),
                                          render(self.tok, conv, False), conv[-1]["content"]))
            except ValueError as e:
                raise ValueError(f"row {r.get('id')!r}: {e}") from e
        arm = contextlib.nullcontext() if adapter else self.model.disable_adapter()
        with arm:
            observed = observed_arm(self.model)
            if observed is not adapter:
                raise RuntimeError(f"the LoRA layers give arm {observed!r} (None is a mixed or merged state); "
                                   f"the call asked for adapter {adapter}")
            vecs = capture_hooked(lambda x: self.inner(input_ids=x, use_cache=False), self.layers, seqs,
                                  sel, self._pad_id(), int(max_batch_tokens), "cuda")
        return [{"id": r["id"], "layers": list(sel), "adapter": observed, **v,
                 "n_prompt_tokens": s[1], "content_end_index": s[2], "n_user_tokens": s[2] - s[3] + 1,
                 "user_span": [s[3], s[2]]}
                for r, s, v in zip(rows, seqs, vecs, strict=True)]

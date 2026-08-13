"""Load AMPLIFY (e.g. SaAMPLIFY-350M) as an OFS encoder, portably.

AMPLIFY's ``trust_remote_code`` modeling file imports ``xformers`` at module load
(``from xformers.ops import SwiGLU, memory_efficient_attention``) even though its
CPU/XPU path never *calls* ``memory_efficient_attention`` -- it uses
``torch.nn.functional.scaled_dot_product_attention`` when the tensors are not CUDA.
On Aurora XPU and on laptops, xformers is usually absent, so the import alone breaks
loading.

Two portability fixes are applied here:

1. **xformers shim** -- if xformers is not importable we inject a tiny stand-in into
   ``sys.modules`` *before* the modeling file loads. It provides a pure-torch fused
   ``SwiGLU`` (submodules ``w12`` / ``w3`` so the checkpoint's
   ``ffn.w12.weight`` / ``ffn.w3.weight`` load unchanged) and an SDPA-based
   ``memory_efficient_attention`` fallback.
2. **XPU SDPA patch** -- on XPU the IPEX build has no fused attention kernel and
   segfaults past ~512 tokens, so we monkeypatch AMPLIFY's module-global
   ``scaled_dot_product_attention`` to a manual matmul+softmax (same fix as
   ``mini-embed-filip/src/encoders.py``). CUDA/CPU are left alone.

AMPLIFY vocab: pad=0, unk=1, mask=2, bos=3, eos=4, ``|``=5, then the amino acids.
"""

from __future__ import annotations

import sys
import types
from typing import List, Optional

import torch
import torch.nn.functional as F

from ..adapters import ModuleAdapter
from ..alphabet import Alphabet
from ..dist import pick_device
from . import LoadedEncoder


# --------------------------------------------------------------- xformers shim
class _SwiGLUShim(torch.nn.Module):
    """Pure-torch fused SwiGLU matching xformers' packed-weight layout.

    ``[x1, x2] = w12(x).chunk(2, -1); out = w3(silu(x1) * x2)``. Submodule names
    ``w12`` and ``w3`` match the AMPLIFY checkpoint keys exactly.
    """

    def __init__(self, in_features, hidden_features, out_features, bias=True, *args, **kwargs):
        super().__init__()
        self.w12 = torch.nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = torch.nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(x1) * x2)


def _memory_efficient_attention_shim(query, key, value, attn_bias=None, p=0.0, *args, **kwargs):
    # Inputs are (B, M, H, K); SDPA wants (B, H, M, K). Only used on the CUDA path,
    # which AMPLIFY does not take on XPU/CPU -- provided for import completeness.
    q = query.transpose(1, 2)
    k = key.transpose(1, 2)
    v = value.transpose(1, 2)
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias, dropout_p=p)
    return out.transpose(1, 2)


def _install_xformers_shim() -> bool:
    """Inject a fake ``xformers`` if the real one is missing. Returns True if shimmed."""
    try:
        import xformers  # noqa: F401
        import xformers.ops  # noqa: F401

        return False
    except Exception:
        pass
    xformers = types.ModuleType("xformers")
    ops = types.ModuleType("xformers.ops")
    ops.SwiGLU = _SwiGLUShim
    ops.memory_efficient_attention = _memory_efficient_attention_shim
    xformers.ops = ops
    sys.modules["xformers"] = xformers
    sys.modules["xformers.ops"] = ops
    return True


def _patch_amplify_sdpa_for_xpu(model) -> None:
    """Replace the modeling module's SDPA with a manual kernel (XPU-safe)."""
    mod = sys.modules.get(type(model).__module__)
    if mod is None or not hasattr(mod, "scaled_dot_product_attention"):
        return

    def _manual_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, **kwargs):
        scale = query.shape[-1] ** -0.5
        scores = torch.matmul(query, key.transpose(-2, -1)) * scale
        if attn_mask is not None:
            scores = scores + attn_mask
        attn = torch.softmax(scores, dim=-1)
        if dropout_p and dropout_p > 0.0:
            attn = F.dropout(attn, p=dropout_p)
        return torch.matmul(attn, value)

    mod.scaled_dot_product_attention = _manual_sdpa


# ------------------------------------------------------------------- adapters
def _amplify_hidden_fn(model, input_ids, padding_mask):
    attn = _additive_mask(padding_mask, model)
    out = model(input_ids, attention_mask=attn, output_hidden_states=True)
    return out.hidden_states[-1]


def _amplify_logits_fn(model, input_ids, padding_mask):
    attn = _additive_mask(padding_mask, model)
    out = model(input_ids, attention_mask=attn)
    return out.logits


def _additive_mask(padding_mask, model):
    """AMPLIFY expects an *additive* attention mask (0 keep, -inf pad), or None."""
    if padding_mask is None:
        return None
    neg = torch.finfo(torch.float32).min
    add = torch.where(
        padding_mask.to(torch.bool),
        torch.zeros(1, dtype=torch.float32, device=padding_mask.device),
        torch.full((1,), neg, dtype=torch.float32, device=padding_mask.device),
    )
    return add


def load_amplify(
    path: str,
    *,
    device="auto",
    local_rank: int = 0,
    dtype: Optional[torch.dtype] = None,
    freeze: bool = True,
) -> LoadedEncoder:
    _install_xformers_shim()
    from transformers import AutoModel, AutoTokenizer

    dev = pick_device(device, local_rank) if isinstance(device, str) else torch.device(device)
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    model = AutoModel.from_pretrained(path, trust_remote_code=True)
    if dtype is not None:
        model = model.to(dtype)
    model = model.eval().to(dev)
    if freeze:
        for p in model.parameters():
            p.requires_grad_(False)
    if dev.type == "xpu":
        _patch_amplify_sdpa_for_xpu(model)

    alphabet = Alphabet.from_amplify_tokenizer(tokenizer)

    def encode_fn(seq: str) -> List[int]:
        return tokenizer(seq, add_special_tokens=True)["input_ids"]

    adapter = ModuleAdapter(
        model,
        alphabet,
        hidden_fn=_amplify_hidden_fn,
        logits_fn=_amplify_logits_fn,
        hidden_size=int(model.config.hidden_size),
    )
    return LoadedEncoder(
        adapter=adapter,
        alphabet=alphabet,
        encode_fn=encode_fn,
        tokenizer=tokenizer,
        model=model,
    )

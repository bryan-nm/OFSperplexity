"""Load ESMC (ESM Cambrian) as an OFS encoder.

ESMC is a standard HuggingFace ``*ForMaskedLM``; scoring works out of the box:
``AutoModelForMaskedLM.from_pretrained`` returns ``.logits`` and, with
``output_hidden_states=True``, per-layer ``.hidden_states`` whose last entry is the
960-d residue embedding OFS projects from.

Requires ``transformers`` new enough to recognise ``model_type: "esmc"`` (the
ESMC-300M card was written for transformers 4.57.6). The 20 amino acids live at
token ids 4-23; mask=32, pad=1, cls=0, eos=2.
"""

from __future__ import annotations

from typing import List, Optional

import torch

from ..adapters import HFMaskedLMAdapter
from ..alphabet import Alphabet
from ..dist import pick_device
from . import LoadedEncoder


def load_encode_fn(tokenizer):
    def encode_fn(seq: str) -> List[int]:
        return tokenizer(seq, add_special_tokens=True)["input_ids"]

    return encode_fn


def load_esmc(
    path: str,
    *,
    device="auto",
    local_rank: int = 0,
    dtype: Optional[torch.dtype] = None,
    hidden_layer: int = -1,
    freeze: bool = True,
) -> LoadedEncoder:
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    dev = pick_device(device, local_rank) if isinstance(device, str) else torch.device(device)
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForMaskedLM.from_pretrained(path)
    if dtype is not None:
        model = model.to(dtype)
    model = model.eval().to(dev)
    if freeze:
        for p in model.parameters():
            p.requires_grad_(False)
    alphabet = Alphabet.from_hf_tokenizer(tokenizer)
    adapter = HFMaskedLMAdapter(model, alphabet, hidden_layer=hidden_layer)
    return LoadedEncoder(
        adapter=adapter,
        alphabet=alphabet,
        encode_fn=load_encode_fn(tokenizer),
        tokenizer=tokenizer,
        model=model,
    )

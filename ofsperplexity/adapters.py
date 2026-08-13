"""Encoder adapters: the seam between OFS scoring and *any* masked language model.

The scoring library never imports ``transformers`` and never assumes a particular
model class. It only needs an object that can, given token ids, return:

* ``forward_hidden`` -> last-layer per-residue embeddings ``(B, L, D)``  (for OFS)
* ``forward_logits`` -> full-vocab MLM logits ``(B, L, V)``              (for exact)

Plus a small amount of metadata (``hidden_size``, ``mask_token_id`` via the
:class:`~ofsperplexity.alphabet.Alphabet`, device/dtype).

Three ready-made adapters are provided:

* :class:`HFMaskedLMAdapter` -- wraps any ``transformers`` ``*ForMaskedLM`` whose
  forward accepts ``output_hidden_states=True`` and returns ``.logits`` /
  ``.hidden_states`` (ESMC works out of the box).
* :class:`ModuleAdapter` -- wraps a plain ``nn.Module`` plus two extractor
  callables. This is the general escape hatch (AMPLIFY uses it).
* :class:`CallableAdapter` -- wraps bare functions with no ``nn.Module`` at all.

To wire OFS into a model **you are training**, write a tiny adapter (or reuse
:class:`ModuleAdapter`) around the *frozen, eval-mode* encoder you want to
evaluate against. See ``docs/INTEGRATION.md`` and ``examples/``.

Padding & attention-mask conventions are owned by the adapter: callers always pass
a boolean *keep-mask* ``padding_mask`` (True = real token). The HF adapter forwards
it as a standard 1/0 attention mask; a custom adapter (e.g. AMPLIFY's additive
mask) converts as needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Protocol, Tuple, runtime_checkable

import torch

from .alphabet import Alphabet


@dataclass
class EncoderOutput:
    """Bundle returned by :meth:`EncoderAdapter.forward`."""

    hidden: Optional[torch.Tensor] = None  # (B, L, D) last-layer embeddings
    logits: Optional[torch.Tensor] = None  # (B, L, V) full-vocab MLM logits


@runtime_checkable
class EncoderAdapter(Protocol):
    """Structural interface every adapter satisfies."""

    alphabet: Alphabet

    @property
    def hidden_size(self) -> int: ...

    @property
    def device(self) -> torch.device: ...

    def forward(
        self,
        input_ids: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        *,
        need_hidden: bool = False,
        need_logits: bool = False,
    ) -> EncoderOutput: ...


class BaseAdapter:
    """Shared conveniences; subclasses implement :meth:`forward`."""

    alphabet: Alphabet

    @property
    def hidden_size(self) -> int:  # pragma: no cover - overridden
        raise NotImplementedError

    @property
    def device(self) -> torch.device:  # pragma: no cover - overridden
        raise NotImplementedError

    def forward_hidden(
        self, input_ids: torch.Tensor, padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        out = self.forward(input_ids, padding_mask, need_hidden=True)
        assert out.hidden is not None
        return out.hidden

    def forward_logits(
        self, input_ids: torch.Tensor, padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        out = self.forward(input_ids, padding_mask, need_logits=True)
        assert out.logits is not None
        return out.logits


class HFMaskedLMAdapter(BaseAdapter):
    """Adapter for a HuggingFace ``*ForMaskedLM`` (ESMC, ESM-2, etc.).

    Parameters
    ----------
    model: the loaded HF model (already ``.eval()`` and ``.to(device)``).
    alphabet: an :class:`Alphabet` built from the matching tokenizer.
    hidden_layer: which hidden state to treat as "the embedding". ``-1`` is the
        final layer (the paper uses the last-layer representation). Note HF's
        ``hidden_states`` tuple includes the input embedding at index 0, so
        ``-1`` is the last transformer block's output.
    """

    def __init__(self, model, alphabet: Alphabet, hidden_layer: int = -1):
        self.model = model
        self.alphabet = alphabet
        self.hidden_layer = hidden_layer

    @property
    def hidden_size(self) -> int:
        cfg = self.model.config
        return int(getattr(cfg, "hidden_size", None) or getattr(cfg, "d_model"))

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        *,
        need_hidden: bool = False,
        need_logits: bool = False,
    ) -> EncoderOutput:
        attn = None if padding_mask is None else padding_mask.to(input_ids.device).long()
        out = self.model(
            input_ids=input_ids,
            attention_mask=attn,
            output_hidden_states=need_hidden,
        )
        hidden = out.hidden_states[self.hidden_layer] if need_hidden else None
        logits = out.logits if need_logits else None
        return EncoderOutput(hidden=hidden, logits=logits)


class ModuleAdapter(BaseAdapter):
    """Adapter around an arbitrary ``nn.Module`` with user-supplied extractors.

    ``hidden_fn(module, input_ids, padding_mask) -> (B, L, D)`` and
    ``logits_fn(module, input_ids, padding_mask) -> (B, L, V)``. Either may be
    ``None`` if that capability is not needed (OFS-only needs ``hidden_fn``;
    exact-only needs ``logits_fn``).

    This is the recommended way to attach OFS to a model you are training: pass
    the frozen encoder and a closure that returns its last hidden state.
    """

    def __init__(
        self,
        module,
        alphabet: Alphabet,
        *,
        hidden_fn: Optional[Callable] = None,
        logits_fn: Optional[Callable] = None,
        hidden_size: Optional[int] = None,
        no_grad: bool = True,
    ):
        self.module = module
        self.alphabet = alphabet
        self._hidden_fn = hidden_fn
        self._logits_fn = logits_fn
        self._hidden_size = hidden_size
        self._no_grad = no_grad

    @property
    def hidden_size(self) -> int:
        if self._hidden_size is not None:
            return self._hidden_size
        cfg = getattr(self.module, "config", None)
        if cfg is not None:
            hs = getattr(cfg, "hidden_size", None) or getattr(cfg, "d_model", None)
            if hs is not None:
                return int(hs)
        raise ValueError("hidden_size unknown; pass hidden_size=... to ModuleAdapter")

    @property
    def device(self) -> torch.device:
        return next(self.module.parameters()).device

    def forward(
        self,
        input_ids: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        *,
        need_hidden: bool = False,
        need_logits: bool = False,
    ) -> EncoderOutput:
        ctx = torch.no_grad() if self._no_grad else _nullcontext()
        with ctx:
            hidden = None
            logits = None
            if need_hidden:
                if self._hidden_fn is None:
                    raise RuntimeError("ModuleAdapter has no hidden_fn")
                hidden = self._hidden_fn(self.module, input_ids, padding_mask)
            if need_logits:
                if self._logits_fn is None:
                    raise RuntimeError("ModuleAdapter has no logits_fn")
                logits = self._logits_fn(self.module, input_ids, padding_mask)
        return EncoderOutput(hidden=hidden, logits=logits)


class CallableAdapter(BaseAdapter):
    """Adapter with no ``nn.Module`` -- just callables and explicit metadata.

    ``hidden_fn(input_ids, padding_mask) -> (B, L, D)`` and/or
    ``logits_fn(input_ids, padding_mask) -> (B, L, V)``.
    """

    def __init__(
        self,
        alphabet: Alphabet,
        *,
        hidden_size: int,
        device: torch.device,
        hidden_fn: Optional[Callable] = None,
        logits_fn: Optional[Callable] = None,
    ):
        self.alphabet = alphabet
        self._hidden_size = int(hidden_size)
        self._device = torch.device(device)
        self._hidden_fn = hidden_fn
        self._logits_fn = logits_fn

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    @property
    def device(self) -> torch.device:
        return self._device

    def forward(
        self,
        input_ids: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        *,
        need_hidden: bool = False,
        need_logits: bool = False,
    ) -> EncoderOutput:
        hidden = self._hidden_fn(input_ids, padding_mask) if need_hidden else None
        logits = self._logits_fn(input_ids, padding_mask) if need_logits else None
        return EncoderOutput(hidden=hidden, logits=logits)


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False

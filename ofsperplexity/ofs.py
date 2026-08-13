"""One Fell Swoop scoring: pseudo-perplexity in a single forward pass.

Given an encoder adapter and a trained :class:`~ofsperplexity.head.OFSHead`, this
runs the encoder **once** on the unmasked sequence, projects every residue's
embedding through the head to estimate its masked amino-acid profile, and scores.
No masking, no ``L`` forward passes -- the whole single-substitution landscape and
the pseudo-perplexity fall out of one pass (paper Sec. III).

Use this as the on-the-fly evaluation hook during generative-model training: it is
cheap enough to call on every eval batch, and it is independent of the model being
trained (it scores against a frozen encoder + its pretrained head).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from .adapters import EncoderAdapter
from .data import SequenceBatch
from .head import OFSHead
from .scoring import ScoreResult, pll_from_profiles


class OFSScorer:
    """Bundle an encoder adapter with a trained OFS head.

    Parameters
    ----------
    adapter: an :class:`EncoderAdapter` that can return last-layer hidden states.
    head: a trained :class:`OFSHead` whose ``input_dim`` matches ``adapter.hidden_size``.
    """

    def __init__(self, adapter: EncoderAdapter, head: OFSHead):
        if head.config.input_dim != adapter.hidden_size:
            raise ValueError(
                f"OFS head input_dim={head.config.input_dim} != encoder hidden_size="
                f"{adapter.hidden_size}. The head must be trained for this encoder."
            )
        self.adapter = adapter
        self.head = head.eval().to(adapter.device)

    @torch.no_grad()
    def profiles(self, batch: SequenceBatch) -> torch.Tensor:
        """Estimated masked profiles ``(B, L, 20)`` for a batch (single pass)."""
        batch = batch.to(self.adapter.device)
        hidden = self.adapter.forward_hidden(batch.input_ids, batch.padding_mask)
        return self.head.predict_profile(hidden)

    @torch.no_grad()
    def score(self, batch: SequenceBatch, *, return_per_position: bool = False) -> ScoreResult:
        """OFS pseudo-perplexity for every sequence in ``batch`` (single pass)."""
        batch = batch.to(self.adapter.device)
        hidden = self.adapter.forward_hidden(batch.input_ids, batch.padding_mask)
        profiles = self.head.predict_profile(hidden)  # (B, L, 20)
        return pll_from_profiles(
            profiles,
            batch.target_aa_index,
            batch.valid,
            return_per_position=return_per_position,
        )

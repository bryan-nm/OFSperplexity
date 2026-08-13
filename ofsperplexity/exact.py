"""Exact (one-at-a-time) pseudo-perplexity.

This is the reference computation the OFS paper approximates: for each scorable
position ``i`` we replace token ``i`` with ``<mask>``, run the encoder, and read the
softmax over the 20 amino-acid logits at position ``i`` (Eq. 4, Fig. 1c). It costs
``L`` forward passes for a length-``L`` sequence, so it is *slow* -- but it is the
ground truth, works for any masked LM with no extra training, and generates the
targets used to train an OFS head.

The masked replicas of one sequence are packed into forward-pass chunks capped by
``max_forward_tokens`` to keep the accelerator busy without exhausting memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from .adapters import EncoderAdapter
from .alphabet import Alphabet
from .data import SequenceBatch
from .scoring import ScoreResult, pll_from_aa_logits, profiles_from_aa_logits


@dataclass
class ExactProfiles:
    """Masked profiles for one sequence at its scorable positions.

    ``positions`` are indices into the token sequence (which include special
    tokens); ``profiles`` is ``(n_scored, 20)`` softmax distributions; ``targets``
    is ``(n_scored,)`` canonical aa indices of the true residues.
    """

    positions: torch.Tensor
    profiles: torch.Tensor
    targets: torch.Tensor


@torch.no_grad()
def exact_aa_logits_for_sequence(
    adapter: EncoderAdapter,
    input_ids_1d: torch.Tensor,
    valid_1d: torch.Tensor,
    *,
    max_forward_tokens: int = 16384,
) -> torch.Tensor:
    """Return ``(L, 20)`` amino-acid logits, one row per scorable position.

    Row ``i`` (for scorable ``i``) holds the length-20 logits obtained by masking
    position ``i`` and reading position ``i`` of the output. Non-scorable rows are
    left as zeros (they are never read by the scorer).
    """
    device = adapter.device
    alphabet: Alphabet = adapter.alphabet
    input_ids_1d = input_ids_1d.to(device)
    valid_1d = valid_1d.to(device)
    L = input_ids_1d.numel()
    aa_logits = torch.zeros((L, 20), dtype=torch.float32, device=device)

    positions = torch.nonzero(valid_1d, as_tuple=False).flatten()
    if positions.numel() == 0:
        return aa_logits

    # Chunk the masked replicas so each forward pass stays under the token budget.
    per_chunk = max(1, max_forward_tokens // max(L, 1))
    mask_id = alphabet.mask_token_id
    for start in range(0, positions.numel(), per_chunk):
        chunk_pos = positions[start : start + per_chunk]
        n = chunk_pos.numel()
        replicas = input_ids_1d.unsqueeze(0).repeat(n, 1)  # (n, L)
        replicas[torch.arange(n, device=device), chunk_pos] = mask_id
        out = adapter.forward(replicas, need_logits=True)
        logits = out.logits  # (n, L, V)
        # gather each replica's masked-position logits -> (n, V)
        masked_logits = logits[torch.arange(n, device=device), chunk_pos]
        aa = alphabet.gather_aa_logits(masked_logits)  # (n, 20)
        aa_logits[chunk_pos] = aa.float()
    return aa_logits


@torch.no_grad()
def exact_score(
    adapter: EncoderAdapter,
    batch: SequenceBatch,
    *,
    max_forward_tokens: int = 16384,
    return_per_position: bool = False,
) -> ScoreResult:
    """Exact pseudo-perplexity for every sequence in ``batch``."""
    device = adapter.device
    B, L = batch.input_ids.shape
    aa_logits = torch.zeros((B, L, 20), dtype=torch.float32, device=device)
    for b in range(B):
        # Slice to the true (unpadded) length so real positions never attend to pad
        # tokens -- the exact scorer masks one position at a time and passes no
        # padding mask, so trailing pad would otherwise contaminate the context.
        true_len = int(batch.padding_mask[b].sum().item())
        aa_logits[b, :true_len] = exact_aa_logits_for_sequence(
            adapter,
            batch.input_ids[b, :true_len],
            batch.valid[b, :true_len],
            max_forward_tokens=max_forward_tokens,
        )
    return pll_from_aa_logits(
        aa_logits,
        batch.target_aa_index.to(device),
        batch.valid.to(device),
        return_per_position=return_per_position,
    )


@torch.no_grad()
def exact_profiles_for_sequence(
    adapter: EncoderAdapter,
    input_ids_1d: torch.Tensor,
    valid_1d: torch.Tensor,
    target_aa_index_1d: torch.Tensor,
    *,
    max_forward_tokens: int = 16384,
) -> ExactProfiles:
    """Compute masked profiles at scorable positions (OFS-head training targets)."""
    aa_logits = exact_aa_logits_for_sequence(
        adapter, input_ids_1d, valid_1d, max_forward_tokens=max_forward_tokens
    )
    positions = torch.nonzero(valid_1d.to(aa_logits.device), as_tuple=False).flatten()
    profiles = profiles_from_aa_logits(aa_logits[positions])  # (n, 20)
    targets = target_aa_index_1d.to(positions.device)[positions]
    return ExactProfiles(positions=positions.cpu(), profiles=profiles.cpu(), targets=targets.cpu())

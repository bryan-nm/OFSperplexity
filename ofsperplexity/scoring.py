"""Pseudo-log-likelihood (PLL) and pseudo-perplexity from amino-acid logits.

This module is the single source of truth for the fitness math, shared by the
exact one-at-a-time scorer (:mod:`ofsperplexity.exact`) and the OFS single-pass
scorer (:mod:`ofsperplexity.ofs`).  Both produce, for every scorable position,
a length-20 vector of logits over the canonical amino acids; this module turns
those into per-residue log-likelihoods, the sequence PLL, and the pseudo-perplexity.

Definitions (OFS paper, Eqs. 4-5)::

    F_PLL(S) = (1/L) * sum_i  ln P(r_i = r_i^S | S_{-i})
    Z(S)     = exp(-F_PLL(S))          # pseudo-perplexity

where the sum runs over the L *scorable* positions (canonical amino acids only),
and P(.|S_{-i}) is the softmax over the 20 amino-acid logits at position i.

Lower pseudo-perplexity  <=>  more natural sequence  <=>  higher fitness.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class ScoreResult:
    """Per-sequence scoring output.

    ``pll`` and ``pseudo_perplexity`` are shape ``(B,)``. ``n_scored`` counts the
    scorable positions per sequence. ``per_position_logprob`` (optional) is
    ``(B, L)`` with 0.0 at non-scored positions.
    """

    pll: torch.Tensor
    pseudo_perplexity: torch.Tensor
    n_scored: torch.Tensor
    per_position_logprob: Optional[torch.Tensor] = None

    def as_dict(self) -> dict:
        return {
            "pll": self.pll.tolist(),
            "pseudo_perplexity": self.pseudo_perplexity.tolist(),
            "n_scored": self.n_scored.tolist(),
        }


def logprobs_from_aa_logits(
    aa_logits: torch.Tensor,
    target_aa_index: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Per-position log P(true residue) from amino-acid logits.

    Parameters
    ----------
    aa_logits: ``(B, L, 20)`` logits over the canonical amino acids.
    target_aa_index: ``(B, L)`` long, canonical aa index of the true residue
        (values in ``[0, 20)``; anything ``< 0`` is treated as non-scored).
    valid: ``(B, L)`` bool, True where the position should be scored.

    Returns ``(B, L)`` float log-probabilities; 0.0 where not valid.
    """
    log_probs = F.log_softmax(aa_logits.float(), dim=-1)  # (B, L, 20)
    safe_index = target_aa_index.clamp(min=0).unsqueeze(-1)  # (B, L, 1)
    gathered = log_probs.gather(-1, safe_index).squeeze(-1)  # (B, L)
    valid = valid & (target_aa_index >= 0)
    return torch.where(valid, gathered, torch.zeros_like(gathered))


def pll_from_aa_logits(
    aa_logits: torch.Tensor,
    target_aa_index: torch.Tensor,
    valid: torch.Tensor,
    *,
    return_per_position: bool = False,
) -> ScoreResult:
    """Sequence PLL and pseudo-perplexity from amino-acid logits.

    See module docstring for the math. Sequences with zero scorable positions get
    ``pll = 0`` and ``pseudo_perplexity = 1`` (a harmless sentinel; ``n_scored``
    is 0 so callers can filter).
    """
    valid = valid & (target_aa_index >= 0)
    per_pos = logprobs_from_aa_logits(aa_logits, target_aa_index, valid)  # (B, L)
    n_scored = valid.sum(dim=-1)  # (B,)
    denom = n_scored.clamp(min=1).to(per_pos.dtype)
    pll = per_pos.sum(dim=-1) / denom
    pseudo_perplexity = torch.exp(-pll)
    return ScoreResult(
        pll=pll,
        pseudo_perplexity=pseudo_perplexity,
        n_scored=n_scored,
        per_position_logprob=per_pos if return_per_position else None,
    )


def pll_from_profiles(
    profiles: torch.Tensor,
    target_aa_index: torch.Tensor,
    valid: torch.Tensor,
    *,
    return_per_position: bool = False,
    eps: float = 1e-12,
) -> ScoreResult:
    """Same as :func:`pll_from_aa_logits` but from probability profiles ``(B, L, 20)``.

    Used by the OFS ensemble head, whose prediction is a *mean of softmaxes* across
    members -- averaging probabilities, not logits, so we take ``log`` of the
    (already-normalised) mean profile directly rather than re-softmaxing.
    """
    valid = valid & (target_aa_index >= 0)
    log_probs = torch.log(profiles.float().clamp_min(eps))
    safe_index = target_aa_index.clamp(min=0).unsqueeze(-1)
    gathered = log_probs.gather(-1, safe_index).squeeze(-1)
    per_pos = torch.where(valid, gathered, torch.zeros_like(gathered))
    n_scored = valid.sum(dim=-1)
    denom = n_scored.clamp(min=1).to(per_pos.dtype)
    pll = per_pos.sum(dim=-1) / denom
    return ScoreResult(
        pll=pll,
        pseudo_perplexity=torch.exp(-pll),
        n_scored=n_scored,
        per_position_logprob=per_pos if return_per_position else None,
    )


def profiles_from_aa_logits(aa_logits: torch.Tensor) -> torch.Tensor:
    """Softmax the length-20 logits into probability profiles ``(..., 20)``.

    These profiles are the OFS "masked sequence profile": the per-position
    substitution distribution used both as OFS-head training targets and as the
    sampling distribution for the Monte-Carlo design workflow in the paper (Sec VI).
    """
    return F.softmax(aa_logits.float(), dim=-1)

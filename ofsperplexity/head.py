"""The One Fell Swoop head: a small MLP ensemble that maps an *unmasked* residue
embedding to that residue's *masked* amino-acid profile.

From the OFS paper (Appendix 1): "The one fell swoop setup is composed of an
ensemble of eight multilayer perceptron models that use ReLU activations and layer
normalization. The input for each MLP is 1280-dimensional followed by layers that
progressively reduce the dimensions of the input vector: [1280, 557, 243, 106, 46,
20]." The ensemble is ~7M parameters total and is trained to minimise the
cross-entropy between its prediction and the exact one-at-a-time masked profile.

Here the taper is generalised to any encoder hidden size ``D`` (ESMC/AMPLIFY are
960): the hidden widths are a geometric interpolation from ``D`` down to 20 over
five layers, which reproduces ``[1280, 557, 243, 106, 46, 20]`` exactly at ``D=1280``.

The head is the *only* trained component of OFS and is specific to one encoder's
embedding space. Train one per encoder with :mod:`ofsperplexity.train`; it is tiny,
so training and checkpoints are cheap. Inference is a single matmul stack applied
to every residue in parallel -- the "one fell swoop".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .alphabet import NUM_AA


def default_hidden_dims(input_dim: int, n_layers: int = 5, out_dim: int = NUM_AA) -> List[int]:
    """Geometric taper from ``input_dim`` to ``out_dim`` in ``n_layers`` steps.

    Returns the ``n_layers - 1`` hidden widths (excludes input and output). At
    ``input_dim=1280`` this yields ``[557, 243, 106, 46]`` -- the paper's dims.
    """
    ratio = (out_dim / input_dim) ** (1.0 / n_layers)
    dims = []
    for k in range(1, n_layers):
        dims.append(int(round(input_dim * (ratio ** k))))
    return dims


@dataclass
class OFSHeadConfig:
    input_dim: int
    hidden_dims: Optional[List[int]] = None
    out_dim: int = NUM_AA
    num_members: int = 8
    dropout: float = 0.0

    def resolved_hidden_dims(self) -> List[int]:
        return self.hidden_dims if self.hidden_dims is not None else default_hidden_dims(self.input_dim)

    def to_dict(self) -> dict:
        return {
            "input_dim": self.input_dim,
            "hidden_dims": self.resolved_hidden_dims(),
            "out_dim": self.out_dim,
            "num_members": self.num_members,
            "dropout": self.dropout,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "OFSHeadConfig":
        return cls(
            input_dim=int(d["input_dim"]),
            hidden_dims=list(d["hidden_dims"]) if d.get("hidden_dims") is not None else None,
            out_dim=int(d.get("out_dim", NUM_AA)),
            num_members=int(d.get("num_members", 8)),
            dropout=float(d.get("dropout", 0.0)),
        )


class _MLP(nn.Module):
    """One ensemble member: (Linear -> LayerNorm -> ReLU) x k -> Linear(->20)."""

    def __init__(self, input_dim: int, hidden_dims: List[int], out_dim: int, dropout: float):
        super().__init__()
        layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class OFSHead(nn.Module):
    """Ensemble of MLPs mapping embeddings ``(..., D)`` -> amino-acid logits.

    * :meth:`member_logits` -> ``(M, ..., 20)`` per-member logits (for training).
    * :meth:`predict_profile` -> ``(..., 20)`` mean-of-softmaxes ensemble profile.
    * :meth:`disagreement` -> ``(...,)`` mean pairwise KL across members (the
      heterogeneity diagnostic of the paper's Fig. 1f).
    """

    def __init__(self, config: OFSHeadConfig):
        super().__init__()
        self.config = config
        hidden = config.resolved_hidden_dims()
        self.members = nn.ModuleList(
            _MLP(config.input_dim, hidden, config.out_dim, config.dropout)
            for _ in range(config.num_members)
        )

    def member_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """Stack of per-member logits, shape ``(M, ..., 20)``."""
        return torch.stack([m(hidden) for m in self.members], dim=0)

    def predict_profile(self, hidden: torch.Tensor) -> torch.Tensor:
        """Ensemble profile: mean over members of ``softmax(member_logits)``."""
        logits = self.member_logits(hidden)                 # (M, ..., 20)
        probs = F.softmax(logits.float(), dim=-1)
        return probs.mean(dim=0)                            # (..., 20)

    @torch.no_grad()
    def disagreement(self, hidden: torch.Tensor) -> torch.Tensor:
        """Mean pairwise KL divergence between member profiles (``(...,)``)."""
        logp = F.log_softmax(self.member_logits(hidden).float(), dim=-1)  # (M, ..., 20)
        p = logp.exp()
        M = logp.shape[0]
        total = torch.zeros(logp.shape[1:-1], device=hidden.device)
        count = 0
        for i in range(M):
            for j in range(M):
                if i == j:
                    continue
                total = total + (p[i] * (logp[i] - logp[j])).sum(-1)
                count += 1
        return total / max(count, 1)

    # --------------------------------------------------------------- checkpoint
    def save(self, path: str) -> None:
        torch.save({"config": self.config.to_dict(), "state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path: str, map_location="cpu") -> "OFSHead":
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        head = cls(OFSHeadConfig.from_dict(ckpt["config"]))
        head.load_state_dict(ckpt["state_dict"])
        return head

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

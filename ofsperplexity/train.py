"""Train an OFS head to distil the exact masked profile into a single pass.

Pipeline (paper Appendix 1):

1. **Targets** -- for each training sequence, run the encoder *unmasked* once to get
   per-residue embeddings, and run the exact one-at-a-time calculation to get the
   masked amino-acid profile at each scorable position. The (embedding, profile)
   pairs are the distillation data.
2. **Fit** -- train the MLP ensemble to minimise the cross-entropy between its
   prediction and the exact profile (soft targets), with AdamW and a small
   validation split. Members differ only in random init + per-member shuffling,
   which is the paper's source of ensemble heterogeneity.

Target generation is the expensive part (it *is* the one-at-a-time calc) and is
embarrassingly parallel over sequences -- shard with :func:`ofsperplexity.dist`.
Fitting the ~7M-parameter head is cheap and runs on a single rank.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from .adapters import EncoderAdapter
from .alphabet import Alphabet
from .data import SequenceBatch, collate
from .exact import exact_aa_logits_for_sequence
from .head import OFSHead, OFSHeadConfig
from .scoring import profiles_from_aa_logits


@dataclass
class TargetSet:
    """Flat distillation dataset: one row per scorable residue across all sequences."""

    embeddings: torch.Tensor  # (N, D) unmasked residue embeddings
    profiles: torch.Tensor    # (N, 20) exact masked profiles (soft targets)
    targets: torch.Tensor     # (N,)   canonical aa index of the true residue

    def __len__(self) -> int:
        return self.embeddings.shape[0]

    def save(self, path: str) -> None:
        torch.save(
            {"embeddings": self.embeddings, "profiles": self.profiles, "targets": self.targets},
            path,
        )

    @classmethod
    def load(cls, path: str, map_location="cpu") -> "TargetSet":
        d = torch.load(path, map_location=map_location, weights_only=True)
        return cls(d["embeddings"], d["profiles"], d["targets"])

    @classmethod
    def concat(cls, parts: Sequence["TargetSet"]) -> "TargetSet":
        return cls(
            torch.cat([p.embeddings for p in parts]),
            torch.cat([p.profiles for p in parts]),
            torch.cat([p.targets for p in parts]),
        )


@torch.no_grad()
def generate_targets(
    adapter: EncoderAdapter,
    records: Sequence[Tuple[str, str]],
    encode_fn,
    *,
    max_forward_tokens: int = 16384,
    store_dtype: torch.dtype = torch.float32,
    progress: bool = False,
) -> TargetSet:
    """Build the distillation dataset from ``(id, seq)`` records.

    Embeddings are collected on CPU to bound accelerator memory; move the returned
    tensors back to the device in :func:`train_ofs_head`.
    """
    alphabet: Alphabet = adapter.alphabet
    device = adapter.device
    emb_parts: List[torch.Tensor] = []
    prof_parts: List[torch.Tensor] = []
    tgt_parts: List[torch.Tensor] = []
    for n, (name, seq) in enumerate(records):
        batch = collate([(name, seq)], encode_fn, alphabet)
        input_ids = batch.input_ids.to(device)
        valid = batch.valid.to(device)
        tgt_idx = batch.target_aa_index.to(device)
        positions = torch.nonzero(valid[0], as_tuple=False).flatten()
        if positions.numel() == 0:
            continue
        # unmasked embeddings (single pass) at scorable positions
        hidden = adapter.forward_hidden(input_ids, batch.padding_mask.to(device))[0]  # (L, D)
        emb = hidden[positions].to("cpu", store_dtype)
        # exact masked profiles at scorable positions
        aa_logits = exact_aa_logits_for_sequence(
            adapter, input_ids[0], valid[0], max_forward_tokens=max_forward_tokens
        )
        prof = profiles_from_aa_logits(aa_logits[positions]).to("cpu", store_dtype)
        emb_parts.append(emb)
        prof_parts.append(prof)
        tgt_parts.append(tgt_idx[0][positions].to("cpu"))
        if progress and (n + 1) % 50 == 0:
            print(f"[generate_targets] {n + 1}/{len(records)} sequences", flush=True)
    if not emb_parts:
        raise ValueError("No scorable positions found in the training records.")
    return TargetSet(torch.cat(emb_parts), torch.cat(prof_parts), torch.cat(tgt_parts))


@dataclass
class TrainConfig:
    num_members: int = 8
    hidden_dims: Optional[List[int]] = None
    epochs: int = 40
    batch_size: int = 4096
    lr: float = 1e-3
    weight_decay: float = 1e-2
    val_fraction: float = 0.01  # paper uses a 99:1 split
    dropout: float = 0.0
    seed: int = 0
    device: str = "auto"


def _soft_cross_entropy(logits: torch.Tensor, target_profile: torch.Tensor) -> torch.Tensor:
    """CE(target || pred) = -sum_a target_a * log_softmax(logits)_a, mean over rows."""
    logp = F.log_softmax(logits.float(), dim=-1)
    return -(target_profile * logp).sum(-1).mean()


def train_ofs_head(
    target_set: TargetSet,
    input_dim: int,
    config: TrainConfig = TrainConfig(),
    *,
    device: Optional[torch.device] = None,
    log_every: int = 5,
) -> Tuple[OFSHead, dict]:
    """Fit the ensemble head on a :class:`TargetSet`. Returns ``(head, history)``."""
    from .dist import pick_device

    dev = device or pick_device(config.device)
    torch.manual_seed(config.seed)

    N = len(target_set)
    perm = torch.randperm(N)
    n_val = max(1, int(N * config.val_fraction))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    emb = target_set.embeddings.to(dev)
    prof = target_set.profiles.to(dev)

    head_cfg = OFSHeadConfig(
        input_dim=input_dim,
        hidden_dims=config.hidden_dims,
        num_members=config.num_members,
        dropout=config.dropout,
    )
    head = OFSHead(head_cfg).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    history = {"train_loss": [], "val_loss": []}
    for epoch in range(config.epochs):
        head.train()
        shuffle = train_idx[torch.randperm(train_idx.numel())]
        running = 0.0
        nb = 0
        for start in range(0, shuffle.numel(), config.batch_size):
            idx = shuffle[start : start + config.batch_size]
            x = emb[idx]
            t = prof[idx]
            # each member sees an independently shuffled view -> ensemble diversity
            member_logits = head.member_logits(x)  # (M, B, 20)
            loss = torch.stack(
                [_soft_cross_entropy(member_logits[m], t) for m in range(head_cfg.num_members)]
            ).sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += loss.item()
            nb += 1
        train_loss = running / max(nb, 1)

        head.eval()
        with torch.no_grad():
            vlogits = head.member_logits(emb[val_idx])
            val_loss = torch.stack(
                [_soft_cross_entropy(vlogits[m], prof[val_idx]) for m in range(head_cfg.num_members)]
            ).mean().item()
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        if log_every and (epoch % log_every == 0 or epoch == config.epochs - 1):
            print(f"[train] epoch {epoch:3d}  train_ce={train_loss:.4f}  val_ce={val_loss:.4f}", flush=True)

    return head, history

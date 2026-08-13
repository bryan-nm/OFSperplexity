"""OFSperplexity -- One Fell Swoop pseudo-perplexity for protein language models.

Fast, distributed (Aurora XPU / CUDA / CPU) estimation of protein pseudo-perplexity
from any masked encoder (ESMC, AMPLIFY, or a model you are training), following
Kantroo, Wagner & Machta, "Pseudo-perplexity in one fell swoop for protein fitness
estimation", PRX Life 3, 033014 (2025).

Two scorers:

* :func:`exact_score` / :class:`~ofsperplexity.exact` -- the reference one-at-a-time
  pseudo-perplexity (L forward passes; ground truth; needs no training).
* :class:`OFSScorer` -- single forward pass via a trained :class:`OFSHead`; the fast
  path for on-the-fly evaluation during training.

Quick start (standalone FASTA)::

    from ofsperplexity.models import load_encoder
    from ofsperplexity import OFSScorer, OFSHead
    from ofsperplexity.data import read_sequences, iter_batches

    enc = load_encoder("esmc", "/path/to/ESMC-300M", device="auto")
    head = OFSHead.load("esmc_ofs_head.pt")
    scorer = OFSScorer(enc.adapter, head)
    for batch in iter_batches(read_sequences("seqs.fasta"), enc.encode_fn, enc.alphabet, max_tokens=16384):
        res = scorer.score(batch)
        for name, pp in zip(batch.ids, res.pseudo_perplexity.tolist()):
            print(name, pp)

To wire into your own model, build an adapter -- see ``docs/INTEGRATION.md``.
"""

from __future__ import annotations

from .alphabet import CANONICAL_AA, NUM_AA, Alphabet
from .adapters import (
    BaseAdapter,
    CallableAdapter,
    EncoderAdapter,
    EncoderOutput,
    HFMaskedLMAdapter,
    ModuleAdapter,
)
from .data import SequenceBatch, collate, iter_batches, read_fasta, read_sequences
from .exact import exact_score, exact_aa_logits_for_sequence, exact_profiles_for_sequence
from .head import OFSHead, OFSHeadConfig, default_hidden_dims
from .ofs import OFSScorer
from .scoring import (
    ScoreResult,
    pll_from_aa_logits,
    pll_from_profiles,
    profiles_from_aa_logits,
)
from .train import TargetSet, TrainConfig, generate_targets, train_ofs_head

__version__ = "0.1.0"

__all__ = [
    "Alphabet",
    "CANONICAL_AA",
    "NUM_AA",
    "EncoderAdapter",
    "EncoderOutput",
    "BaseAdapter",
    "HFMaskedLMAdapter",
    "ModuleAdapter",
    "CallableAdapter",
    "SequenceBatch",
    "collate",
    "iter_batches",
    "read_fasta",
    "read_sequences",
    "exact_score",
    "exact_aa_logits_for_sequence",
    "exact_profiles_for_sequence",
    "OFSHead",
    "OFSHeadConfig",
    "default_hidden_dims",
    "OFSScorer",
    "ScoreResult",
    "pll_from_aa_logits",
    "pll_from_profiles",
    "profiles_from_aa_logits",
    "TargetSet",
    "TrainConfig",
    "generate_targets",
    "train_ofs_head",
]

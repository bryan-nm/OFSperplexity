"""Sequence I/O, tokenization, and length-bucketed batching.

The scorers consume :class:`SequenceBatch` objects. A batch carries everything the
math needs and nothing model-specific:

* ``input_ids``      ``(B, L)`` long, padded, *with* the encoder's special tokens
* ``padding_mask``   ``(B, L)`` bool, True on real (non-pad) tokens
* ``target_aa_index````(B, L)`` long, canonical aa index of each token (-1 if not an AA)
* ``valid``          ``(B, L)`` bool, positions that enter the PLL mean
* ``ids``            list of FASTA headers / sequence names (len B)

Tokenization is injected as an ``encode_fn: str -> list[int]`` so the library stays
free of any particular tokenizer. The model loaders in :mod:`ofsperplexity.models`
return a ready ``encode_fn``; for a model you are training, pass your own.

Length bucketing (:func:`iter_batches`) sorts by length and packs a token budget per
batch so padding waste stays low -- this is the main throughput lever for exact
scoring, where every masked replica of a sequence is a separate row.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Iterator, List, Optional, Sequence, Tuple

import torch

from .alphabet import Alphabet

EncodeFn = Callable[[str], List[int]]


# --------------------------------------------------------------------- FASTA I/O
def read_fasta(path: str) -> Iterator[Tuple[str, str]]:
    """Yield ``(header, sequence)`` from a plain FASTA file.

    Header is the text after ``>`` up to the first whitespace. Sequence is
    uppercased with internal whitespace removed. Handles multi-line records.
    """
    header: Optional[str] = None
    chunks: List[str] = []
    with open(path, "r") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(chunks).upper()
                header = line[1:].strip().split()[0] if line[1:].strip() else ""
                chunks = []
            else:
                chunks.append(line.strip())
        if header is not None:
            yield header, "".join(chunks).upper()


def read_sequences(path: str) -> List[Tuple[str, str]]:
    """Read a FASTA (``.fasta``/``.fa``/``.faa``) or a plain one-sequence-per-line
    file into a list of ``(id, seq)``. For a line file, ids are ``seq{i}``."""
    lower = path.lower()
    if lower.endswith((".fasta", ".fa", ".faa", ".fna")):
        return list(read_fasta(path))
    records: List[Tuple[str, str]] = []
    with open(path) as fh:
        for i, line in enumerate(fh):
            s = line.strip().upper()
            if s:
                records.append((f"seq{i}", s))
    return records


# ------------------------------------------------------------------------ batch
@dataclass
class SequenceBatch:
    input_ids: torch.Tensor       # (B, L) long
    padding_mask: torch.Tensor    # (B, L) bool  (True = real token)
    target_aa_index: torch.Tensor # (B, L) long  (-1 where not a canonical AA)
    valid: torch.Tensor           # (B, L) bool  (scorable positions)
    ids: List[str]

    @property
    def batch_size(self) -> int:
        return self.input_ids.shape[0]

    def to(self, device) -> "SequenceBatch":
        return SequenceBatch(
            input_ids=self.input_ids.to(device),
            padding_mask=self.padding_mask.to(device),
            target_aa_index=self.target_aa_index.to(device),
            valid=self.valid.to(device),
            ids=self.ids,
        )


def collate(
    records: Sequence[Tuple[str, str]],
    encode_fn: EncodeFn,
    alphabet: Alphabet,
) -> SequenceBatch:
    """Tokenize and pad a list of ``(id, seq)`` into a :class:`SequenceBatch`."""
    ids = [r[0] for r in records]
    encoded = [torch.as_tensor(encode_fn(r[1]), dtype=torch.long) for r in records]
    lengths = [t.numel() for t in encoded]
    maxlen = max(lengths) if lengths else 0
    pad_id = alphabet.pad_token_id
    B = len(encoded)
    input_ids = torch.full((B, maxlen), pad_id, dtype=torch.long)
    padding_mask = torch.zeros((B, maxlen), dtype=torch.bool)
    for i, t in enumerate(encoded):
        input_ids[i, : t.numel()] = t
        padding_mask[i, : t.numel()] = True
    target_aa_index = alphabet.token_ids_to_aa_index(input_ids)
    valid = (target_aa_index >= 0) & padding_mask
    return SequenceBatch(input_ids, padding_mask, target_aa_index, valid, ids)


def iter_batches(
    records: Sequence[Tuple[str, str]],
    encode_fn: EncodeFn,
    alphabet: Alphabet,
    *,
    batch_size: Optional[int] = None,
    max_tokens: Optional[int] = None,
    sort_by_length: bool = True,
) -> Iterator[SequenceBatch]:
    """Yield :class:`SequenceBatch` objects, optionally length-sorted and token-capped.

    Provide ``batch_size`` (fixed rows) or ``max_tokens`` (dynamic rows, packs up to
    ~``max_tokens`` residues per batch to bound memory while minimising pad waste).
    If both are given, ``max_tokens`` also caps a fixed-size batch.
    """
    if batch_size is None and max_tokens is None:
        batch_size = 8
    order = list(range(len(records)))
    if sort_by_length:
        order.sort(key=lambda i: len(records[i][1]))
    batch: List[Tuple[str, str]] = []
    cur_max = 0
    for i in order:
        rec = records[i]
        approx_len = len(rec[1]) + 2  # +bos/eos
        prospective = max(cur_max, approx_len) * (len(batch) + 1)
        over_tokens = max_tokens is not None and batch and prospective > max_tokens
        over_rows = batch_size is not None and len(batch) >= batch_size
        if batch and (over_tokens or over_rows):
            yield collate(batch, encode_fn, alphabet)
            batch, cur_max = [], 0
        batch.append(rec)
        cur_max = max(cur_max, approx_len)
    if batch:
        yield collate(batch, encode_fn, alphabet)

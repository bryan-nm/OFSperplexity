#!/usr/bin/env python3
"""Subsample a raw UniRef90 FASTA(.gz) into a curated FASTA for OFS head training.

Borrowed in spirit from ProLoopDiff's ``src/preprocess_fasta.py`` (same length +
residue filtering, same gzip streaming), with three changes for OFS:

  * output is a **plain FASTA** (fed to ``ofs-pppl gen-targets``), not packed
    training shards;
  * it **reservoir-subsamples to N sequences** in a single streaming pass -- you
    do NOT want all ~190M UniRef90 sequences for a 7M-param head (see docs/SIZING.md);
  * it does **NOT exclude SwissProt**. UniRef90 already contains the SwissProt
    entries (UniRef is built from UniProt = SwissProt + TrEMBL); keeping them is a
    feature here, so there is no ``--exclude-csv``.

Reservoir sampling gives a uniform random sample of N without knowing the total
count up front, in O(N) memory, and the output order is already shuffled -- which is
exactly what ``gen-targets`` wants for balanced length across MPI shards.

Torch-free and dependency-free (stdlib only) so it can run on a login/DTN node.

Usage
-----
    python scripts/preprocess_uniref.py uniref90.fasta.gz uniref90_100k.fasta \
        --num 100000 --min-len 30 --max-len 512 --seed 0

    # keep everything that passes the filters (no subsample):
    python scripts/preprocess_uniref.py in.fasta.gz out.fasta --num 0
"""
from __future__ import annotations

import argparse
import gzip
import random
import sys
from typing import Iterator, List, Optional, Tuple

# Canonical 20 amino acids (kept local so this script has no package/torch deps).
CANONICAL_AA = "ACDEFGHIKLMNPQRSTVWY"
_STANDARD = set(CANONICAL_AA)
# Ambiguity / non-standard codes accepted as *context* (they are excluded from the
# OFS PLL mean anyway). Matches ProLoopDiff's _RARE keys plus X. Anything else
# (gaps '-', '.', stop '*', digits, ...) rejects the sequence.
_AMBIGUOUS = set("BZJUOX")
_ALLOWED = _STANDARD | _AMBIGUOUS


def open_maybe_gz(path: str, mode: str = "rt"):
    return gzip.open(path, mode) if path.endswith(".gz") else open(path, mode)


def fasta_iter(path: str) -> Iterator[Tuple[str, str]]:
    """Yield ``(header, sequence)``; header is the text after '>' (first token)."""
    name, buf = None, []
    with open_maybe_gz(path) as f:
        for line in f:
            if line.startswith(">"):
                if name is not None:
                    yield name, "".join(buf)
                name, buf = line[1:].strip(), []
            else:
                buf.append(line.strip())
    if name is not None:
        yield name, "".join(buf)


def is_valid(seq: str, min_len: int, max_len: int, max_ambiguous_frac: float) -> bool:
    n = len(seq)
    if not (min_len <= n <= max_len):
        return False
    ambig = 0
    for c in seq:
        if c in _STANDARD:
            continue
        if c in _AMBIGUOUS:
            ambig += 1
        else:
            return False  # illegal character (gap, stop, digit, ...)
    return ambig <= max_ambiguous_frac * n


def reservoir_subsample(
    records: Iterator[Tuple[str, str]], num: int, rng: random.Random
) -> Tuple[List[Tuple[str, str]], int]:
    """Uniform sample of ``num`` items from a stream (Algorithm R). Returns
    ``(reservoir, n_seen)``. If the stream has <= num items, returns them all."""
    reservoir: List[Tuple[str, str]] = []
    seen = 0
    for rec in records:
        seen += 1
        if len(reservoir) < num:
            reservoir.append(rec)
        else:
            j = rng.randint(0, seen - 1)
            if j < num:
                reservoir[j] = rec
    return reservoir, seen


def write_fasta(records: List[Tuple[str, str]], path: str) -> None:
    with open_maybe_gz(path, "wt") as out:
        for header, seq in records:
            out.write(f">{header}\n{seq}\n")


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("fasta", help="Input FASTA(.gz), e.g. uniref90.fasta.gz")
    ap.add_argument("out", help="Output FASTA(.gz)")
    ap.add_argument("--num", type=int, default=100_000,
                    help="Target #sequences (reservoir sample). 0 = keep all that pass filters.")
    ap.add_argument("--min-len", type=int, default=30)
    ap.add_argument("--max-len", type=int, default=512,
                    help="Upper length cap. gen-targets cost is O(L^2), so keep this modest.")
    ap.add_argument("--max-ambiguous-frac", type=float, default=0.10,
                    help="Reject sequences with more than this fraction of non-standard residues.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--report-every", type=int, default=5_000_000)
    a = ap.parse_args(argv)

    rng = random.Random(a.seed)

    def filtered() -> Iterator[Tuple[str, str]]:
        kept = seen = 0
        for header, seq in fasta_iter(a.fasta):
            seen += 1
            seq = seq.strip().upper()
            if is_valid(seq, a.min_len, a.max_len, a.max_ambiguous_frac):
                kept += 1
                yield header, seq
            if a.report_every and seen % a.report_every == 0:
                print(f"[preprocess] scanned {seen:,}  passed filter {kept:,}", flush=True)

    if a.num and a.num > 0:
        reservoir, _ = reservoir_subsample(filtered(), a.num, rng)
        # Already random order from the reservoir; a final shuffle is belt-and-braces.
        rng.shuffle(reservoir)
        write_fasta(reservoir, a.out)
        print(f"[preprocess] wrote {len(reservoir):,} sequences -> {a.out}", flush=True)
        if len(reservoir) < a.num:
            print(f"[preprocess] NOTE: only {len(reservoir):,} sequences passed filters "
                  f"(< requested {a.num:,}); wrote all of them.", flush=True)
    else:
        # Stream straight through; no subsample, bounded memory.
        n = 0
        with open_maybe_gz(a.out, "wt") as out:
            for header, seq in filtered():
                out.write(f">{header}\n{seq}\n")
                n += 1
        print(f"[preprocess] wrote {n:,} sequences (no subsample) -> {a.out}", flush=True)


if __name__ == "__main__":
    main()

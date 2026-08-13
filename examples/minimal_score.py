"""Minimal standalone scoring against a real encoder (ESMC or AMPLIFY).

Usage:
    python examples/minimal_score.py esmc    /path/to/ESMC-300M
    python examples/minimal_score.py amplify /path/to/SaAMPLIFY_350M

Runs the EXACT one-at-a-time pseudo-perplexity (no head needed) on a few sequences
and prints them. For the fast single-pass path, train a head (`ofs-pppl train`) and
use OFSScorer -- see the README.
"""

import sys

from ofsperplexity.models import load_encoder
from ofsperplexity import collate, exact_score

SEQS = [
    ("natural_gfp_frag", "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLV"),
    ("random_iid",       "QWNPCTVMHRDKFYAEILGSQWNPCTVMHRDKFYAEILGSQWNPCTVMHRDKFYAEILGS"),
]


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    kind, path = sys.argv[1], sys.argv[2]
    enc = load_encoder(kind, path, device="auto")
    batch = collate(SEQS, enc.encode_fn, enc.alphabet)
    res = exact_score(enc.adapter, batch)
    print(f"{'id':20s} {'pseudo_perplexity':>18s} {'n_scored':>9s}")
    for name, pp, n in zip(batch.ids, res.pseudo_perplexity.tolist(), res.n_scored.tolist()):
        print(f"{name:20s} {pp:18.4f} {n:9d}")


if __name__ == "__main__":
    main()

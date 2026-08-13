"""Tests for scripts/preprocess_uniref.py (FASTA filtering + reservoir subsample)."""

import gzip
import importlib.util
import os
import random

_HERE = os.path.dirname(__file__)
_SCRIPT = os.path.join(_HERE, "..", "scripts", "preprocess_uniref.py")
_spec = importlib.util.spec_from_file_location("preprocess_uniref", _SCRIPT)
pp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pp)


def _write_mock(path, n_valid=500, seed=0):
    rng = random.Random(seed)
    aa = pp.CANONICAL_AA
    recs = [(f"UniRef90_V{i}", "".join(rng.choice(aa) for _ in range(rng.randint(40, 200))))
            for i in range(n_valid)]
    recs += [
        ("too_short", "ACDE"),
        ("too_long", "M" * 5000),
        ("has_gap", "ACDE-FGHIK" * 6),
        ("has_stop", "ACDEFG*HIK" * 6),
        ("x_heavy", "X" * 50 + "ACDEFG"),
        ("ambiguous_ok", "ACDEFGHIKLMNBZUACDEFGHIKLM"),  # a few B/Z/U, within frac
    ]
    rng.shuffle(recs)
    with gzip.open(path, "wt") as f:
        for h, s in recs:
            f.write(f">{h}\n{s}\n")


def test_is_valid_rules():
    assert pp.is_valid("ACDEFGHIKL", 5, 100, 0.1)
    assert not pp.is_valid("ACDE", 30, 100, 0.1)          # too short
    assert not pp.is_valid("A" * 200, 30, 100, 0.1)       # too long
    assert not pp.is_valid("ACDE-FGHIK", 5, 100, 0.1)     # gap
    assert not pp.is_valid("ACDE*FGHIK", 5, 100, 0.1)     # stop
    assert not pp.is_valid("X" * 6 + "ACDE", 5, 100, 0.1) # >10% ambiguous
    assert pp.is_valid("ACDEFGHIKLB", 5, 100, 0.2)        # one ambiguous ok


def test_subsample_count_and_filtering(tmp_path):
    src = str(tmp_path / "uniref.fasta.gz")
    out = str(tmp_path / "sub.fasta")
    _write_mock(src, n_valid=500)
    pp.main([src, out, "--num", "100", "--min-len", "30", "--max-len", "512", "--seed", "5"])
    recs = list(pp.fasta_iter(out))
    assert len(recs) == 100
    for _, s in recs:                      # no illegal residues leaked
        assert all(c in pp._ALLOWED for c in s)
        assert 30 <= len(s) <= 512


def test_reproducible_with_seed(tmp_path):
    src = str(tmp_path / "u.fasta.gz")
    _write_mock(src, n_valid=300)
    a, b = str(tmp_path / "a.fasta"), str(tmp_path / "b.fasta")
    pp.main([src, a, "--num", "50", "--seed", "7"])
    pp.main([src, b, "--num", "50", "--seed", "7"])
    assert open(a).read() == open(b).read()


def test_keep_all_excludes_edge_cases(tmp_path):
    src = str(tmp_path / "u.fasta.gz")
    out = str(tmp_path / "all.fasta")
    _write_mock(src, n_valid=200)
    pp.main([src, out, "--num", "0", "--min-len", "30", "--max-len", "512"])
    recs = list(pp.fasta_iter(out))
    assert len(recs) == 200                # all 6 edge cases dropped

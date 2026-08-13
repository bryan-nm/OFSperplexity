"""End-to-end smoke test with a tiny fake encoder (no model downloads needed).

Verifies the scoring math, the exact<->OFS plumbing, alphabet mapping, batching,
and head train/save/load. Run: ``pytest tests/test_smoke.py`` in a torch env.
"""

import torch

from ofsperplexity import (
    Alphabet, CANONICAL_AA, NUM_AA, OFSHead, OFSHeadConfig, OFSScorer,
    CallableAdapter, collate, exact_score, generate_targets, train_ofs_head,
    TrainConfig,
)


# ---- a fake 24-token vocab: 20 AAs at ids 0..19, then pad/mask/bos/eos ----
PAD, MASK, BOS, EOS = 20, 21, 22, 23
VOCAB = 24
HID = 32


def make_alphabet():
    t2i = {aa: i for i, aa in enumerate(CANONICAL_AA)}
    return Alphabet.from_token_maps(
        t2i, mask_token_id=MASK, pad_token_id=PAD, vocab_size=VOCAB,
        bos_token_id=BOS, eos_token_id=EOS,
    )


def encode_fn(seq):
    idx = {aa: i for i, aa in enumerate(CANONICAL_AA)}
    return [BOS] + [idx.get(c, 0) for c in seq] + [EOS]


class FakeModel(torch.nn.Module):
    """Deterministic embedding + linear head over the vocab."""

    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(VOCAB, HID)
        self.dec = torch.nn.Linear(HID, VOCAB)
        torch.manual_seed(1)
        for p in self.parameters():
            torch.nn.init.normal_(p, std=0.3)

    def hidden(self, input_ids, padding_mask=None):
        return self.emb(input_ids)

    def logits(self, input_ids, padding_mask=None):
        return self.dec(self.emb(input_ids))


def make_adapter():
    model = FakeModel().eval()
    ab = make_alphabet()
    return CallableAdapter(
        ab, hidden_size=HID, device=torch.device("cpu"),
        hidden_fn=lambda ids, pm: model.hidden(ids, pm),
        logits_fn=lambda ids, pm: model.logits(ids, pm),
    ), model


def test_alphabet_roundtrip():
    ab = make_alphabet()
    assert ab.aa_token_ids.tolist() == list(range(NUM_AA))
    ids = torch.tensor([[BOS, 0, 1, 2, MASK, EOS, PAD]])
    scorable = ab.scorable_mask(ids)
    assert scorable.tolist() == [[False, True, True, True, False, False, False]]


def test_exact_score_runs():
    adapter, _ = make_adapter()
    records = [("a", "ACDEFG"), ("b", "MKLWYY")]
    batch = collate(records, encode_fn, adapter.alphabet)
    res = exact_score(adapter, batch)
    assert res.pseudo_perplexity.shape == (2,)
    assert (res.pseudo_perplexity > 0).all()
    assert res.n_scored.tolist() == [6, 6]


def test_ofs_matches_exact_after_distillation():
    """A head distilled on the exact targets should track exact PP (loose tol)."""
    adapter, _ = make_adapter()
    records = [("s%d" % i, "".join(CANONICAL_AA[(i + j) % NUM_AA] for j in range(15)))
               for i in range(40)]
    ts = generate_targets(adapter, records, encode_fn)
    head, hist = train_ofs_head(
        ts, adapter.hidden_size,
        TrainConfig(num_members=2, epochs=60, batch_size=256, lr=5e-3, val_fraction=0.1),
        log_every=0,
    )
    assert hist["val_loss"][-1] < hist["val_loss"][0]  # it learned something

    scorer = OFSScorer(adapter, head)
    batch = collate(records[:4], encode_fn, adapter.alphabet)
    ofs = scorer.score(batch).pseudo_perplexity
    exact = exact_score(adapter, batch).pseudo_perplexity
    # correlated & same ballpark
    assert torch.allclose(ofs, exact, rtol=0.5, atol=2.0)


class ContextFakeModel(FakeModel):
    """Like FakeModel but position i's embedding depends on the whole sequence,
    so trailing-pad contamination in exact scoring would change the result."""

    def hidden(self, input_ids, padding_mask=None):
        e = self.emb(input_ids)                     # (B, L, H)
        return e + e.mean(dim=1, keepdim=True)      # context-mixed

    def logits(self, input_ids, padding_mask=None):
        return self.dec(self.hidden(input_ids, padding_mask))


def test_exact_is_padding_invariant():
    model = ContextFakeModel().eval()
    ab = make_alphabet()
    adapter = CallableAdapter(
        ab, hidden_size=HID, device=torch.device("cpu"),
        hidden_fn=lambda ids, pm: model.hidden(ids, pm),
        logits_fn=lambda ids, pm: model.logits(ids, pm),
    )
    recs = [("a", "ACDEFGHIKLMN"), ("b", "MKLW"), ("c", "ACDEFGH")]
    padded = exact_score(adapter, collate(recs, encode_fn, ab)).pseudo_perplexity
    solo = torch.tensor(
        [exact_score(adapter, collate([r], encode_fn, ab)).pseudo_perplexity.item() for r in recs]
    )
    assert torch.allclose(padded, solo, atol=1e-4), (padded, solo)


def test_head_save_load(tmp_path):
    cfg = OFSHeadConfig(input_dim=HID, num_members=3)
    head = OFSHead(cfg)
    p = str(tmp_path / "head.pt")
    head.save(p)
    head2 = OFSHead.load(p)
    x = torch.randn(5, HID)
    assert torch.allclose(head.predict_profile(x), head2.predict_profile(x))

"""Template: OFS pseudo-perplexity as an on-the-fly eval during model training.

This is a self-contained, runnable sketch of the pattern described in
``docs/INTEGRATION.md``. It uses a tiny fake encoder so it runs anywhere with just
torch; replace ``build_reference_encoder`` and the two extractor closures with your
real encoder, and load a real pretrained head, and this drops straight into a
training loop.

Run:  python examples/integrate_training_loop.py
"""

from __future__ import annotations

import torch

from ofsperplexity import (
    Alphabet, CANONICAL_AA, ModuleAdapter, OFSHead, OFSScorer, TrainConfig,
    collate, exact_score, generate_targets, train_ofs_head,
)

# --------------------------------------------------------------------------- #
# 0. A stand-in reference encoder. Replace ALL of this with your real encoder.  #
# --------------------------------------------------------------------------- #
PAD, MASK, BOS, EOS, VOCAB, D = 20, 21, 22, 23, 24, 48


class ReferenceEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(VOCAB, D)
        self.enc = torch.nn.TransformerEncoderLayer(D, nhead=4, batch_first=True)
        self.head = torch.nn.Linear(D, VOCAB)

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False):
        # attention_mask here is a boolean keep-mask (True=real); Transformer wants
        # a padding mask that is True where padded, so invert.
        kpm = None if attention_mask is None else ~attention_mask.bool()
        h = self.enc(self.emb(input_ids), src_key_padding_mask=kpm)
        return h, self.head(h)


def build_reference_encoder():
    torch.manual_seed(0)
    model = ReferenceEncoder().eval()
    for p in model.parameters():
        p.requires_grad_(False)          # frozen: scoring must not touch its grads
    return model


def build_alphabet():
    t2i = {aa: i for i, aa in enumerate(CANONICAL_AA)}   # AAs at ids 0..19
    return Alphabet.from_token_maps(
        t2i, mask_token_id=MASK, pad_token_id=PAD, vocab_size=VOCAB,
        bos_token_id=BOS, eos_token_id=EOS,
    )


def encode_fn(seq: str):
    idx = {aa: i for i, aa in enumerate(CANONICAL_AA)}
    return [BOS] + [idx.get(c, 0) for c in seq] + [EOS]


# --------------------------------------------------------------------------- #
# 1. Wrap the encoder in an adapter (the ONLY integration surface).            #
# --------------------------------------------------------------------------- #
def build_adapter(encoder, alphabet):
    def hidden_fn(module, input_ids, padding_mask):
        h, _ = module(input_ids, attention_mask=padding_mask, output_hidden_states=True)
        return h                                    # (B, L, D) last-layer embeddings

    def logits_fn(module, input_ids, padding_mask):
        _, logits = module(input_ids, attention_mask=padding_mask)
        return logits                               # (B, L, V) full-vocab logits

    return ModuleAdapter(
        encoder, alphabet, hidden_fn=hidden_fn, logits_fn=logits_fn,
        hidden_size=D, no_grad=True,
    )


# --------------------------------------------------------------------------- #
# 2. One-time: train an OFS head for this encoder (normally done offline).      #
# --------------------------------------------------------------------------- #
def train_head_once(adapter):
    # A representative set. In practice: curated Swissprot-like FASTA, 1000s of seqs.
    records = [("s%d" % i, "".join(CANONICAL_AA[(i * 7 + j) % 20] for j in range(24)))
               for i in range(64)]
    targets = generate_targets(adapter, records, encode_fn)
    head, _ = train_ofs_head(
        targets, adapter.hidden_size,
        TrainConfig(num_members=4, epochs=80, batch_size=512, lr=3e-3, val_fraction=0.1),
        log_every=0,
    )
    return head


# --------------------------------------------------------------------------- #
# 3. In your training loop: score a batch of sequences in ONE forward pass.     #
# --------------------------------------------------------------------------- #
def main():
    encoder = build_reference_encoder()
    alphabet = build_alphabet()
    adapter = build_adapter(encoder, alphabet)

    # Offline in real usage; loaded with OFSHead.load("head.pt").
    head = train_head_once(adapter)
    scorer = OFSScorer(adapter, head)

    # Pretend these came out of the model you are training this step.
    generated = ["ACDEFGHIKLMNPQRSTVWY", "MKKLLPTAAAGLLLLAAQPAM", "GGGGGGSSSSSSAAAAAA"]
    batch = collate([(f"g{i}", s) for i, s in enumerate(generated)], encode_fn, alphabet)

    ofs = scorer.score(batch)                       # single forward pass
    exact = exact_score(adapter, batch)             # ground truth, for comparison

    print(f"{'sequence':22s} {'OFS_pppl':>10s} {'exact_pppl':>11s} {'n_scored':>9s}")
    for name, o, e, n in zip(batch.ids, ofs.pseudo_perplexity, exact.pseudo_perplexity, ofs.n_scored):
        print(f"{name:22s} {o.item():10.3f} {e.item():11.3f} {n.item():9d}")

    # This is the number you would log every N steps:
    print(f"\nmean OFS pseudo-perplexity = {ofs.pseudo_perplexity.mean().item():.3f}")


if __name__ == "__main__":
    main()

# Wiring OFSperplexity into your own model

This guide is for an engineer **or an agent** who wants pseudo-perplexity as an
on-the-fly metric during training/fine-tuning of a protein model, computed against
some encoder, independently of the model being optimized.

You do **not** need to subclass anything or match a model API. The whole library
talks to one small interface, the **`EncoderAdapter`**. Give it a way to get hidden
states (and/or logits) out of your encoder and you are done.

---

## The contract

An adapter must provide:

```python
adapter.hidden_size -> int            # D, the per-residue embedding width
adapter.device      -> torch.device
adapter.alphabet    -> Alphabet       # maps token ids <-> the 20 amino acids
adapter.forward(input_ids, padding_mask, *, need_hidden, need_logits) -> EncoderOutput
#   input_ids:    (B, L) long, WITH the encoder's special tokens (bos/eos/cls)
#   padding_mask: (B, L) bool, True on real tokens (None = no padding)
#   EncoderOutput.hidden: (B, L, D) last-layer residue embeddings   (for OFS)
#   EncoderOutput.logits: (B, L, V) full-vocab MLM logits           (for exact)
```

* **OFS scoring** (the fast path) needs only `hidden`.
* **Exact scoring** and **target generation** need `logits` (the module must accept
  a `<mask>` token in `input_ids`).

You will almost always use the ready-made `ModuleAdapter` rather than writing a
class. Three ingredients:

1. **The alphabet** — how your tokenizer's integer ids map onto the 20 canonical
   amino acids, plus the id of `<mask>`/`<pad>`/`<bos>`/`<eos>`.
2. **Extractor closures** — `hidden_fn(module, input_ids, padding_mask) -> (B,L,D)`
   and (for exact/targets) `logits_fn(...) -> (B,L,V)`.
3. **A trained `OFSHead`** for that encoder (for the OFS path only).

---

## Step 1 — build the `Alphabet`

If your encoder has a HuggingFace tokenizer:

```python
from ofsperplexity import Alphabet
alphabet = Alphabet.from_hf_tokenizer(tokenizer)
```

If you only have raw maps (say your training pipeline has its own tokenizer):

```python
alphabet = Alphabet.from_token_maps(
    token_to_id={"A": 5, "C": 23, ...},   # every single-letter amino acid
    mask_token_id=32, pad_token_id=1,
    vocab_size=64, bos_token_id=0, eos_token_id=2,
)
```

The alphabet is what tells the scorer which positions are real amino acids (the
ones that enter the PLL mean) and which token columns to read for the 20-way
softmax. It is derived from the tokenizer at runtime — nothing is hard-coded.

---

## Step 2 — wrap your encoder in a `ModuleAdapter`

```python
import torch
from ofsperplexity import ModuleAdapter

# Your frozen encoder in eval mode. It can be the SAME weights you are training,
# or a separate reference encoder -- OFS just needs a forward pass.
encoder = my_encoder.eval()

def hidden_fn(module, input_ids, padding_mask):
    # Return the LAST-LAYER per-residue embeddings, shape (B, L, D).
    out = module(input_ids, attention_mask=padding_mask, output_hidden_states=True)
    return out.hidden_states[-1]

def logits_fn(module, input_ids, padding_mask):
    # Return full-vocab MLM logits, shape (B, L, V). Needed for exact/targets only.
    return module(input_ids, attention_mask=padding_mask).logits

adapter = ModuleAdapter(
    encoder, alphabet,
    hidden_fn=hidden_fn,
    logits_fn=logits_fn,        # omit if you only ever use OFS
    hidden_size=encoder.config.hidden_size,
    no_grad=True,               # scoring must not perturb the graph of the model you train
)
```

`no_grad=True` (default) wraps every forward in `torch.no_grad()` so scoring never
touches autograd of your training step. Keep it on unless you specifically want
gradients (e.g. sequence design).

> **Attention-mask conventions live in the adapter.** Most HF encoders take a 1/0
> keep-mask as `attention_mask`. Some models (AMPLIFY) want an *additive* mask; put
> that conversion inside your `hidden_fn`/`logits_fn`. The scorers only ever hand
> you a boolean keep-mask.

---

## Step 3 — train an OFS head for this encoder (one time)

The head is the only trained piece and is **specific to one encoder's embedding
space**. Distil it from the exact one-at-a-time profiles on a set of representative
sequences (the paper uses Swissprot < 700 aa clustered at 50% identity):

```python
from ofsperplexity import generate_targets, train_ofs_head, TrainConfig

records = [("id1", "MSK..."), ("id2", "MKK..."), ...]      # representative set
targets = generate_targets(adapter, records, encode_fn)     # runs the exact calc
head, history = train_ofs_head(targets, adapter.hidden_size,
                               TrainConfig(num_members=8, epochs=40))
head.save("my_encoder_ofs_head.pt")
```

`encode_fn(seq) -> list[int]` turns an amino-acid string into token ids **with**
special tokens. For a HF tokenizer: `lambda s: tokenizer(s)["input_ids"]`.

Target generation is the expensive part (it *is* the one-at-a-time calc); it is
embarrassingly parallel over sequences — see the sharding note below, or use
`ofs-pppl gen-targets` under `mpiexec` and pass the shards to `ofs-pppl train
--targets shard.*.pt`. Fitting the head itself is seconds-to-minutes on one device.

If the encoder you are training **changes** the embedding space you want to score
against, retrain (or periodically refresh) the head. If you score against a
*frozen reference* encoder, train the head once and reuse it for the whole run.

---

## Step 4 — score inside your eval loop

```python
from ofsperplexity import OFSScorer, OFSHead, collate

scorer = OFSScorer(adapter, OFSHead.load("my_encoder_ofs_head.pt"))

@torch.no_grad()
def pseudo_perplexity(sequences):
    batch = collate([(f"s{i}", s) for i, s in enumerate(sequences)],
                    encode_fn, adapter.alphabet)
    res = scorer.score(batch)               # ONE forward pass through the encoder
    return res.pseudo_perplexity            # (B,) tensor; lower = more natural

# e.g. every N steps, log the mean OFS pseudo-perplexity of a validation set or of
# your model's own generated samples:
if step % eval_every == 0:
    pp = pseudo_perplexity(val_sequences)
    logger.log({"ofs_pppl/mean": pp.mean().item()}, step=step)
```

That is the whole integration. A complete runnable version is in
[`../examples/integrate_training_loop.py`](../examples/integrate_training_loop.py).

---

## Common variations

**I only want the fast path and never mask.** Provide only `hidden_fn`. You still
need a trained head (train it once with `logits_fn` present, or on another machine,
then drop `logits_fn`).

**I don't have an `nn.Module`, just functions / a remote service.** Use
`CallableAdapter(alphabet, hidden_size=..., device=..., hidden_fn=..., logits_fn=...)`
where the fns take `(input_ids, padding_mask)`.

**My model already emits logits during training and I just want exact PP on a few
sequences.** Skip the head entirely: build the adapter with a `logits_fn` and call
`exact_score(adapter, batch)`. Costs L passes but needs no training.

**Scoring generated samples for a design loop (paper Sec. VI).** Use
`scorer.profiles(batch)` to get the full `(B, L, 20)` single-substitution landscape
in one pass, and use it as the Metropolis proposal distribution with
`pseudo_perplexity` as the energy.

**Batching for throughput.** Use `iter_batches(records, encode_fn, alphabet,
max_tokens=16384)` — it length-sorts and packs a token budget per batch so padding
waste stays low. This matters most for exact scoring, where each masked replica is
a row.

---

## Sharding target generation / scoring across ranks

`ofsperplexity.dist` gives you MPI-first topology and strided shards:

```python
from ofsperplexity.dist import init_distributed, shard_indices, barrier

info = init_distributed("xpu", init_pg=False)     # rank/world/device; no collective backend
mine = [records[i] for i in shard_indices(len(records), info)]
targets = generate_targets(adapter, mine, encode_fn)
targets.save(f"targets.rank{info.rank:04d}.pt")
barrier(info)
```

Then `ofs-pppl train --targets targets.rank*.pt --head head.pt` concatenates them
and fits once. The CLI's `score` subcommand does the same sharding + a rank-0 merge
automatically.

---

## Sanity checks to run after wiring in

1. **Directionality** — a natural sequence should score *lower* pseudo-perplexity
   than a random permutation of it, which scores lower than an i.i.d.-random
   sequence. (Verified for ESMC in this repo: 17.9 < 20.1 < 22.3.)
2. **Exact vs OFS agreement** — on a handful of sequences, `OFSScorer.score` should
   track `exact_score` closely once the head is trained on a representative set
   (paper Fig. 2a: nearly identical). Large gaps mean the head has not seen
   sequences like your eval set — broaden the training distribution.
3. **`n_scored`** in the result should equal the number of canonical amino acids in
   each sequence (special tokens, gaps, and `X/B/U/Z/O` are excluded by design).

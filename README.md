# OFSperplexity

Fast, distributed **pseudo-perplexity** for protein language models, using the
**One Fell Swoop (OFS)** method of Kantroo, Wagner & Machta,
*"Pseudo-perplexity in one fell swoop for protein fitness estimation"*,
PRX Life **3**, 033014 (2025) — DOI [10.1103/zhx7-hcmm](https://doi.org/10.1103/zhx7-hcmm).

It works with **ESMC**, **AMPLIFY** (e.g. SaAMPLIFY-350M), and — the main design
goal — **any masked protein encoder you are training or fine-tuning**, as an
on-the-fly evaluation that is *independent of the model being trained*. Built to
run on **Aurora XPU** (Intel oneAPI), CUDA, and CPU.

> **New here / an agent wiring this into a model?** Read
> [`docs/INTEGRATION.md`](docs/INTEGRATION.md). It is the point of this repo.

---

## What is pseudo-perplexity, and what does "one fell swoop" buy us?

For a masked encoder, the **pseudo-log-likelihood** of a sequence `S` is

```
F_PLL(S) = (1/L) · Σ_i  ln P(r_i = r_i^S | S_{-i})          # paper Eq. 4
Z(S)     = exp(-F_PLL(S))     # pseudo-perplexity            # paper Eq. 5
```

where `P(·|S_{-i})` is the model's distribution at position `i` when position `i`
is masked, restricted to the 20 amino acids. **Lower pseudo-perplexity ⇒ more
natural sequence ⇒ higher fitness.** It is competitive with state-of-the-art
sequence-only fitness predictors and sets the state of the art on ProteinGym
**indels** (paper Tables I–II).

Computing it exactly needs **L forward passes** — mask each position in turn. OFS
replaces that with **one** forward pass: run the encoder on the *unmasked*
sequence, then project each residue's embedding through a tiny trained MLP ensemble
that estimates that residue's *masked* profile (paper Sec. III, Fig. 1). Speedup
grows linearly with length — ~100–1000× for real proteins (paper Fig. 1h).

This package gives you **both**:

| Path | Cost | Needs a trained head? | Use it for |
|------|------|-----------------------|------------|
| **Exact** (`exact_score`) | L fwd passes | no | ground truth, targets, short runs |
| **OFS** (`OFSScorer`) | **1 fwd pass** | yes (per encoder) | on-the-fly eval, MC design, scale |

The exact path also **generates the training targets** for the OFS head, so the two
are the same code end to end.

---

## Install

```bash
conda env create -f environment.yml
conda activate ofsperplexity
pip install -e .
```

On **Aurora**, get torch/IPEX/oneCCL from `module load frameworks` and install on
top without pulling a pip torch:

```bash
module load frameworks
pip install -e . --no-deps
pip install transformers safetensors mpi4py   # if not already present
```

The core library imports only `torch` + `numpy`. `transformers` is needed **only**
to load the reference encoders from disk (`ofsperplexity.models`).

---

## Quick start

### 1. Score a FASTA with the exact method (no training needed)

```bash
ofs-pppl score \
  --model-kind esmc --model-path /path/to/ESMC-300M \
  --method exact --input seqs.fasta --out pppl.tsv --device auto
```

### 2. Train an OFS head once, then score in a single pass

```bash
# Fit the head (distils the exact profile; ~7M params; cheap).
ofs-pppl train \
  --model-kind esmc --model-path /path/to/ESMC-300M \
  --input representative_seqs.fasta --head esmc_ofs_head.pt --device auto

# Score fast (one forward pass per sequence).
ofs-pppl score \
  --model-kind esmc --model-path /path/to/ESMC-300M \
  --method ofs --head esmc_ofs_head.pt --input seqs.fasta --out pppl.tsv
```

### 3. From Python

```python
from ofsperplexity.models import load_encoder
from ofsperplexity import OFSScorer, OFSHead, exact_score, collate

enc = load_encoder("esmc", "/path/to/ESMC-300M", device="auto")

# exact (ground truth)
batch = collate([("gfp", "MSKGEELFTG...")], enc.encode_fn, enc.alphabet)
print(exact_score(enc.adapter, batch).pseudo_perplexity)

# OFS (single pass, needs a trained head)
scorer = OFSScorer(enc.adapter, OFSHead.load("esmc_ofs_head.pt"))
print(scorer.score(batch).pseudo_perplexity)
```

---

## Wire it into a model you are training

This is the primary use case and has its own guide:
[`docs/INTEGRATION.md`](docs/INTEGRATION.md). The one-paragraph version:

The library never assumes a model class — it talks to an **`EncoderAdapter`** that
returns per-residue hidden states (`(B,L,D)`) and/or full-vocab logits `(B,L,V)`.
Wrap the *frozen, eval-mode* encoder you want to score against in a
`ModuleAdapter`, build an `Alphabet` from its tokenizer, load a pretrained
`OFSHead`, and call `OFSScorer(...).score(batch)` inside your eval loop. It is a
single extra forward pass and is fully decoupled from the model under training.
A runnable template is in [`examples/integrate_training_loop.py`](examples/integrate_training_loop.py).

---

## Distributed / Aurora

Scoring a FASTA is embarrassingly parallel: under `mpiexec` each rank scores a
strided shard and rank 0 merges. `ofsperplexity/dist.py` mirrors
`mini-embed-filip/src/dist.py` — XPU-first device pick with per-tile pinning,
optional IPEX/oneCCL, MPI-first topology detection, and an `init_pg=False` mode for
sharding without a collective backend. Example PBS scripts (one rank per tile, 12
per node, `--pmi=pmix`, `ONEAPI_DEVICE_SELECTOR=level_zero:gpu`) are in
[`scripts/`](scripts/).

```bash
mpiexec -n 24 -ppn 12 --pmi=pmix \
  python -m ofsperplexity.cli score --device xpu \
    --model-kind amplify --model-path $MODEL --method ofs --head head.pt \
    --input big.fasta --out big.pppl.tsv
```

---

## Repository layout

```
ofsperplexity/
  alphabet.py    Canonical 20-AA alphabet <-> a model's token vocab
  adapters.py    EncoderAdapter seam: HF / nn.Module / callable wrappers
  scoring.py     PLL + pseudo-perplexity math (shared by both scorers)
  exact.py       One-at-a-time scorer (ground truth + target generator)
  head.py        OFSHead: the MLP ensemble (the only trained component)
  ofs.py         OFSScorer: single-pass scoring (encoder + head)
  data.py        FASTA I/O, tokenization wrapper, length-bucketed batching
  train.py       Target generation + head distillation
  dist.py        Device + distributed bootstrap (Aurora XPU / CUDA / CPU)
  cli.py         `ofs-pppl {score,gen-targets,train}`
  models/        Reference loaders: esmc.py, amplify.py (xformers shim + XPU patch)
docs/            INTEGRATION.md (wire-in guide), METHOD.md (design notes)
examples/        integrate_training_loop.py, minimal_score.py
scripts/         Aurora PBS: score_fasta.pbs, gen_targets.pbs, train_head.pbs
tests/           test_smoke.py
```

See [`docs/METHOD.md`](docs/METHOD.md) for how each design decision maps to the
paper, and the important caveats (per-encoder heads, alphabet restriction, what the
head does and does not approximate).

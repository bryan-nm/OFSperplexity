# Method notes & design decisions

How this implementation maps to Kantroo, Wagner & Machta, *PRX Life* **3**, 033014
(2025), and the choices made where the paper leaves room.

## The two quantities

* **Exact pseudo-perplexity** (`ofsperplexity/exact.py`) — the reference. For each
  scorable position `i`, replace token `i` with `<mask>`, run the encoder, softmax
  the 20 amino-acid logits at position `i`, read the true residue's probability
  (paper Eq. 4, Fig. 1c). `Z = exp(-mean_i ln P)`. Cost: `L` forward passes.

* **OFS pseudo-perplexity** (`ofsperplexity/ofs.py`) — the approximation. Run the
  encoder once on the *unmasked* sequence; for each residue, map its last-layer
  embedding `E_i` through a trained ensemble `f` to estimate the masked profile
  `f(E_i) ≈ P(r_i | S_{-i})` (paper Eq. 6, Fig. 1d). Cost: 1 forward pass.

Both funnel through the same math in `ofsperplexity/scoring.py`, so a head trained
against exact targets is scored on exactly the same footing.

## The 20-amino-acid restriction

The probability vector at each position is a softmax over **only** the 20 canonical
amino-acid logits (paper Appendix 1: *"the softmax operation over the logits for the
20 natural amino acids"*). `Alphabet.gather_aa_logits` selects those 20 columns from
the encoder's full-vocab logits; the head outputs 20 logits directly. Positions
whose true residue is not one of the 20 (gaps, `X/B/U/Z/O`, `*`, special tokens) are
**excluded from the PLL mean** — they get no canonical index (`ofsperplexity/
alphabet.py`, `NON_SCORED_RESIDUES` and `scorable_mask`). This keeps exact and OFS
in the same 20-dim space and makes the head portable across checkpoints of the same
encoder.

The canonical order is fixed (`CANONICAL_AA = "ACDEFGHIKLMNPQRSTVWY"`). Order within
the 20 is arbitrary but must be consistent between target generation and scoring;
fixing it also makes a saved head interpretable and reusable.

## The head (`ofsperplexity/head.py`)

Paper Appendix 1: *"an ensemble of eight multilayer perceptron models that use ReLU
activations and layer normalization. The input for each MLP is 1280-dimensional
followed by layers that progressively reduce the dimensions: [1280, 557, 243, 106,
46, 20]."* ~7M params total.

* **Ensemble of 8** MLPs, each `(Linear → LayerNorm → ReLU) × k → Linear(→20)`.
* **Generalised taper.** The paper's widths are specific to `D = 1280` (ESM2-650M).
  `default_hidden_dims(D)` uses a geometric interpolation from `D` to `20` over 5
  layers, which **reproduces `[557, 243, 106, 46]` exactly at `D = 1280`** and gives
  a sensible taper for ESMC/AMPLIFY (`D = 960`). Override with `hidden_dims=[...]`.
* **Ensemble prediction** is the mean of member softmaxes (mean of probabilities,
  not logits), matching "ensemble prediction" in Fig. 1f. `pll_from_profiles`
  therefore takes `log` of the already-normalised mean profile rather than
  re-softmaxing.
* **Diversity** comes from independent random init + per-member shuffling, as in the
  paper (which attributes ensemble heterogeneity to init; see Fig. 1f KL diagnostic,
  exposed here as `OFSHead.disagreement`).

## Training (`ofsperplexity/train.py`)

* **Targets** = exact one-at-a-time profiles at scorable positions, paired with the
  unmasked embeddings from a single pass. Soft targets (full 20-way distributions),
  not one-hot.
* **Loss** = cross-entropy between the member prediction and the exact profile
  (`-Σ_a target_a · log_softmax(pred)_a`), summed over members. This is the
  distillation objective of Appendix 1.
* **Optimizer** AdamW; **split** 99:1 train/val (paper's ratio). Defaults:
  40 epochs, lr 1e-3, wd 1e-2 — tune per encoder.
* **Data.** The paper filters Swissprot to < 700 aa and clusters at 50% identity via
  mmseqs2, and withholds anything > 50% identical to ProteinGym. We don't ship a
  clustering step; feed `train` / `gen-targets` a representative FASTA you have
  already curated. Breadth of this set is what determines how well OFS generalises
  (see caveats).

## Adapters (`ofsperplexity/adapters.py`)

The library never imports `transformers` in its core and never assumes a model
class. Everything goes through `EncoderAdapter` (hidden states + logits + a bit of
metadata). This is what makes the "score a model you are training" use case a
three-line wrap rather than a fork. See `docs/INTEGRATION.md`.

## Reference encoders (`ofsperplexity/models/`)

* **ESMC** — a standard HF `*ForMaskedLM`; loads and exposes `hidden_states[-1]`
  directly. Needs a `transformers` that recognises `model_type: "esmc"` (the
  ESMC-300M card targets 4.57.6; note a *newer* transformers 5.x in the lab dropped
  native `esmc` support — pin ~4.57 for ESMC).
* **AMPLIFY** — the `trust_remote_code` modeling file imports `xformers` at module
  load even though its CPU/XPU path uses `F.scaled_dot_product_attention`. We inject
  a pure-torch **xformers shim** (fused `SwiGLU` with `w12`/`w3` submodules matching
  the checkpoint keys, plus an SDPA-based `memory_efficient_attention`) before load,
  and on XPU apply the same **manual-SDPA monkeypatch** as `mini-embed-filip`
  (Aurora's IPEX build has no fused attention kernel and segfaults past ~512 tokens).

## Distributed / Aurora (`ofsperplexity/dist.py`)

Mirrors `mini-embed-filip/src/dist.py`: XPU→CUDA→CPU device pick with per-tile
pinning, optional IPEX / oneCCL imports, MPI-first topology (the env-var path can
report `world=1` under Aurora MPICH), and an `init_pg=False` mode for
embarrassingly-parallel sharding with only an MPI barrier. Scoring and target
generation shard by strided index; head fitting runs on one rank.

## Caveats & non-goals

* **The head is per-encoder.** It maps *this* encoder's embedding space to masked
  profiles. A head trained for ESMC-300M is meaningless for AMPLIFY, and a head
  trained for one checkpoint drifts if you keep training that encoder. Score against
  a frozen reference, or refresh the head.
* **OFS estimates the encoder's profile, not the exact value.** As the paper notes,
  OFS pseudo-perplexity is an *estimate* of the exact pseudo-perplexity; expect a
  small, systematic gap. On the substitutions benchmark, the masked-marginal
  heuristic slightly outperforms pseudo-perplexity — OFS's decisive win is on
  **indels** (Table II), which decoder log-likelihoods handle but the masked-marginal
  substitution heuristic cannot.
* **Generalisation is bounded by the training distribution.** If your eval
  sequences look nothing like the head's training set, the single-pass estimate
  degrades (visible in this repo's tiny-head demos). Broaden the target set; the
  exact path is always available as a fallback and a check.
* **No mmseqs2 clustering / no bundled weights.** You bring the encoder and a
  representative FASTA. Everything else is here.

# Orientation for agents

This repo computes **protein pseudo-perplexity** two ways: an **exact** one-at-a-time
calculation (ground truth, `L` forward passes, no training) and a **One Fell Swoop
(OFS)** single-pass estimate via a trained MLP-ensemble head. It targets ESMC,
AMPLIFY, and any masked encoder — including a model under training. Aurora XPU /
CUDA / CPU. Based on Kantroo, Wagner & Machta, *PRX Life* 3, 033014 (2025).

## If your task is "add pseudo-perplexity eval to model X"

Read [`docs/INTEGRATION.md`](docs/INTEGRATION.md) — it is written for you. Summary:

1. Build an `Alphabet` from X's tokenizer (`Alphabet.from_hf_tokenizer` or
   `.from_token_maps`).
2. Wrap X's **frozen, eval** encoder in a `ModuleAdapter` with a `hidden_fn`
   (returns `(B,L,D)` last-layer embeddings) and, for exact/targets, a `logits_fn`
   (returns `(B,L,V)`).
3. Train an `OFSHead` once for X (`generate_targets` → `train_ofs_head`) and save it.
4. In the eval loop: `OFSScorer(adapter, head).score(batch).pseudo_perplexity`.

`examples/integrate_training_loop.py` is a runnable end-to-end template.

## Where things live

| You want to… | Look at |
|---|---|
| the fitness math (PLL, pseudo-perplexity) | `ofsperplexity/scoring.py` |
| the ground-truth scorer / target generator | `ofsperplexity/exact.py` |
| the trained head (architecture, ensemble) | `ofsperplexity/head.py` |
| single-pass scoring | `ofsperplexity/ofs.py` |
| the model-agnostic seam | `ofsperplexity/adapters.py` |
| token↔amino-acid mapping | `ofsperplexity/alphabet.py` |
| FASTA I/O, batching | `ofsperplexity/data.py` |
| curate raw UniRef90 -> training FASTA | `scripts/preprocess_uniref.py` |
| device + distributed (Aurora) | `ofsperplexity/dist.py` |
| the reference encoder loaders | `ofsperplexity/models/{esmc,amplify}.py` |
| CLI | `ofsperplexity/cli.py` (`ofs-pppl {score,gen-targets,train}`) |
| design rationale & caveats | `docs/METHOD.md` |
| sizing head-training jobs on Aurora | `docs/SIZING.md` |

## Invariants to preserve

* Scoring is restricted to the **20 canonical amino acids**, fixed order
  `ACDEFGHIKLMNPQRSTVWY`; non-canonical residues and special tokens are excluded
  from the PLL mean. Exact and OFS must stay in this same 20-dim space.
* The **head is per-encoder** (and drifts if that encoder keeps training). Score
  against a frozen reference or refresh the head.
* The core library must not `import transformers` — only `ofsperplexity/models/`
  may. Keep the adapter seam clean.

## Test / verify

No pytest in the lab envs; run the smoke tests directly:

```bash
PYTHONPATH=. python - <<'PY'
import tests.test_smoke as T
T.test_alphabet_roundtrip(); T.test_exact_score_runs()
T.test_ofs_matches_exact_after_distillation()
print("ok")
PY
```

Torch lives in the conda envs `filip_guide` (transformers 4.57.6 — use this for
**ESMC**), `mini-embed-filip`, and `mini-embed`. Real models are at
`/Users/bryan/Documents/models/{ESMC-300M,SaAMPLIFY_350M}`.

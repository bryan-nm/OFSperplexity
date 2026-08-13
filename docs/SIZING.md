# Sizing the head-training pipeline (UniRef90 on Aurora)

Training an OFS head has two stages with very different costs:

1. **`gen-targets`** — the exact one-at-a-time calculation over a representative
   sequence set. **This is essentially all the cost.** It is `L` masked forward
   passes per sequence, so per-sequence work scales as **L²**.
2. **`train`** — fitting the ~7M-parameter MLP ensemble on the resulting
   (embedding, profile) rows. Minutes on one tile.

## Do not train on all of UniRef90

UniRef90 is ~190M sequences. The head is 7M parameters doing an *easy* regression
(the paper's whole premise is that the unmasked embedding already encodes the masked
profile). You need enough **position-examples** to fit it, not the whole database.
The paper used Swissprot filtered to < 700 aa and clustered at 50% identity —
order 10⁵ representative sequences, i.e. tens of millions of positions. That is the
target scale.

**Recommended data prep:** take UniRef90 representatives, filter to ≤ 700 aa (ideally
≤ 512 — see the L² note), and **randomly subsample** to the tier you want below.
(UniRef90 is already dereplicated at 90%; further mmseqs2 clustering at 50% is
optional and only reduces redundancy.) Shuffle the FASTA so length is spread across
shards.

## Cost model (validated)

Per-sequence cost ≈ `2·P·L²` FLOPs (P = encoder params, L = length). Measured on
ESMC-300M (CPU, 8 threads): L=100 → 5.9 s, L=160 → 14.7 s — the 2.49× ratio matches
(160/100)² = 2.56, confirming the L² model. Both ESMC-300M and SaAMPLIFY-350M have
P ≈ 3.3×10⁸, so their costs are within ~10% of each other.

**Aurora planning throughput** (state your assumptions, then calibrate — see below):

| assumption | value |
|---|---|
| encoder params `P` | 3.3×10⁸ |
| mean length `L` of the rep. set (≤700 filter) | ~300 |
| sustained per-tile compute (bf16, ~25% of peak) | **50 TFLOP/s** |
| tiles per node | 12 |
| overhead factor (I/O, load, imbalance) | 0.7 |

⇒ ~1.2 s/seq/tile at L=300 ⇒ **~25,000 sequences / node-hour** (both models similar).

## Sizing table (per model, gen-targets)

| Tier | #seqs | ~positions | node-hours | suggested job | walltime |
|------|------:|-----------:|-----------:|---------------|----------|
| Minimal | 25 K | ~7.5 M | 1 | `select=2` | 00:45 |
| **Recommended** | 100 K | ~30 M | 4 | `select=8` | 01:00 |
| Generous | 500 K | ~150 M | 20 | `select=16` | 02:00 |
| Excessive | 5 M | ~1.5 B | 200 | `select=64` | 04:30 |

Run **once per encoder** (ESMC and AMPLIFY are separate heads → separate jobs, or
double the allocation). The **Recommended** 100 K tier matches the paper's data scale
and is the right default; go to Generous only if your eval sequences are far from
the training distribution and OFS is under-tracking exact (see `docs/METHOD.md`).

### The L² lever

Because cost is `L²`, a few near-700-aa sequences dominate and create stragglers.
`--max-len 512` instead of 700 roughly halves worst-case per-seq cost with little
effect on the head (most proteins are shorter). If your corpus skews long, either
lower `--max-len` or scale nodes by `(mean_L/300)²`.

## Storage (fp16 targets — the default)

Each position stores a 960-d embedding + 20-d profile + index ≈ **1.9 KB** (fp16;
fp32 doubles it). So:

| positions | fp16 size |
|----------:|----------:|
| 7.5 M | ~14 GB |
| 30 M | ~58 GB |
| 150 M | ~290 GB |

`gen-targets` writes one shard per rank (`<out>.rankNNNN.pt`); keep them on
`flare_fs`.

## Training the head

Cheap and single-tile. The trainer streams mini-batches from host RAM (it does **not**
load all embeddings onto the device), so it scales to large sets — but `TargetSet`
concatenation still holds the rows you pass in **host RAM**. Two rules:

* Keep the concatenated target set within node RAM (Aurora nodes have ≥512 GB, so the
  Recommended 58 GB tier is fine; for the Generous/Excessive tiers pass a **subset**
  of shard files to `--targets`, or use `--max-positions`).
* `--max-positions 15000000` (≈15 M rows) is plenty for a 7M-param head and bounds
  both RAM and epoch time. Fitting ~15 M rows × 40 epochs ≈ **10–15 min on one tile**.

So `train_head.pbs` is a `select=1`, `walltime=00:30:00` debug-queue job regardless of
tier.

## Calibrate before the big job (recommended)

The 50 TFLOP/s/tile assumption is the biggest unknown. Measure it on **one tile** with
a small sample, then scale:

```bash
# one tile, ~500 sequences from your prepped FASTA
python -m ofsperplexity.cli gen-targets --device xpu \
    --model-kind amplify --model-path $MODEL \
    --input <(head -n 1000 prepped.fasta) --out /tmp/cal.pt --max-len 512
# note the wall time T for N sequences -> seq/s/tile = N/T
# node-hours(total) = total_seqs / (seq/s/tile * 12 tiles * 3600 * 0.7)
```

Then set `select` and `walltime` in `gen_targets.pbs` from the measured rate. This
turns the estimate above into a number you can trust for your corpus and machine.

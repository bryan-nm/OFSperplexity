"""Command-line interface: score a FASTA, generate distillation targets, train a head.

Subcommands
-----------
* ``ofs-pppl score``       -- pseudo-perplexity for a FASTA (exact or OFS).
* ``ofs-pppl gen-targets`` -- exact one-at-a-time distillation targets (shardable).
* ``ofs-pppl train``       -- fit an OFS head from targets (or straight from a FASTA).

All subcommands are Aurora-aware: under ``mpiexec`` the sequence set is sharded
across ranks (strided) and each rank writes its own output; ``score`` merges on
rank 0 if ``--out`` is given. See ``scripts/*.pbs``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional, Tuple

import torch

from .data import read_sequences, iter_batches
from .dist import DistInfo, barrier, init_distributed, shard_indices


def _add_common_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model-kind", choices=["esmc", "amplify"], required=True)
    p.add_argument("--model-path", required=True, help="Path to the encoder folder.")
    p.add_argument("--device", default="auto", help="auto|cpu|xpu|cuda|mps|cuda:0 ...")
    p.add_argument("--dtype", default=None, choices=[None, "float32", "bfloat16", "float16"])


def _load_encoder(args, info: DistInfo):
    from .models import load_encoder

    dtype = None if args.dtype in (None, "None") else getattr(torch, args.dtype)
    return load_encoder(
        args.model_kind,
        args.model_path,
        device=args.device,
        local_rank=info.local_rank,
        dtype=dtype,
    )


# --------------------------------------------------------------------- score
def cmd_score(args) -> None:
    info = init_distributed(args.device, init_pg=False)
    enc = _load_encoder(args, info)
    records = read_sequences(args.input)
    mine = [records[i] for i in shard_indices(len(records), info)]

    rows: List[Tuple[str, float, float, int]] = []
    if args.method == "ofs":
        from .head import OFSHead
        from .ofs import OFSScorer

        head = OFSHead.load(args.head, map_location=enc.adapter.device)
        scorer = OFSScorer(enc.adapter, head)
        for batch in iter_batches(mine, enc.encode_fn, enc.alphabet,
                                  batch_size=args.batch_size, max_tokens=args.max_tokens):
            res = scorer.score(batch)
            rows.extend(_rows(batch.ids, res))
    else:
        from .exact import exact_score

        for batch in iter_batches(mine, enc.encode_fn, enc.alphabet,
                                  batch_size=args.batch_size or 1, max_tokens=args.max_tokens):
            res = exact_score(enc.adapter, batch, max_forward_tokens=args.max_forward_tokens)
            rows.extend(_rows(batch.ids, res))

    _write_rows(rows, args, info)


def _rows(ids, res) -> List[Tuple[str, float, float, int]]:
    return list(zip(ids, res.pseudo_perplexity.tolist(), res.pll.tolist(), res.n_scored.tolist()))


def _write_rows(rows, args, info: DistInfo) -> None:
    header = "id\tpseudo_perplexity\tpll\tn_scored"
    if info.world_size > 1 and args.out:
        part = f"{args.out}.rank{info.rank:04d}"
        _dump_tsv(part, header, rows)
        barrier(info)
        if info.is_main:
            merged = []
            for r in range(info.world_size):
                p = f"{args.out}.rank{r:04d}"
                with open(p) as fh:
                    next(fh)  # skip header
                    merged.extend(fh.read().splitlines())
                os.remove(p)
            with open(args.out, "w") as fh:
                fh.write(header + "\n")
                fh.write("\n".join(merged) + "\n")
            print(f"[score] wrote {len(merged)} rows -> {args.out}", flush=True)
    elif args.out:
        _dump_tsv(args.out, header, rows)
        print(f"[score] wrote {len(rows)} rows -> {args.out}", flush=True)
    else:
        print(header)
        for name, pp, pll, n in rows:
            print(f"{name}\t{pp:.6f}\t{pll:.6f}\t{n}")


def _dump_tsv(path, header, rows) -> None:
    with open(path, "w") as fh:
        fh.write(header + "\n")
        for name, pp, pll, n in rows:
            fh.write(f"{name}\t{pp:.6f}\t{pll:.6f}\t{n}\n")


# ---------------------------------------------------------------- gen-targets
def cmd_gen_targets(args) -> None:
    from .train import generate_targets

    info = init_distributed(args.device, init_pg=False)
    enc = _load_encoder(args, info)
    records = read_sequences(args.input)
    if args.max_len:
        records = [r for r in records if len(r[1]) <= args.max_len]
    mine = [records[i] for i in shard_indices(len(records), info)]
    store_dtype = getattr(torch, args.store_dtype)
    ts = generate_targets(
        enc.adapter, mine, enc.encode_fn,
        max_forward_tokens=args.max_forward_tokens, store_dtype=store_dtype,
        progress=info.is_main,
    )
    out = args.out if info.world_size == 1 else f"{args.out}.rank{info.rank:04d}"
    ts.save(out)
    print(f"[gen-targets] rank {info.rank}: {len(ts)} rows -> {out}", flush=True)
    barrier(info)


# --------------------------------------------------------------------- train
def cmd_train(args) -> None:
    from .train import TargetSet, TrainConfig, generate_targets, train_ofs_head

    info = init_distributed(args.device, init_pg=False)
    if args.targets:
        parts = [TargetSet.load(p) for p in args.targets]
        ts = TargetSet.concat(parts) if len(parts) > 1 else parts[0]
        input_dim = ts.embeddings.shape[1]
    else:
        enc = _load_encoder(args, info)
        records = read_sequences(args.input)
        if args.max_len:
            records = [r for r in records if len(r[1]) <= args.max_len]
        ts = generate_targets(enc.adapter, records, enc.encode_fn,
                              max_forward_tokens=args.max_forward_tokens,
                              store_dtype=getattr(torch, args.store_dtype), progress=True)
        input_dim = enc.adapter.hidden_size

    cfg = TrainConfig(
        num_members=args.num_members, epochs=args.epochs, batch_size=args.train_batch_size,
        lr=args.lr, weight_decay=args.weight_decay, val_fraction=args.val_fraction,
        device=args.device, seed=args.seed, max_positions=args.max_positions,
    )
    head, history = train_ofs_head(ts, input_dim, cfg)
    head.save(args.head)
    with open(args.head + ".history.json", "w") as fh:
        json.dump(history, fh, indent=2)
    print(f"[train] saved head ({head.num_parameters/1e6:.2f}M params) -> {args.head}", flush=True)


# ----------------------------------------------------------------------- main
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="ofs-pppl", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser("score", help="Score a FASTA (exact or OFS).")
    _add_common_model_args(ps)
    ps.add_argument("--input", required=True, help="FASTA / one-seq-per-line file.")
    ps.add_argument("--method", choices=["ofs", "exact"], default="ofs")
    ps.add_argument("--head", help="OFS head checkpoint (required for --method ofs).")
    ps.add_argument("--out", default=None, help="Output TSV (stdout if omitted).")
    ps.add_argument("--batch-size", type=int, default=None)
    ps.add_argument("--max-tokens", type=int, default=16384)
    ps.add_argument("--max-forward-tokens", type=int, default=16384)
    ps.set_defaults(func=cmd_score)

    pg = sub.add_parser("gen-targets", help="Exact one-at-a-time distillation targets.")
    _add_common_model_args(pg)
    pg.add_argument("--input", required=True)
    pg.add_argument("--out", required=True, help="Output .pt (per-rank .rankNNNN under MPI).")
    pg.add_argument("--max-len", type=int, default=700, help="Skip sequences longer than this.")
    pg.add_argument("--max-forward-tokens", type=int, default=16384)
    pg.add_argument("--store-dtype", default="float16", choices=["float16", "bfloat16", "float32"],
                    help="dtype for the stored embeddings/profiles (fp16 halves the target footprint).")
    pg.set_defaults(func=cmd_gen_targets)

    pt = sub.add_parser("train", help="Fit an OFS head from targets or a FASTA.")
    pt.add_argument("--targets", nargs="*", help="One or more TargetSet .pt files.")
    pt.add_argument("--input", help="FASTA (used if --targets not given).")
    pt.add_argument("--model-kind", choices=["esmc", "amplify"])
    pt.add_argument("--model-path")
    pt.add_argument("--device", default="auto")
    pt.add_argument("--dtype", default=None)
    pt.add_argument("--head", required=True, help="Output head checkpoint path.")
    pt.add_argument("--max-len", type=int, default=700)
    pt.add_argument("--max-forward-tokens", type=int, default=16384)
    pt.add_argument("--store-dtype", default="float16", choices=["float16", "bfloat16", "float32"],
                    help="dtype for inline-generated targets (ignored when --targets is given).")
    pt.add_argument("--max-positions", type=int, default=None,
                    help="Subsample this many distillation rows before fitting (bounds RAM/time).")
    pt.add_argument("--num-members", type=int, default=8)
    pt.add_argument("--epochs", type=int, default=40)
    pt.add_argument("--train-batch-size", type=int, default=4096)
    pt.add_argument("--lr", type=float, default=1e-3)
    pt.add_argument("--weight-decay", type=float, default=1e-2)
    pt.add_argument("--val-fraction", type=float, default=0.01)
    pt.add_argument("--seed", type=int, default=0)
    pt.set_defaults(func=cmd_train)
    return ap


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    if args.cmd == "score" and args.method == "ofs" and not args.head:
        print("error: --head is required for --method ofs", file=sys.stderr)
        sys.exit(2)
    args.func(args)


if __name__ == "__main__":
    main()

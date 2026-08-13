"""Device selection and distributed bootstrap (Aurora XPU / CUDA / CPU).

This mirrors the conventions in ``mini-embed-filip/src/dist.py`` so OFS jobs drop
into the same Aurora setup:

* device priority **explicit request -> XPU -> CUDA -> CPU**, with per-rank tile
  pinning via ``local_rank % device_count``;
* Intel Extension for PyTorch (IPEX) and oneCCL bindings imported *optionally* so
  the module still loads on a laptop;
* topology read from **MPI first** (ground truth under ``mpiexec`` on Aurora's
  MPICH, where the env-var path can silently report world=1), falling back to
  PALS / PMI / torch launcher env vars;
* an ``init_pg=False`` mode for embarrassingly-parallel jobs (FASTA scoring) that
  only need rank/world for sharding plus an MPI barrier -- no collective backend.

Scoring a FASTA is embarrassingly parallel: each rank scores a disjoint shard
(:func:`shard_indices`) and writes its own output file; a downstream merge (or the
CLI's rank-0 gather) stitches them. Training the OFS head can use ``init_pg=True``
with the ``xccl``/``ccl`` backend if you want data-parallel gradient averaging.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch

# Optional Intel stacks -- present on Aurora via `module load frameworks`, absent
# on a laptop. Importing IPEX enables XPU; importing the oneCCL bindings registers
# the legacy "ccl" backend for torch.distributed.
try:  # noqa: SIM105
    import intel_extension_for_pytorch as _ipex  # noqa: F401
except Exception:  # pragma: no cover
    _ipex = None
try:
    import oneccl_bindings_for_pytorch  # noqa: F401
except Exception:  # pragma: no cover
    pass


@dataclass
class DistInfo:
    rank: int
    world_size: int
    local_rank: int
    device: torch.device
    backend: Optional[str] = None
    pg_initialized: bool = False

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def _first_env(*names: str, default: int = 0) -> int:
    for n in names:
        v = os.environ.get(n)
        if v is not None and v != "":
            try:
                return int(v)
            except ValueError:
                pass
    return default


def _read_topology() -> Tuple[int, int, int]:
    """Return ``(rank, world_size, local_rank)``, querying MPI before env vars."""
    try:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        world = comm.Get_size()
        if world > 1:
            rank = comm.Get_rank()
            local = comm.Split_type(MPI.COMM_TYPE_SHARED).Get_rank()
            return rank, world, local
    except Exception:
        pass
    world = _first_env("PALS_NRANKS", "PMI_SIZE", "WORLD_SIZE", "OMPI_COMM_WORLD_SIZE", default=1)
    rank = _first_env("PALS_RANKID", "PMI_RANK", "RANK", "OMPI_COMM_WORLD_RANK", default=0)
    local = _first_env(
        "PALS_LOCAL_RANKID", "MPI_LOCALRANKID", "PMI_LOCAL_RANK",
        "LOCAL_RANK", "OMPI_COMM_WORLD_LOCAL_RANK", default=0,
    )
    return rank, world, local


def pick_device(device_name: str = "auto", local_rank: int = 0) -> torch.device:
    """Select and (for accelerators) pin the device for this rank.

    ``device_name`` in ``{"auto", "xpu", "cuda", "cpu", "mps", "cuda:0", ...}``.
    A non-auto, non-xpu request is honoured verbatim (smoke tests).
    """
    if device_name not in ("auto", "xpu") and not device_name.startswith("xpu"):
        return torch.device(device_name)
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        n = max(torch.xpu.device_count(), 1)
        idx = local_rank % n
        torch.xpu.set_device(idx)
        return torch.device(f"xpu:{idx}")
    if torch.cuda.is_available():
        n = max(torch.cuda.device_count(), 1)
        idx = local_rank % n
        torch.cuda.set_device(idx)
        return torch.device(f"cuda:{idx}")
    if device_name == "mps" or (device_name == "auto" and getattr(torch.backends, "mps", None)
                                and torch.backends.mps.is_available()):
        return torch.device("mps")
    return torch.device("cpu")


def _pick_backend(device: torch.device) -> str:
    override = os.environ.get("OFS_DIST_BACKEND")
    if override:
        return override
    if device.type == "xpu":
        # PyTorch >=2.6 on the `frameworks` module ships native xccl; older builds
        # need the oneccl_bindings "ccl" backend (imported above).
        if hasattr(torch.distributed, "is_xccl_available") and torch.distributed.is_xccl_available():
            return "xccl"
        return "ccl"
    if device.type == "cuda":
        return "nccl"
    return "gloo"


def init_distributed(
    device_name: str = "auto",
    *,
    init_pg: bool = False,
) -> DistInfo:
    """Bootstrap rank/world/device (and optionally a process group).

    Set ``init_pg=False`` (default) for FASTA scoring: each rank just needs its
    identity to pick a shard; synchronisation is an MPI barrier. Set
    ``init_pg=True`` to also start a collective backend for data-parallel training.
    """
    rank, world, local = _read_topology()
    device = pick_device(device_name, local)
    info = DistInfo(rank=rank, world_size=world, local_rank=local, device=device)
    if not init_pg or world <= 1:
        return info

    backend = _pick_backend(device)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world)
    os.environ["LOCAL_RANK"] = str(local)
    kwargs = dict(backend=backend, init_method="env://", world_size=world, rank=rank)
    try:
        torch.distributed.init_process_group(
            device_id=device if device.type in ("xpu", "cuda") else None, **kwargs
        )
    except TypeError:  # older torch without device_id kwarg
        torch.distributed.init_process_group(**kwargs)
    info.backend = backend
    info.pg_initialized = True
    return info


def barrier(info: DistInfo) -> None:
    """Synchronise ranks; falls back to an MPI barrier when no PG exists."""
    if info.world_size <= 1:
        return
    if info.pg_initialized and torch.distributed.is_initialized():
        torch.distributed.barrier()
        return
    try:
        from mpi4py import MPI

        MPI.COMM_WORLD.Barrier()
    except Exception:
        pass


def cleanup(info: DistInfo) -> None:
    if info.pg_initialized and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def shard_indices(n_items: int, info: DistInfo) -> List[int]:
    """Indices assigned to this rank (strided, so shards are length-balanced)."""
    return list(range(info.rank, n_items, info.world_size))


def autocast_ctx(device: torch.device, dtype: torch.dtype = torch.bfloat16):
    """bf16 autocast on XPU/CUDA, a no-op on CPU (matches mini-embed-filip)."""
    if device.type in ("cuda", "xpu"):
        return torch.autocast(device_type=device.type, dtype=dtype)
    return torch.autocast(device_type="cpu", enabled=False)

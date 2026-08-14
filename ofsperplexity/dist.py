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
import sys
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch

# Optional Intel stacks -- present on Aurora via `module load frameworks`, absent
# on a laptop. Importing IPEX enables XPU; importing the oneCCL bindings registers
# the legacy "ccl" backend for torch.distributed.
try:  # noqa: SIM105  -- enables the XPU device; needed for any XPU work
    import intel_extension_for_pytorch as _ipex  # noqa: F401
except Exception:  # pragma: no cover
    _ipex = None


def _load_oneccl() -> None:
    """Register the oneCCL ``ccl`` backend. Imported lazily -- ONLY when a collective
    process group is actually initialised -- so the embarrassingly-parallel jobs
    never load a fabric-touching component."""
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


# Per-rank id vars set by various launchers. PMIX_RANK is set by PALS when launched
# with --pmi=pmix even though PALS_RANKID sometimes is not; OFS_* are set by our own
# PBS scripts from the topology they already know (NRANKS / RANKS_PER_NODE).
_RANK_ENV = ("PALS_RANKID", "PMIX_RANK", "PMI_RANK", "RANK", "OMPI_COMM_WORLD_RANK")
_WORLD_ENV = ("PALS_NRANKS", "OFS_WORLD_SIZE", "PMIX_SIZE", "PMI_SIZE",
              "WORLD_SIZE", "OMPI_COMM_WORLD_SIZE")
_LOCAL_ENV = ("PALS_LOCAL_RANKID", "PMIX_LOCAL_RANK", "MPI_LOCALRANKID", "PMI_LOCAL_RANK",
              "LOCAL_RANK", "OMPI_COMM_WORLD_LOCAL_RANK")


def _read_topology(allow_mpi: bool = True) -> Tuple[int, int, int]:
    """Return ``(rank, world_size, local_rank)``.

    Prefers the **launcher environment variables** -- read with NO fabric or MPI
    initialisation, so it never touches the CXI NIC (essential for the
    embarrassingly-parallel jobs, which otherwise call ``MPI_Init`` via mpi4py purely
    to learn their rank and, at scale, hit "CXI alloc failed ... limits" at launch).

    Gated on a *per-rank* id being present (``_RANK_ENV``): a world-size var alone is
    not enough, since without a distinct rank every process would claim rank 0. When a
    rank id is present, world comes from ``_WORLD_ENV`` (PALS_NRANKS, or OFS_WORLD_SIZE
    exported by our PBS scripts) and local rank from ``_LOCAL_ENV`` or, failing that,
    ``rank % OFS_RANKS_PER_NODE``.

    Falls back to mpi4py (which opens the fabric) only when no rank env is present AND
    ``allow_mpi`` is set -- i.e. a real collective backend is wanted.
    """
    if any(k in os.environ for k in _RANK_ENV):
        rank = _first_env(*_RANK_ENV, default=0)
        world = _first_env(*_WORLD_ENV, default=1)
        local = _first_env(*_LOCAL_ENV, default=-1)
        if local < 0:
            rpn = _first_env("OFS_RANKS_PER_NODE", "PALS_LOCAL_SIZE", default=0)
            local = (rank % rpn) if rpn > 0 else 0
        return rank, world, local
    if allow_mpi:
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
    return 0, 1, 0


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
    # MPS is honoured only when *explicitly* requested -- Apple's fp16/LayerNorm
    # kernels are numerically flaky for the OFS head ensemble and can yield NaNs.
    # `auto` therefore falls through to CPU on a Mac (matches mini-embed-filip).
    if device_name == "mps":
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
    allow_mpi: Optional[bool] = None,
) -> DistInfo:
    """Bootstrap rank/world/device (and optionally a process group).

    Set ``init_pg=False`` (default) for FASTA scoring / target generation: each rank
    just needs its identity to pick a strided shard and writes its own output file --
    NO inter-rank communication, so nothing touches the fabric. Set ``init_pg=True``
    to also start a collective backend for data-parallel training.

    ``allow_mpi`` controls whether topology detection may fall back to ``mpi4py``
    (which calls ``MPI_Init`` and opens the CXI fabric). It defaults to ``init_pg``:
    the embarrassingly-parallel path never imports mpi4py and therefore cannot trip
    the "CXI alloc failed" launch error, while the collective path may. Launcher env
    vars (PALS_*) are always tried first regardless.
    """
    if allow_mpi is None:
        allow_mpi = init_pg
    rank, world, local = _read_topology(allow_mpi=allow_mpi)
    device = pick_device(device_name, local)
    info = DistInfo(rank=rank, world_size=world, local_rank=local, device=device)
    if not init_pg or world <= 1:
        return info

    if device.type == "xpu":
        _load_oneccl()  # register the ccl backend only now (collective path)
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
    """Synchronise ranks, if a synchronisation mechanism is already up.

    Uses the torch PG when initialised. Otherwise uses an MPI barrier ONLY if mpi4py
    is already imported and MPI initialised -- it will NOT import mpi4py (and open the
    fabric) just to barrier. For the embarrassingly-parallel jobs there is nothing to
    synchronise, so this is a no-op, which is correct: each rank writes its own file.
    """
    if info.world_size <= 1:
        return
    if info.pg_initialized and torch.distributed.is_initialized():
        torch.distributed.barrier()
        return
    mpi = sys.modules.get("mpi4py")
    if mpi is not None:
        try:
            from mpi4py import MPI

            if MPI.Is_initialized():
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

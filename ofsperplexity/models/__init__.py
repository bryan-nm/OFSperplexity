"""Reference encoder loaders (ESMC, AMPLIFY).

These are *conveniences* for the standalone FASTA-scoring path. The core library
does not depend on them -- to attach OFS to a model you are training, build an
adapter directly (see ``docs/INTEGRATION.md``). Importing this subpackage pulls in
``transformers``; install the ``models`` extra.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

from ..adapters import EncoderAdapter
from ..alphabet import Alphabet


@dataclass
class LoadedEncoder:
    """Everything the scorers need for one encoder."""

    adapter: EncoderAdapter
    alphabet: Alphabet
    encode_fn: Callable[[str], List[int]]
    tokenizer: object = None
    model: object = None

    @property
    def hidden_size(self) -> int:
        return self.adapter.hidden_size


def load_encoder(kind: str, path: str, **kwargs) -> LoadedEncoder:
    """Dispatch on ``kind`` in ``{"esmc", "amplify"}``."""
    kind = kind.lower()
    if kind == "esmc":
        from .esmc import load_esmc

        return load_esmc(path, **kwargs)
    if kind in ("amplify", "saamplify"):
        from .amplify import load_amplify

        return load_amplify(path, **kwargs)
    raise ValueError(f"Unknown encoder kind: {kind!r} (expected 'esmc' or 'amplify')")

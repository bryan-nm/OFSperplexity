"""Canonical amino-acid alphabet and the mapping to a model's token vocabulary.

The One Fell Swoop (OFS) scheme, like the masked-marginal / pseudo-log-likelihood
heuristics it approximates, restricts the probability vector at each position to
the 20 canonical amino acids and renormalises over just those (see the OFS paper,
Sec. II and Appendix 1: "The probability vector for each position is calculated
via the softmax operation over the logits for the 20 natural amino acids").

Everything downstream (exact one-at-a-time scoring, OFS head training, OFS head
inference) operates in this fixed 20-dimensional space, in a fixed order, so that
a head trained against exact targets scores identically at inference and so that a
head is portable across checkpoints of the *same* encoder.

`Alphabet` is the single object that knows how a particular encoder's integer
token ids relate to that 20-dim space, plus which token ids are structural
(pad / mask / bos / eos / etc.).  Build one with :meth:`Alphabet.from_hf_tokenizer`
for HuggingFace tokenizers (ESMC), :meth:`Alphabet.from_amplify_tokenizer` for the
AMPLIFY `ProteinTokenizer`, or :meth:`Alphabet.from_token_maps` for anything else.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

import torch

# Fixed canonical ordering. Do NOT reorder: OFS heads are trained against targets
# laid out in this order, and the head's 20 output logits are interpreted here.
CANONICAL_AA: str = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_INDEX: Dict[str, int] = {aa: i for i, aa in enumerate(CANONICAL_AA)}
NUM_AA: int = len(CANONICAL_AA)  # 20

# Residue letters that are valid FASTA characters but are NOT scored: ambiguous
# codes, gaps, and translation-stop. These positions are skipped in the PLL mean.
NON_SCORED_RESIDUES: frozenset = frozenset("BJOUXZ.-*")


@dataclass
class Alphabet:
    """Maps a model's token vocabulary onto the 20 canonical amino acids.

    Attributes
    ----------
    aa_token_ids:
        LongTensor ``(20,)``. ``aa_token_ids[k]`` is the model token id of the
        canonical amino acid ``CANONICAL_AA[k]``. Used to gather the 20 relevant
        columns out of a full-vocab logits tensor.
    mask_token_id, pad_token_id:
        Structural token ids (required). ``mask_token_id`` is used by the exact
        one-at-a-time scorer; ``pad_token_id`` marks padding in batches.
    bos_token_id, eos_token_id:
        Optional structural ids (cls/bos and eos). ``None`` if the encoder has none.
    special_token_ids:
        Every token id that must never be scored (pad/mask/bos/eos + any extras
        such as AMPLIFY's ``|`` chain separator or ESMC's ``X/B/U/Z/O/./-``).
    vocab_size:
        Size of the model's token vocabulary (width of its logits).
    """

    aa_token_ids: torch.Tensor
    mask_token_id: int
    pad_token_id: int
    vocab_size: int
    bos_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    special_token_ids: frozenset = field(default_factory=frozenset)

    # ------------------------------------------------------------------ derived
    def __post_init__(self) -> None:
        self.aa_token_ids = torch.as_tensor(self.aa_token_ids, dtype=torch.long)
        if self.aa_token_ids.shape != (NUM_AA,):
            raise ValueError(
                f"aa_token_ids must have shape ({NUM_AA},), got {tuple(self.aa_token_ids.shape)}"
            )
        # token id -> canonical aa index, with -1 for everything non-canonical.
        lut = torch.full((self.vocab_size,), -1, dtype=torch.long)
        lut[self.aa_token_ids] = torch.arange(NUM_AA, dtype=torch.long)
        self._token_to_aa_index = lut
        specials = set(self.special_token_ids)
        specials.add(self.mask_token_id)
        specials.add(self.pad_token_id)
        if self.bos_token_id is not None:
            specials.add(self.bos_token_id)
        if self.eos_token_id is not None:
            specials.add(self.eos_token_id)
        self.special_token_ids = frozenset(specials)

    # ------------------------------------------------------------- constructors
    @classmethod
    def from_token_maps(
        cls,
        token_to_id: Dict[str, int],
        *,
        mask_token_id: int,
        pad_token_id: int,
        vocab_size: int,
        bos_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        extra_special_ids: Iterable[int] = (),
    ) -> "Alphabet":
        """Build from a ``{token_str: id}`` dict (single-character AA tokens)."""
        try:
            aa_ids = [token_to_id[aa] for aa in CANONICAL_AA]
        except KeyError as e:  # pragma: no cover - defensive
            raise KeyError(
                f"Tokenizer vocabulary is missing canonical amino acid token {e}. "
                "OFS requires all 20 single-letter amino acids in the vocab."
            ) from e
        return cls(
            aa_token_ids=torch.tensor(aa_ids, dtype=torch.long),
            mask_token_id=mask_token_id,
            pad_token_id=pad_token_id,
            vocab_size=vocab_size,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            special_token_ids=frozenset(extra_special_ids),
        )

    @classmethod
    def from_hf_tokenizer(cls, tokenizer) -> "Alphabet":
        """Build from a HuggingFace tokenizer (e.g. the ESMC ``AutoTokenizer``).

        Uses ``convert_tokens_to_ids`` for the 20 amino acids and the tokenizer's
        declared special-token ids. Works for ESMC (mask=32, pad=1, cls=0, eos=2).
        """
        vocab_size = int(getattr(tokenizer, "vocab_size", None) or len(tokenizer))
        aa_ids = tokenizer.convert_tokens_to_ids(list(CANONICAL_AA))
        if any(i is None or i == getattr(tokenizer, "unk_token_id", -999) for i in aa_ids):
            raise KeyError("Tokenizer could not map all 20 amino acids to ids.")
        extra = set()
        # collect any remaining declared special ids so we never score them.
        for tid in getattr(tokenizer, "all_special_ids", []) or []:
            extra.add(int(tid))
        return cls(
            aa_token_ids=torch.tensor(aa_ids, dtype=torch.long),
            mask_token_id=int(tokenizer.mask_token_id),
            pad_token_id=int(tokenizer.pad_token_id),
            vocab_size=vocab_size,
            bos_token_id=_maybe_int(getattr(tokenizer, "cls_token_id", None)
                                    or getattr(tokenizer, "bos_token_id", None)),
            eos_token_id=_maybe_int(getattr(tokenizer, "eos_token_id", None)),
            special_token_ids=frozenset(int(i) for i in extra),
        )

    @classmethod
    def from_amplify_tokenizer(cls, tokenizer) -> "Alphabet":
        """Build from the AMPLIFY ``ProteinTokenizer`` (or the fast HF variant).

        AMPLIFY vocab: pad=0, unk=1, mask=2, bos=3, eos=4, ``|``=5, then the amino
        acids.  Handles both the repo's ``ProteinTokenizer`` (``token_to_id``) and
        a ``PreTrainedTokenizerFast`` loaded from the same folder.
        """
        if hasattr(tokenizer, "token_to_id") and callable(tokenizer.token_to_id):
            t2i = tokenizer.token_to_id
            aa_ids = [t2i(aa) for aa in CANONICAL_AA]
            return cls(
                aa_token_ids=torch.tensor(aa_ids, dtype=torch.long),
                mask_token_id=int(tokenizer.mask_token_id),
                pad_token_id=int(tokenizer.pad_token_id),
                vocab_size=len(tokenizer),
                bos_token_id=int(tokenizer.bos_token_id),
                eos_token_id=int(tokenizer.eos_token_id),
                special_token_ids=frozenset(int(i) for i in tokenizer.special_token_ids),
            )
        # Fast HF tokenizer loaded from the AMPLIFY folder.
        return cls.from_hf_tokenizer(tokenizer)

    # --------------------------------------------------------------- operations
    def to(self, device) -> "Alphabet":
        self.aa_token_ids = self.aa_token_ids.to(device)
        self._token_to_aa_index = self._token_to_aa_index.to(device)
        return self

    def gather_aa_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Select the 20 amino-acid columns from full-vocab ``logits``.

        Parameters
        ----------
        logits: ``(..., vocab_size)``
        Returns ``(..., 20)`` in canonical order.
        """
        idx = self.aa_token_ids.to(logits.device)
        return logits.index_select(-1, idx)

    def token_ids_to_aa_index(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Map token ids ``(...)`` to canonical aa index ``(...)`` (-1 if not an AA)."""
        lut = self._token_to_aa_index.to(token_ids.device)
        return lut[token_ids]

    def scorable_mask(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Boolean mask ``(...)``: True where the token is a canonical amino acid.

        This is exactly the set of positions that enter the PLL mean: it excludes
        pad/mask/bos/eos/separators *and* non-scored residues (X, B, U, Z, O, gaps)
        because those never receive a canonical aa index.
        """
        return self.token_ids_to_aa_index(token_ids) >= 0

    def encode_string(self, seq: str) -> List[int]:
        """Map a bare amino-acid string to canonical aa indices (for tests/targets).

        Non-canonical residues become -1.
        """
        return [AA_TO_INDEX.get(c, -1) for c in seq.upper()]


def _maybe_int(x) -> Optional[int]:
    return None if x is None else int(x)

"""Ternary hypervector encoding and bit-packed similarity search.

The encoder has no trained weights.  Every feature (word, word pair,
character trigram) is hashed with a keyed BLAKE2b to a handful of signed
positions in a D-dimensional vector; a text is the weighted sum of its
features, quantised to {-1, 0, +1}.  Two peers that share the swarm key
therefore share one vector space without exchanging a model, and somebody
without the key cannot map a vector back to words by dictionary attack.

A ternary vector is stored as two bit masks (positive, negative) packed into
uint64 words, so a similarity search is four AND + popcount passes.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

WORD_RE = re.compile(r"[a-z0-9_]+(?:'[a-z]+)?")

# Function words carry almost no topical signal; they are down-weighted, not
# dropped, so exact phrases still match.
STOPWORDS = frozenset(
    """a an and are as at be but by for from has have he her his i if in into is it its
    not of on or she so that the their them then there these they this to was we were
    what when which who will with you your can do does did been being than too very
    all any each no nor only own same such both more most other some up out about
    over under again once here why how where also may might must shall should would
    could just""".split()
)

FEATURE_NNZ = 32  # signed positions set by one feature


def tokenize(text: str) -> List[str]:
    return WORD_RE.findall(text.lower())


try:  # numpy >= 2.0
    _bitcount = np.bitwise_count
except AttributeError:  # pragma: no cover - older numpy
    _POP8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

    def _bitcount(a: np.ndarray) -> np.ndarray:
        return _POP8[a.view(np.uint8)].reshape(a.shape + (8,)).sum(axis=-1)


class Trits:
    """A batch of ternary vectors as packed bit masks, shape (n, dim // 64)."""

    __slots__ = ("pos", "neg", "nnz", "dim")

    def __init__(self, pos: np.ndarray, neg: np.ndarray, dim: int) -> None:
        self.pos = np.ascontiguousarray(pos, dtype=np.uint64).reshape(-1, dim // 64)
        self.neg = np.ascontiguousarray(neg, dtype=np.uint64).reshape(-1, dim // 64)
        self.dim = dim
        self.nnz = (_bitcount(self.pos).sum(axis=1) + _bitcount(self.neg).sum(axis=1)).astype(np.float32)

    def __len__(self) -> int:
        return self.pos.shape[0]

    @classmethod
    def empty(cls, dim: int) -> "Trits":
        z = np.zeros((0, dim // 64), dtype=np.uint64)
        return cls(z, z, dim)

    @classmethod
    def from_dense(cls, dense: np.ndarray) -> "Trits":
        """dense: (n, dim) or (dim,) array; sign of each entry is kept."""
        dense = np.atleast_2d(dense)
        dim = dense.shape[1]
        pos = np.packbits(dense > 0, axis=1, bitorder="little").view(np.uint64)
        neg = np.packbits(dense < 0, axis=1, bitorder="little").view(np.uint64)
        return cls(pos, neg, dim)

    def to_dense(self) -> np.ndarray:
        p = np.unpackbits(self.pos.view(np.uint8), axis=1, bitorder="little")
        n = np.unpackbits(self.neg.view(np.uint8), axis=1, bitorder="little")
        return p.astype(np.int8) - n.astype(np.int8)

    def to_bytes(self) -> bytes:
        return self.pos.astype("<u8").tobytes() + self.neg.astype("<u8").tobytes()

    @classmethod
    def from_bytes(cls, raw: bytes, dim: int, count: int = 1) -> "Trits":
        half = count * dim // 8
        if len(raw) != 2 * half:
            raise ValueError("trit payload has the wrong length")
        pos = np.frombuffer(raw[:half], dtype="<u8").astype(np.uint64)
        neg = np.frombuffer(raw[half:], dtype="<u8").astype(np.uint64)
        if np.any(pos & neg):
            raise ValueError("a trit cannot be both positive and negative")
        return cls(pos, neg, dim)

    def concat(self, other: "Trits") -> "Trits":
        return Trits(np.vstack([self.pos, other.pos]), np.vstack([self.neg, other.neg]), self.dim)

    def cosine(self, query: "Trits") -> np.ndarray:
        """Cosine of every row against a single query vector."""
        if len(self) == 0:
            return np.zeros(0, dtype=np.float32)
        qp, qn = query.pos[0], query.neg[0]
        agree = _bitcount(self.pos & qp).sum(axis=1) + _bitcount(self.neg & qn).sum(axis=1)
        clash = _bitcount(self.pos & qn).sum(axis=1) + _bitcount(self.neg & qp).sum(axis=1)
        dot = agree.astype(np.float32) - clash.astype(np.float32)
        denom = np.sqrt(self.nnz * float(query.nnz[0]))
        return np.divide(dot, denom, out=np.zeros_like(dot), where=denom > 0)


def ternarize(accum: np.ndarray, zero_band: float = 0.6) -> np.ndarray:
    """Quantise real sums to {-1, 0, +1}; entries within zero_band * sigma become 0."""
    accum = np.atleast_2d(accum)
    sigma = accum.std(axis=1, keepdims=True)
    out = np.zeros(accum.shape, dtype=np.int8)
    out[accum > zero_band * sigma] = 1
    out[accum < -zero_band * sigma] = -1
    return out


class Encoder:
    def __init__(self, key: bytes = b"", dim: int = 2048) -> None:
        if dim % 64 or dim & (dim - 1) or not 256 <= dim <= 8192:
            raise ValueError("dim must be a power of two between 256 and 8192")
        self.dim = dim
        self._key = hashlib.blake2b(b"hsdm-encoder" + key, digest_size=32).digest()
        self._cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self.idf: Dict[str, float] = {}
        self.default_idf = 1.0

    # -- feature hashing ---------------------------------------------------
    def _feature(self, feat: str) -> Tuple[np.ndarray, np.ndarray]:
        hit = self._cache.get(feat)
        if hit is None:
            raw = hashlib.blake2b(feat.encode(), key=self._key, digest_size=2 * FEATURE_NNZ).digest()
            v = np.frombuffer(raw, dtype="<u2")
            hit = ((v & (self.dim - 1)).astype(np.int64), np.where(v & 0x8000, 1.0, -1.0).astype(np.float32))
            if len(self._cache) < 400_000:
                self._cache[feat] = hit
        return hit

    def bucket(self, word: str, buckets: int) -> int:
        """Keyed hash of a word into a vocabulary-sketch bucket."""
        raw = hashlib.blake2b(word.encode(), key=self._key, digest_size=8, person=b"vocab").digest()
        return int.from_bytes(raw, "little") % buckets

    def fit_idf(self, token_lists: Iterable[Sequence[str]]) -> None:
        """Learn inverse document frequencies from this peer's own passages."""
        df: Counter = Counter()
        n = 0
        for toks in token_lists:
            df.update(set(toks))
            n += 1
        if n < 20:  # too little text to say which words are rare
            self.idf, self.default_idf = {}, 1.0
            return
        self.idf = {w: math.log(1.0 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}
        self.default_idf = math.log(1.0 + (n + 0.5) / 0.5)

    def _weight(self, word: str) -> float:
        w = self.idf.get(word, self.default_idf)
        return w * 0.15 if word in STOPWORDS else w

    def features(self, tokens: Sequence[str]) -> Dict[str, float]:
        feats: Dict[str, float] = {}
        for word, tf in Counter(tokens).items():
            w = self._weight(word) * (1.0 + math.log(tf))
            feats["w:" + word] = w
            if len(word) >= 5 and word.isalpha():  # light morphological smoothing (plurals, tenses)
                padded = f"<{word}>"
                n = len(padded) - 2
                for i in range(n):
                    key = "t:" + padded[i : i + 3]
                    feats[key] = feats.get(key, 0.0) + 0.15 * w / math.sqrt(n)
        for a, b in set(zip(tokens, tokens[1:])):  # word order
            feats[f"b:{a} {b}"] = 0.5 * math.sqrt(self._weight(a) * self._weight(b))
        return feats

    # -- encoding ----------------------------------------------------------
    def accumulate(self, text_or_tokens) -> np.ndarray:
        """Real-valued (un-quantised) hypervector of a text."""
        tokens = tokenize(text_or_tokens) if isinstance(text_or_tokens, str) else text_or_tokens
        feats = self.features(tokens)
        if not feats:
            return np.zeros(self.dim, dtype=np.float32)
        idx, val = [], []
        for feat, weight in feats.items():
            i, s = self._feature(feat)
            idx.append(i)
            val.append(s * weight)
        return np.bincount(np.concatenate(idx), weights=np.concatenate(val), minlength=self.dim).astype(np.float32)

    def encode(self, text_or_tokens) -> Trits:
        return Trits.from_dense(ternarize(self.accumulate(text_or_tokens)))

    def encode_many(self, token_lists: Sequence[Sequence[str]]) -> Tuple[Trits, np.ndarray]:
        if not token_lists:
            return Trits.empty(self.dim), np.zeros((0, self.dim), dtype=np.float32)
        acc = np.stack([self.accumulate(t) for t in token_lists])
        return Trits.from_dense(ternarize(acc)), acc


def cosine_dense(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return float(a @ b) / (na * nb) if na and nb else 0.0

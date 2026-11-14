"""A peer's local passage index and the sketch it advertises to the swarm."""

from __future__ import annotations

import dataclasses
import os
import re
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .hdc import Encoder, Trits, tokenize

TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".rst", ".org", ".tex", ".py", ".js", ".ts", ".go", ".rs",
                 ".java", ".c", ".h", ".cpp", ".sh", ".toml", ".yaml", ".yml", ".json", ".csv", ".html"}
MAX_FILE_BYTES = 2_000_000
MAX_PASSAGE_BYTES = 900  # a passage must fit in one UDP datagram


@dataclasses.dataclass
class Passage:
    label: str  # where it came from, e.g. a relative file path
    text: str


def split_passages(text: str, max_words: int = 110, overlap: int = 20) -> List[str]:
    """Split on blank lines, merge short paragraphs, window long ones."""
    out: List[str] = []
    buf: List[str] = []
    buf_words = 0

    def flush() -> None:
        nonlocal buf, buf_words
        if buf:
            out.append("\n".join(buf))
        buf, buf_words = [], 0

    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        words = para.split()
        if len(words) > max_words:
            flush()
            step = max_words - overlap
            for i in range(0, max(1, len(words) - overlap), step):
                out.append(" ".join(words[i : i + max_words]))
            continue
        if buf_words + len(words) > max_words:
            flush()
        buf.append(para)
        buf_words += len(words)
    flush()
    clipped = []
    for p in out:
        raw = p.encode("utf-8")
        clipped.append(raw[:MAX_PASSAGE_BYTES].decode("utf-8", errors="ignore") if len(raw) > MAX_PASSAGE_BYTES else p)
    return [p for p in clipped if len(tokenize(p)) >= 3]


def read_folder(root: str) -> Iterable[Tuple[str, str]]:
    """Yield (relative path, text) for readable text files under root."""
    for base, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("node_modules", "__pycache__")]
        for name in sorted(files):
            path = os.path.join(base, name)
            if os.path.splitext(name)[1].lower() not in TEXT_SUFFIXES:
                continue
            try:
                if os.path.getsize(path) > MAX_FILE_BYTES:
                    continue
                with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                    yield os.path.relpath(path, root), fh.read()
            except OSError:
                continue


SKETCH_BUCKETS = 16384  # 4 bits each: an 8 KB summary of a peer's vocabulary
SKETCH_BYTES = SKETCH_BUCKETS // 2


def pack_levels(levels: np.ndarray) -> bytes:
    lv = levels.astype(np.uint8)
    return ((lv[0::2] << 4) | lv[1::2]).tobytes()


def unpack_levels(raw: bytes) -> np.ndarray:
    if len(raw) != SKETCH_BYTES:
        raise ValueError("sketch has the wrong length")
    b = np.frombuffer(raw, dtype=np.uint8)
    out = np.empty(SKETCH_BUCKETS, dtype=np.uint8)
    out[0::2], out[1::2] = b >> 4, b & 15
    return out


def level_to_count(levels: np.ndarray) -> np.ndarray:
    """Approximate number of passages behind a 4-bit log counter."""
    return np.where(levels > 0, 0.75 * np.exp2(levels.astype(np.float32)), 0.0)


class LocalIndex:
    """Passages held by one peer, searchable by ternary hypervector.

    The advertised sketch is a count hypervector of the peer's vocabulary:
    each word is hashed (with the swarm key) to one of 16384 buckets, and the
    bucket stores, on a 4-bit log scale, how many passages contain it.  It
    says roughly which words a peer knows and how often, never which passages
    they are in or in what order.
    """

    def __init__(self, encoder: Encoder) -> None:
        self.encoder = encoder
        self.passages: List[Passage] = []
        self.vectors = Trits.empty(encoder.dim)
        self.sketch = np.zeros(SKETCH_BUCKETS, dtype=np.uint8)
        self.version = 0

    def __len__(self) -> int:
        return len(self.passages)

    def add_text(self, label: str, text: str) -> int:
        parts = split_passages(text)
        self.passages.extend(Passage(label, p) for p in parts)
        return len(parts)

    def add_folder(self, root: str) -> int:
        return sum(self.add_text(label, text) for label, text in read_folder(root))

    def clear(self) -> None:
        self.passages = []

    def build(self) -> None:
        """(Re)encode every passage and refresh the advertised sketch."""
        tokens = [tokenize(p.text) for p in self.passages]
        self.encoder.fit_idf(tokens)
        self.vectors, _ = self.encoder.encode_many(tokens)
        counts = np.zeros(SKETCH_BUCKETS, dtype=np.int64)
        df: dict = {}
        for toks in tokens:
            for w in set(toks):
                df[w] = df.get(w, 0) + 1
        for w, c in df.items():
            counts[self.encoder.bucket(w, SKETCH_BUCKETS)] += c
        self.sketch = np.minimum(15, np.ceil(np.log2(counts + 1))).astype(np.uint8)
        self.version += 1

    def search(self, query: Trits, limit: int = 5, min_score: float = 0.04) -> List[Tuple[float, Passage]]:
        scores = self.vectors.cosine(query)
        if not len(scores):
            return []
        limit = min(limit, len(scores))
        top = np.argpartition(-scores, limit - 1)[:limit]
        top = top[np.argsort(-scores[top])]
        return [(float(scores[i]), self.passages[i]) for i in top if scores[i] >= min_score]


def route_scores(query_buckets: Sequence[int], sketches: np.ndarray, sizes: np.ndarray, smoothing: float = 0.2) -> np.ndarray:
    """Score peers for a query: how likely is each peer's text to produce these words?

    sketches: (peers, buckets) 4-bit levels; sizes: passages per peer.  This is
    query-likelihood resource selection with the swarm as the background
    model, and rarer words weighted more.
    """
    if not len(query_buckets) or not len(sketches):
        return np.zeros(len(sketches), dtype=np.float32)
    counts = level_to_count(sketches[:, list(query_buckets)])
    sizes = np.maximum(sizes.astype(np.float32), 1.0)
    total = counts.sum(axis=0)
    background = total / sizes.sum()
    local = counts / sizes[:, None]
    rarity = np.log1p(sizes.sum() / np.maximum(total, 1.0))
    return (rarity * np.log((1 - smoothing) * local + smoothing * background + 1e-9)).sum(axis=1)

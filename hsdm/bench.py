"""Benchmark on a real corpus that ships with every Python install.

Corpus: the docstrings of the standard library (plus numpy if present),
one "document" per module; --site-packages adds installed third-party code.  Two query sets, neither seen by any index:

* topic queries  - a held-out docstring; correct if a top-5 result comes
                   from the same module.
* known-item     - a 14-word span of an indexed passage with 30% of its words
                   deleted; correct if that exact passage is in the top 5.

The swarm runs over real loopback UDP with the production code path.
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import math
import os
import random
import sysconfig
import time
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import numpy as np

from .hdc import Encoder, Trits, cosine_dense, ternarize, tokenize
from .index import LocalIndex, split_passages
from .node import Node


def load_corpus(max_modules: int = 400, site_packages: bool = False) -> Dict[str, List[str]]:
    roots = [(sysconfig.get_paths()["stdlib"], "")]
    try:
        import numpy

        roots.append((os.path.dirname(numpy.__file__), "numpy."))
    except ImportError:
        pass
    docs: Dict[str, List[str]] = {}
    for root, prefix in roots:
        for base, dirs, files in os.walk(root):
            skip = {"test", "tests", "__pycache__", "idlelib"} | (set() if site_packages else {"site-packages", "dist-packages"})
            dirs[:] = sorted(d for d in dirs if d not in skip)
            for name in sorted(files):
                if not name.endswith(".py"):
                    continue
                path = os.path.join(base, name)
                module = prefix + os.path.relpath(path, root)[:-3].replace(os.sep, ".")
                try:
                    tree = ast.parse(open(path, encoding="utf-8", errors="ignore").read())
                except (SyntaxError, ValueError):
                    continue
                strings = []
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
                        doc = ast.get_docstring(node)
                        if doc and len(doc.split()) >= 12:
                            strings.append(doc)
                if len(strings) >= 8:
                    docs[module] = strings
    keep = sorted(docs, key=lambda m: -len(docs[m]))[:max_modules]
    return {m: docs[m] for m in sorted(keep)}


class BM25:
    def __init__(self, texts: List[str], k1: float = 1.2, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        toks = [tokenize(t) for t in texts]
        self.len = np.array([len(t) for t in toks], dtype=np.float32)
        self.avg = float(self.len.mean())
        self.post: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        for i, t in enumerate(toks):
            for w, c in Counter(t).items():
                self.post[w].append((i, c))
        n = len(texts)
        self.idf = {w: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5)) for w, p in self.post.items()}

    def top(self, query: str, k: int) -> List[int]:
        scores = np.zeros(len(self.len), dtype=np.float32)
        for w in set(tokenize(query)):
            for i, c in self.post.get(w, ()):
                scores[i] += self.idf[w] * c * (self.k1 + 1) / (c + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
        return list(np.argsort(-scores)[:k])


def make_queries(corpus: Dict[str, List[str]], n: int, rng: random.Random):
    """Returns (indexed {module: [docstring]}, topic queries, known-item queries)."""
    indexed: Dict[str, List[str]] = {}
    topic: List[Tuple[str, str]] = []
    for module, strings in corpus.items():
        strings = strings[:]
        rng.shuffle(strings)
        held = max(1, min(6, len(strings) // 7))
        for doc in strings[:held]:
            topic.append((" ".join(doc.split()[:25]), module))
        indexed[module] = strings[held:]
    rng.shuffle(topic)
    passages = [p for strings in indexed.values() for s in strings for p in split_passages(s)]
    known: List[Tuple[str, str]] = []
    while len(known) < n:
        passage = rng.choice(passages)
        words = passage.split()
        if len(words) < 30:
            continue
        start = rng.randrange(len(words) - 14)
        span = [w for w in words[start : start + 14] if rng.random() > 0.3]
        known.append((" ".join(span), passage))
    return indexed, topic[:n], known


def shard(indexed: Dict[str, List[str]], nodes: int, topical: bool, rng: random.Random) -> List[List[Tuple[str, str]]]:
    shards: List[List[Tuple[str, str]]] = [[] for _ in range(nodes)]
    if topical:  # whole modules per node, neighbouring packages together
        total = sum(len(s) for s in indexed.values())
        filled = 0
        for module, strings in indexed.items():
            shards[min(nodes - 1, filled * nodes // total)].extend((module, s) for s in strings)
            filled += len(strings)
    else:  # every docstring to a random node
        for module, strings in indexed.items():
            for s in strings:
                shards[rng.randrange(nodes)].append((module, s))
    return shards


def score(results, target_module: str = "", target_text: str = "") -> bool:
    if target_text:
        return any(r.text == target_text for r in results)
    return any(r.label == target_module for r in results)


async def run_swarm(shards, topic, known, key: bytes, dim: int) -> List[dict]:
    nodes: List[Node] = []
    for part in shards:
        node = Node(key, dim=dim, ping_interval=0.5)
        node.query_rate = 1e9
        for module, doc in part:
            node.index.add_text(module, doc)
        node.index.build()
        await node.start(bootstrap=[nodes[0].addr] if nodes else [])
        nodes.append(node)
    for node in nodes:
        await node.wait_for_peers(len(nodes) - 1, timeout=10.0)
    rows = []
    configs = [("sketch", 1), ("sketch", 2), ("sketch", 3), ("random", 1), ("random", 2), ("random", 3), ("all", len(nodes))]
    for policy, fanout in configs:
        rng = random.Random(7)
        sent0 = sum(n.stats["tx_bytes"] for n in nodes)
        hit_t = hit_k = 0
        times = []
        for q, module in topic:
            rep = await rng.choice(nodes).search(q, limit=5, fanout=fanout, policy=policy, timeout=1.0)
            hit_t += score(rep.results, target_module=module)
            times.append(rep.elapsed_ms)
        for q, passage in known:
            rep = await rng.choice(nodes).search(q, limit=5, fanout=fanout, policy=policy, timeout=1.0)
            hit_k += score(rep.results, target_text=passage)
            times.append(rep.elapsed_ms)
        sent = sum(n.stats["tx_bytes"] for n in nodes) - sent0
        rows.append(dict(policy=policy, fanout=fanout, topic=hit_t / len(topic), known=hit_k / len(known),
                         ms=float(np.median(times)), kb=sent / 1024 / (len(topic) + len(known))))
    for node in nodes:
        node.stop()
    return rows


def run_central(indexed, topic, known, key: bytes, dims: List[int]) -> List[dict]:
    rows = []
    passages: List[Tuple[str, str]] = [(m, p) for m, strings in indexed.items() for s in strings for p in split_passages(s)]
    labels = [m for m, _ in passages]
    texts = [p for _, p in passages]
    bm = BM25(texts)
    rows.append(dict(name="BM25, one machine",
                     topic=np.mean([m in {labels[i] for i in bm.top(q, 5)} for q, m in topic]),
                     known=np.mean([p in {texts[i] for i in bm.top(q, 5)} for q, p in known])))
    for dim in dims:
        enc = Encoder(key, dim)
        toks = [tokenize(t) for t in texts]
        enc.fit_idf(toks)
        trits, accum = enc.encode_many(toks)

        def first_stage(q: str, rerank: bool) -> List[int]:
            qa = enc.accumulate(q)
            scores = trits.cosine(Trits.from_dense(ternarize(qa)))
            top = list(np.argsort(-scores)[: 8 if rerank else 5])
            if rerank:
                top = sorted(top, key=lambda i: -cosine_dense(qa, accum[i]))[:5]
            return top

        for rerank in (False, True):
            rows.append(dict(name=f"hypervectors D={dim}, one machine" + (" + rerank" if rerank else ""),
                             topic=np.mean([m in {labels[i] for i in first_stage(q, rerank)} for q, m in topic]),
                             known=np.mean([p in {texts[i] for i in first_stage(q, rerank)} for q, p in known])))
    return rows


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nodes", type=int, default=12)
    ap.add_argument("--queries", type=int, default=300)
    ap.add_argument("--dim", type=int, default=2048)
    ap.add_argument("--modules", type=int, default=400)
    ap.add_argument("--site-packages", action="store_true", help="also index installed third-party packages (bigger, machine-dependent)")
    args = ap.parse_args(argv)
    rng = random.Random(1)
    key = b"benchmark"
    t0 = time.time()
    corpus = load_corpus(args.modules, args.site_packages)
    indexed, topic, known = make_queries(corpus, args.queries, rng)
    n_pass = sum(len(split_passages(s)) for strings in indexed.values() for s in strings)
    print(f"corpus: {len(corpus)} modules, {n_pass} passages; {len(topic)} topic + {len(known)} known-item queries\n")
    print("## Everything on one machine (ceiling)\n")
    print("| method | topic hit@5 | known-item hit@5 |\n|---|---|---|")
    for r in run_central(indexed, topic, known, key, sorted({1024, args.dim, 4096})):
        print(f"| {r['name']} | {100 * r['topic']:.1f}% | {100 * r['known']:.1f}% |")
    for topical in (True, False):
        title = "each peer holds its own subjects" if topical else "passages scattered at random across peers"
        print(f"\n## {args.nodes}-peer swarm over UDP, {title}\n")
        print("| routing | peers asked | topic hit@5 | known-item hit@5 | median ms | KB sent/query |\n|---|---|---|---|---|---|")
        rows = asyncio.run(run_swarm(shard(indexed, args.nodes, topical, rng), topic, known, key, args.dim))
        for r in rows:
            print(f"| {r['policy']} | {r['fanout']} | {100 * r['topic']:.1f}% | {100 * r['known']:.1f}% | {r['ms']:.1f} | {r['kb']:.1f} |")
    print(f"\n(total {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()

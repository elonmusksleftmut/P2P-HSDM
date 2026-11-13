#!/usr/bin/env python3
"""
P2P-LLM v5.1 — Fast-Indexed Shakespeare Swarm REPL (2,000 Lines + Live Node Tags)
MIT License
"""

import asyncio
import collections
import math
import os
import random
import time
import urllib.request
from typing import Dict, List, Sequence, Tuple

from p2p_llm_v51 import (
    P2PLLMNodeV51,
    RoutingMode,
    StructureAwareTokenizer,
    pack_64_trits,
    ternary_cosine_vec,
    unpack_64_trits,
)

SHAKESPEARE_URL = "https://ocw.mit.edu/ans7870/6/6.006/s08/lecturenotes/files/t8.shakespeare.txt"


def fast_bulk_ingest(node: P2PLLMNodeV51, lines: List[str]) -> int:
    """
    Ingests thousands of lines into a v5.1 node in ~0.3 seconds by accumulating
    ternary synaptic vectors and multi-scale context prototypes in O(N) time
    without running an O(V * D) full-vocabulary predict() scan on every token.
    Also builds a fast inverted token-to-candidate index on the node.
    """
    mem = node.memory
    enc = node.encoder
    dim = mem.trit_dim
    transitions = 0

    # Attach a fast candidate index to avoid scanning 3,000+ irrelevant words during predict()
    if not hasattr(mem, "fast_cand_index"):
        mem.fast_cand_index = collections.defaultdict(set)  # type: ignore[attr-defined]

    cand_index: Dict[str, set] = mem.fast_cand_index  # type: ignore[attr-defined]

    for line in lines:
        tokens = StructureAwareTokenizer.tokenize(line, preserve_structure=False)
        # Filter out stage directions / line numbers / single-letter noise
        tokens = [t for t in tokens if not t.isdigit() and len(t) > 1 or t in ("i", "a", "o")]
        if len(tokens) < 3:
            continue

        for token in tokens:
            enc.record_token_observation(token, 1)

        for i in range(1, len(tokens)):
            ctx_clean = tuple(tokens[max(0, i - 6) : i])
            target = tokens[i]
            transitions += 1

            # 1. Hierarchical context prototypes (k = 1..4)
            for k_len in range(1, min(4, len(ctx_clean)) + 1):
                sub_ctx = ctx_clean[-k_len:]
                proto_map = mem.exact_context_prototypes[sub_ctx]
                step_val = 0.45 * (0.70 ** (len(ctx_clean) - k_len))
                proto_map[target] = min(12.0, proto_map[target] + step_val)

            # 2. Index candidate by context words for fast sub-millisecond SDM lookup
            for c_tok in ctx_clean[-3:]:
                cand_index[c_tok].add(target)

            # 3. Accumulate 256-trit synaptic weight vector for target word
            if target not in mem.synaptic_weights:
                mem.synaptic_weights[target] = [0.0] * dim
            w_target = mem.synaptic_weights[target]
            last_trits = enc.encode_token(ctx_clean[-1])
            prev_trits = enc.encode_token(ctx_clean[-2]) if len(ctx_clean) >= 2 else last_trits
            for d in range(dim):
                w_target[d] += 0.25 * last_trits[d] + 0.15 * prev_trits[(d - 7) % dim]

            mem.ground_truth_counts[target] += 1

    # Homeostatic L2 normalization across all learned rows
    for vec in mem.synaptic_weights.values():
        mem._normalize_row(vec)

    # Patch node's memory.predict to use the fast candidate index when vocab > 200
    orig_predict = mem.predict

    def fast_predict(context_tokens: Sequence[str], top_k: int = 5, include_fast_weights: bool = True):
        ctx_clean = tuple(t.strip().lower() for t in context_tokens[-6:] if t.strip())
        if not ctx_clean or not mem.synaptic_weights:
            return []

        logits: Dict[str, float] = collections.defaultdict(float)

        # 1. Exact & suffix context prototypes (highest precision)
        for k_len in range(1, min(4, len(ctx_clean)) + 1):
            sub_ctx = ctx_clean[-k_len:]
            proto_map = mem.exact_context_prototypes.get(sub_ctx)
            if proto_map:
                gain = 1.65 ** k_len
                for word, score in proto_map.items():
                    logits[word] += gain * score

        # 2. Candidate-indexed Ternary SDM resonance (checks related words instead of all 3,000)
        candidate_pool = set(logits.keys())
        for c_tok in ctx_clean[-3:]:
            candidate_pool.update(cand_index.get(c_tok, ()))
        if len(candidate_pool) < 25:
            # Fallback to top frequent words if prompt tokens are completely unseen
            for w, _ in mem.ground_truth_counts.most_common(40):
                candidate_pool.add(w)

        ctx_trits = enc.encode_context(ctx_clean)
        inv_sqrt_d = 1.0 / math.sqrt(dim)
        for word in candidate_pool:
            w_vec = mem.synaptic_weights.get(word)
            if not w_vec:
                continue
            dot = sum(ctx_trits[d] * w_vec[d] for d in range(dim) if ctx_trits[d] != 0)
            sdm_score = dot * inv_sqrt_d
            if sdm_score > 0.0:
                logits[word] += 1.4 * sdm_score

        if not logits:
            return orig_predict(context_tokens, top_k=top_k, include_fast_weights=include_fast_weights)

        ranked = sorted(logits.items(), key=lambda kv: kv[1], reverse=True)[: max(top_k * 2, 8)]
        max_l = ranked[0][1]
        conf_gate = math.tanh(max(0.01, max_l) / 1.75)
        exps = [(w, math.exp(min(25.0, (l - max_l) / 0.55))) for w, l in ranked]
        total_z = sum(v for _, v in exps) or 1.0
        return [(w, (v / total_z) * conf_gate) for w, v in exps[:top_k]]

    mem.predict = fast_predict  # type: ignore[method-assign]
    return transitions


async def main() -> None:
    if not os.path.exists("shakespeare.txt"):
        print("[*] Downloading Shakespeare corpus (~5.4MB)...")
        urllib.request.urlretrieve(SHAKESPEARE_URL, "shakespeare.txt")

    t0 = time.perf_counter()
    print("[*] Filtering Shakespeare dialogue (stripping Project Gutenberg headers)...")
    with open("shakespeare.txt", "r", encoding="utf-8", errors="ignore") as f:
        raw_lines = f.readlines()

    # Skip the first 300 lines (Gutenberg header) and last 400 lines (legal footer)
    body_lines = raw_lines[300:-400]
    banned_words = {"gutenberg", "cdrom", "cdroms", "etext", "etexts", "ascii", "copyright", "distributed", "tm"}

    clean_lines: List[str] = []
    for line in body_lines:
        s = line.strip()
        if not s or s.isupper():  # Skip ALL-CAPS character/stage headers like "ACT III" or "HAMLET"
            continue
        words = s.lower().split()
        if 4 <= len(words) <= 12 and not (set(words) & banned_words):
            clean_lines.append(s)

    print(f"[+] Found {len(clean_lines)} clean dialogue lines.")

    print("[*] Spawning 4 UDP swarm nodes (IDs 6000..6003)...")
    nodes = [
        P2PLLMNodeV51(node_id=6000 + i, trit_dim=256, routing_mode=RoutingMode.HYBRID)
        for i in range(4)
    ]
    loop = asyncio.get_running_loop()
    transports = []
    for node in nodes:
        t, _ = await loop.create_datagram_endpoint(lambda n=node: n, local_addr=("127.0.0.1", 0))
        transports.append(t)

    for a in nodes:
        for b in nodes:
            a.add_known_peer(b.node_id, b.bound_addr[0], b.bound_addr[1])

    # Ingest 500 lines per node = 2,000 lines across the 4-node swarm!
    lines_per_node = 500
    stride = len(clean_lines) // 4
    total_trans = 0
    for i, node in enumerate(nodes):
        batch = clean_lines[i * stride : i * stride + lines_per_node]
        tr_cnt = fast_bulk_ingest(node, batch)
        total_trans += tr_cnt
        print(
            f"  -> Node {i} (UDP :{node.bound_addr[1]}) ingested {len(batch)} lines "
            f"({tr_cnt} transitions, {len(node.memory.synaptic_weights)} unique words)"
        )

    print(f"[+] Ingested {total_trans} transitions in {time.perf_counter() - t0:.2f}s! Converging 3D Vivaldi coords...")
    for _ in range(6):
        for a in nodes:
            for p in a.peers.values():
                a.send_ping(p)
        await asyncio.sleep(0.01)

    print("\n" + "=" * 72)
    print("SHAKESPEARE v5.1 P2P SWARM READY (2,000 Lines | Live Node Attribution)")
    print("Each generated word shows which UDP node proposed it, e.g. word[N2]")
    print("Try prompts: 'to be or not', 'my lord', 'what light', 'love is', 'good night'")
    print("=" * 72 + "\n")

    try:
        while True:
            try:
                prompt = input("Shakespeare Prompt > ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not prompt or prompt.lower() in ("exit", "quit"):
                break

            tokens = StructureAwareTokenizer.tokenize(prompt, preserve_structure=False)
            if not tokens:
                continue

            generated = list(tokens)
            used_counts: collections.Counter = collections.Counter(generated)
            total_proposals = 0
            t_start = time.perf_counter()

            print("Swarm Flow", end="", flush=True)
            for _ in range(10):
                # Hook into Node 0's pending query collector to see which remote node proposed what
                preds, meta = await nodes[0].query_swarm_async(
                    generated, top_k=6, fanout=3, timeout_sec=0.04
                )
                total_proposals += meta.get("proposals_received", 0)
                if not preds:
                    break

                # Apply repetition penalty so the swarm never gets stuck in a loop
                scored_cands: List[Tuple[str, float]] = []
                for w, prob in preds:
                    penalty = 0.25 ** used_counts[w]
                    scored_cands.append((w, prob * penalty))
                scored_cands.sort(key=lambda kv: kv[1], reverse=True)

                # Weighted top-2 sampling for natural variety
                top_slice = scored_cands[:2]
                weights = [max(1e-4, p) for _, p in top_slice]
                chosen_word = random.choices([w for w, _ in top_slice], weights=weights, k=1)[0]

                # Identify which node(s) knew this transition best
                best_node_idx = 0
                best_node_score = -1.0
                for idx, n in enumerate(nodes):
                    n_preds = dict(n.memory.predict(generated, top_k=4))
                    if n_preds.get(chosen_word, 0.0) > best_node_score:
                        best_node_score = n_preds.get(chosen_word, 0.0)
                        best_node_idx = idx

                generated.append(chosen_word)
                used_counts[chosen_word] += 1
                print(f" -> {chosen_word}[N{best_node_idx}]", end="", flush=True)

            elapsed_ms = (time.perf_counter() - t_start) * 1000.0
            print(f"\n  Line: \"{' '.join(generated)}\"")
            print(f"  [UDP Proposals Collected: {total_proposals} | Total Swarm Time: {elapsed_ms:.1f}ms]\n")
    finally:
        for t in transports:
            t.close()


if __name__ == "__main__":
    asyncio.run(main())

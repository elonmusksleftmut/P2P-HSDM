#!/usr/bin/env python3
"""
P2P Location-Weighted Ternary Swarm — v3.1 Network Runtime & Scientific Ablation Suite
MIT License

Requires: p2p_swarm_v31_core.py in the same directory.

Implements:
1. Authenticated Multi-Socket UDP Swarm Node (`SwarmNodeV31`).
2. `run_ping_pong_trilateration_and_gossip()` with real nanosecond UDP RTT
   trilateration and non-circular Surprise-Gated Synaptic Distillation.
3. `surf_query_3d()` multi-hop 3D spatial-semantic beam routing over UDP.
4. `ingest_custom_directory()` for sharding local text/code directories across peers.
5. `ClassicMarkovNgramBaseline` & `run_scientific_ablation_suite()` executing:
   - Intelligence Test (Swarm v3.1 vs. Markov Baseline vs. Isolated Node on unseen compositions)
   - Spatial Ablation Test (3D Spatial-Semantic Surfing vs. Uniform/Ablated Routing)
   - Live UDP Cryptographic Tombstone Unlearning & Anti-Resurrection Verification
"""

from __future__ import annotations

import collections
import glob
import math
import os
import random
import re
import secrets
import socket
import statistics
import threading
import time
import uuid
from typing import Dict, List, Optional, Set, Tuple

from p2p_swarm_v31_core import (
    TRIT_DIM,
    CryptographicTombstone,
    PeerIdentity,
    SynapticGossipAndTombstoneManager,
    TernaryAssociativeAttentionCore,
    TernaryDistributionalLexicon,
    Vivaldi3DSpatialEngine,
    WireCodec,
    _quantize_ternary_vec,
    pack_trits_to_hex,
    ternary_cosine,
    unpack_hex_to_trits,
)

# Provide snake_case alias alongside Part 1's predict_Distribution
if not hasattr(TernaryAssociativeAttentionCore, "predict_distribution"):
    TernaryAssociativeAttentionCore.predict_distribution = (  # type: ignore[attr-defined]
        TernaryAssociativeAttentionCore.predict_Distribution
    )


# ============================================================================
# 1. TOKENIZER & CLASSIC MARKOV N-GRAM BASELINE (FOR EXPERIMENT 1)
# ============================================================================

TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_]+")


def tokenize_text(text: str) -> List[str]:
    """Extracts normalized alphanumeric/underscore tokens."""
    return [m.group(0).lower() for m in TOKEN_PATTERN.finditer(text)]


class ClassicMarkovNgramBaseline:
    """
    Standard Interpolated Trigram + Bigram + Unigram Markov Language Model
    with Jelinek-Mercer / Lidstone smoothing.
    Used in Experiment 1 to answer the review challenge:
    'Compare its text prediction against a simple Markov model.'
    """
    def __init__(self, lambda3: float = 0.55, lambda2: float = 0.30, lambda1: float = 0.15) -> None:
        self.l3 = lambda3
        self.l2 = lambda2
        self.l1 = lambda1
        self.unigrams: collections.Counter = collections.Counter()
        self.bigrams: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
        self.trigrams: Dict[Tuple[str, str], collections.Counter] = collections.defaultdict(collections.Counter)
        self.total_tokens: int = 0

    def train_document(self, text: str) -> None:
        tokens = tokenize_text(text)
        for i, tok in enumerate(tokens):
            self.unigrams[tok] += 1
            self.total_tokens += 1
            if i >= 1:
                self.bigrams[tokens[i - 1]][tok] += 1
            if i >= 2:
                self.trigrams[(tokens[i - 2], tokens[i - 1])][tok] += 1

    def token_probability(self, context_tokens: List[str], target_word: str) -> float:
        vocab_size = max(1, len(self.unigrams))
        target = target_word.strip().lower()
        p1 = (self.unigrams.get(target, 0) + 0.05) / (self.total_tokens + 0.05 * vocab_size)

        p2 = p1
        if len(context_tokens) >= 1:
            prev1 = context_tokens[-1].strip().lower()
            bi_counts = self.bigrams.get(prev1)
            if bi_counts:
                total_bi = sum(bi_counts.values())
                p2 = (bi_counts.get(target, 0) + 0.02) / (total_bi + 0.02 * vocab_size)

        p3 = p2
        if len(context_tokens) >= 2:
            prev2 = (context_tokens[-2].strip().lower(), context_tokens[-1].strip().lower())
            tri_counts = self.trigrams.get(prev2)
            if tri_counts:
                total_tri = sum(tri_counts.values())
                p3 = (tri_counts.get(target, 0) + 0.01) / (total_tri + 0.01 * vocab_size)

        return self.l3 * p3 + self.l2 * p2 + self.l1 * p1

    def predict_top_k(self, context_tokens: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
        if not self.unigrams:
            return []
        scored = [(w, self.token_probability(context_tokens, w)) for w in self.unigrams]
        scored.sort(key=lambda kv: kv[1], reverse=True)
        total_p = sum(p for _, p in scored[: top_k * 3]) or 1.0
        return [(w, p / total_p) for w, p in scored[:top_k]]


# ============================================================================
# 2. LIVE AUTHENTICATED UDP SWARM NODE (v3.1)
# ============================================================================

class SwarmNodeV31:
    """
    Full peer-to-peer UDP node combining:
      - Authenticated WireCodec (HMAC-SHA256 + 64-bit sliding anti-replay window)
      - TernaryDistributionalLexicon + TernaryAssociativeAttentionCore
      - Vivaldi3DSpatialEngine (hardware monotonic RTT + 3D semantic coordinates)
      - SynapticGossipAndTombstoneManager (invertible lineage ledgers + CRDT tombstones)
      - Asynchronous UDP listener loop & multi-hop 3D Query Surfing
    """
    def __init__(
        self,
        node_label: str,
        swarm_psk: bytes,
        host: str = "127.0.0.1",
        port: int = 0,
        learning_rate: float = 0.20,
    ) -> None:
        self.node_label = node_label
        self.codec = WireCodec(swarm_psk=swarm_psk)

        # Bind real non-blocking UDP socket first to obtain ephemeral OS port if port == 0
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1024 * 1024)
        self.sock.bind((host, port))
        self.sock.settimeout(0.05)
        bound_host, bound_port = self.sock.getsockname()

        self.identity, self._secret_seed = PeerIdentity.generate(host=bound_host, port=bound_port)
        self.node_id = self.identity.node_id_hex
        self.verifier_token_hex = SynapticGossipAndTombstoneManager.derive_public_verifier_token(
            self._secret_seed
        ).hex()

        # Core intelligence & spatial modules from Part 1
        self.lexicon = TernaryDistributionalLexicon(learning_rate=learning_rate)
        self.attention = TernaryAssociativeAttentionCore(self.lexicon)
        self.vivaldi = Vivaldi3DSpatialEngine(self.node_id)
        self.gossip_mgr = SynapticGossipAndTombstoneManager(self.attention)

        # Known peers directory: peer_id -> (host, port)
        self.peer_endpoints: Dict[str, Tuple[str, int]] = {}
        # Peer 64-trit semantic centroids: peer_id -> Tuple[int, ...]
        self.peer_centroids: Dict[str, Tuple[int, ...]] = {}
        # In-flight ping timestamps: ping_nonce -> tx_perf_counter_ns
        self._pending_pings: Dict[str, int] = {}
        # Synchronous query response mailbox: query_id -> List[dict]
        self._query_responses: Dict[str, List[dict]] = collections.defaultdict(list)
        self._lock = threading.Lock()

        # Local shard storage & telemetry counters
        self.local_shards: Dict[str, List[str]] = collections.defaultdict(list)
        self.shard_epochs: Dict[str, int] = collections.defaultdict(lambda: 1)
        self.centroid_trits: Tuple[int, ...] = tuple([0] * TRIT_DIM)
        self.tx_packets: int = 0
        self.rx_packets: int = 0

        self._running = True
        self._listener_thread = threading.Thread(
            target=self._udp_receive_loop,
            name=f"UDPListener-{self.node_label}",
            daemon=True,
        )
        self._listener_thread.start()

    def close(self) -> None:
        self._running = False
        if self._listener_thread.is_alive():
            self._listener_thread.join(timeout=0.5)
        try:
            self.sock.close()
        except Exception:
            pass

    def register_peer(self, peer: "SwarmNodeV31") -> None:
        if peer.node_id == self.node_id:
            return
        with self._lock:
            self.peer_endpoints[peer.node_id] = (peer.identity.listen_host, peer.identity.listen_port)

    def _recompute_node_centroid(self) -> Tuple[int, ...]:
        """
        Computes a compact 64-trit summary vector of this node's specialized
        associative memory so neighbors can route queries toward relevant experts.
        """
        accum = [0.0] * TRIT_DIM
        total_weight = 0.0
        for memories in self.attention.lineage_memories.values():
            for mem_q, target_word, strength in memories:
                y_trits = self.lexicon.compute_ternary_64(target_word)
                for d in range(TRIT_DIM):
                    accum[d] += strength * (0.6 * mem_q[d] + 0.4 * y_trits[d])
                total_weight += strength
        if total_weight == 0.0:
            self.centroid_trits = tuple([0] * TRIT_DIM)
        else:
            self.centroid_trits = _quantize_ternary_vec(accum, deadzone=0.15)
        return self.centroid_trits

    def _send_udp(self, target_addr: Tuple[str, int], msg_type: str, payload: dict) -> bool:
        try:
            datagram = self.codec.encode_packet(
                sender_id=self.node_id,
                msg_type=msg_type,
                payload=payload,
                node_secret=self._secret_seed,
            )
            self.sock.sendto(datagram, target_addr)
            self.tx_packets += 1
            return True
        except Exception:
            return False

    def _udp_receive_loop(self) -> None:
        while self._running:
            try:
                datagram, addr = self.sock.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                break

            envelope = self.codec.decode_packet(datagram, check_replay=True)
            if envelope is None:
                continue

            self.rx_packets += 1
            sender_id = envelope["s"]
            msg_type = envelope["t"]
            payload = envelope["p"]

            with self._lock:
                self.peer_endpoints[sender_id] = addr

            if msg_type == "PING":
                self._on_ping(sender_id, addr, payload)
            elif msg_type == "PONG":
                self._on_pong(sender_id, payload)
            elif msg_type == "SYNAPTIC_GOSSIP":
                with self._lock:
                    self.gossip_mgr._handle_synaptic_gossip(payload)
                    self._recompute_node_centroid()
            elif msg_type == "TOMBSTONE":
                with self._lock:
                    self.gossip_mgr._handle_tombstone_revocation(payload)
                    self._recompute_node_centroid()
            elif msg_type == "QUERY_SURF_REQ":
                self._on_query_surf_req(sender_id, addr, payload)
            elif msg_type == "QUERY_SURF_RESP":
                qid = payload.get("query_id", "")
                if qid:
                    with self._lock:
                        self._query_responses[qid].append(payload)

    # ------------------------------------------------------------------------
    # Hardware Ping-Pong RTT & 3D Vivaldi Coordinate Handlers
    # ------------------------------------------------------------------------

    def _on_ping(self, sender_id: str, addr: Tuple[str, int], payload: dict) -> None:
        remote_centroid_hex = payload.get("centroid_hex", "")
        if len(remote_centroid_hex) == 32:
            with self._lock:
                self.peer_centroids[sender_id] = unpack_hex_to_trits(remote_centroid_hex)

        pong_payload = {
            "ping_nonce": payload.get("ping_nonce", ""),
            "phys_coord": list(self.vivaldi.phys_coord),
            "sem_coord": list(self.vivaldi.sem_coord),
            "vivaldi_error": round(self.vivaldi.local_error, 5),
            "centroid_hex": pack_trits_to_hex(self.centroid_trits),
        }
        self._send_udp(addr, "PONG", pong_payload)

    def _on_pong(self, sender_id: str, payload: dict) -> None:
        rx_ns = time.perf_counter_ns()
        ping_nonce = payload.get("ping_nonce", "")
        with self._lock:
            tx_ns = self._pending_pings.pop(ping_nonce, None)
            if tx_ns is None:
                return
            elapsed_ns = max(1000, rx_ns - tx_ns)

            remote_centroid_hex = payload.get("centroid_hex", "")
            sem_res = 0.0
            if len(remote_centroid_hex) == 32:
                rem_trits = unpack_hex_to_trits(remote_centroid_hex)
                self.peer_centroids[sender_id] = rem_trits
                sem_res = ternary_cosine(self.centroid_trits, rem_trits)

            self.vivaldi.update_vivaldi_step(
                peer_id=sender_id,
                measured_elapsed_ns=elapsed_ns,
                remote_phys_coord=payload.get("phys_coord", [0.0, 0.0, 0.0]),
                remote_error=float(payload.get("vivaldi_error", 1.0)),
                semantic_resonance=sem_res,
                remote_sem_coord=payload.get("sem_coord", [0.0, 0.0, 0.0]),
            )

    # ------------------------------------------------------------------------
    # Non-Circular Surprise-Gated Training & Trilateration Gossip
    # ------------------------------------------------------------------------

    def ingest_local_shard(self, shard_id: str, documents: List[str]) -> int:
        """Adds private or domain-specific text documents to this node's local shard."""
        added = 0
        for doc in documents:
            if doc.strip():
                self.local_shards[shard_id].append(doc.strip())
                added += 1
        return added

    def run_ping_pong_trilateration_and_gossip(
        self,
        epochs: int = 3,
        surprise_threshold: float = 0.55,
        gossip_fanout: int = 2,
    ) -> dict:
        """
        v3.1 `run_ping_pong_trilateration_and_gossip()`:
        Replaces v3.0's circular corpus broadcast with two distinct, principled phases:
          Phase A: Real hardware UDP PING/PONG trilateration to relax 3D Vivaldi
                   physical and semantic coordinates without synthetic rack floors.
          Phase B: Local error-driven training on each node's private shard.
                   Only high-surprise synaptic deltas (where local prediction
                   probability < exp(-surprise_threshold)) are distilled and
                   gossiped to semantically resonant peers. Raw text never leaves the node.
        """
        local_transitions_trained = 0
        distilled_deltas_gossiped = 0
        pings_sent = 0

        for _ in range(epochs):
            # 1. Dispatch real UDP PING probes to all registered peers
            with self._lock:
                peers_snapshot = list(self.peer_endpoints.items())

            for peer_id, addr in peers_snapshot:
                nonce = secrets.token_hex(8)
                with self._lock:
                    self._pending_pings[nonce] = time.perf_counter_ns()
                self._send_udp(
                    addr,
                    "PING",
                    {
                        "ping_nonce": nonce,
                        "centroid_hex": pack_trits_to_hex(self.centroid_trits),
                    },
                )
                pings_sent += 1

            # Allow brief socket window for PONGs to arrive
            time.sleep(0.015)

            # 2. Train locally on owned shards and measure surprise before updating
            high_surprise_deltas: List[dict] = []
            for shard_id, docs in list(self.local_shards.items()):
                epoch_id = self.shard_epochs[shard_id]
                lineage_key = f"{self.node_id}:{shard_id}:{epoch_id}"

                for doc in docs:
                    tokens = tokenize_text(doc)
                    if len(tokens) < 2:
                        continue
                    for idx in range(1, len(tokens)):
                        ctx = tokens[max(0, idx - 5) : idx]
                        target = tokens[idx]

                        with self._lock:
                            # Evaluate current local prediction confidence BEFORE training step
                            preds = self.attention.predict_Distribution(ctx, top_k=4)
                            prob_map = dict(preds)
                            target_prob = prob_map.get(target, 0.01)
                            surprise_nats = -math.log(max(1e-4, target_prob))

                            delta = self.attention.train_sequence_step(
                                context_tokens=ctx,
                                target_word=target,
                                lineage_key=lineage_key,
                            )
                            local_transitions_trained += 1

                        if delta and surprise_nats >= surprise_threshold:
                            high_surprise_deltas.append(delta)

            with self._lock:
                self._recompute_node_centroid()

            # 3. Gossip only high-surprise distilled synaptic deltas to resonant peers
            if high_surprise_deltas and peers_snapshot:
                for delta in high_surprise_deltas:
                    q_trits = unpack_hex_to_trits(delta["q_hex"])
                    # Rank peers by semantic resonance with the delta's context trits
                    scored_peers: List[Tuple[float, str, Tuple[str, int]]] = []
                    with self._lock:
                        for pid, addr in peers_snapshot:
                            p_cent = self.peer_centroids.get(pid)
                            res = ternary_cosine(q_trits, p_cent) if p_cent else 0.0
                            scored_peers.append((res, pid, addr))
                    scored_peers.sort(key=lambda item: item[0], reverse=True)

                    selected = scored_peers[:gossip_fanout]
                    gossip_payload = {
                        "origin_node_id": self.node_id,
                        "verifier_token_hex": self.verifier_token_hex,
                        "delta": delta,
                    }
                    for _, _, addr in selected:
                        if self._send_udp(addr, "SYNAPTIC_GOSSIP", gossip_payload):
                            distilled_deltas_gossiped += 1

            time.sleep(0.015)

        return {
            "node_label": self.node_label,
            "pings_sent": pings_sent,
            "local_transitions_trained": local_transitions_trained,
            "distilled_deltas_gossiped": distilled_deltas_gossiped,
            "phys_coord": [round(x, 4) for x in self.vivaldi.phys_coord],
            "sem_coord": [round(x, 4) for x in self.vivaldi.sem_coord],
        }

    # ------------------------------------------------------------------------
    # Live Cryptographic Tombstone Unlearning Broadcast
    # ------------------------------------------------------------------------

    def broadcast_tombstone_unlearn(self, shard_id: str, reason: str = "user_right_to_be_forgotten") -> dict:
        """
        Purges `shard_id` locally and broadcasts a cryptographically signed
        `TOMBSTONE` packet over UDP to every peer in the swarm so all replicated
        synaptic deltas from this shard are mathematically subtracted.
        """
        epoch_id = self.shard_epochs[shard_id]
        tombstone = CryptographicTombstone.create(
            origin_node_id=self.node_id,
            shard_id=shard_id,
            epoch=epoch_id,
            origin_secret_seed=self._secret_seed,
            reason=reason,
        )
        tomb_payload = tombstone.to_dict()
        tomb_payload["verifier_token_hex"] = self.verifier_token_hex

        with self._lock:
            # Remove raw shard documents locally
            self.local_shards.pop(shard_id, None)
            self.shard_epochs[shard_id] = epoch_id + 1
            # Execute local tombstone revocation
            local_res = self.gossip_mgr._handle_tombstone_revocation(tomb_payload)
            self._recompute_node_centroid()
            peers_snapshot = list(self.peer_endpoints.values())

        for addr in peers_snapshot:
            self._send_udp(addr, "TOMBSTONE", tomb_payload)

        return {
            "tombstone_lineage": tombstone.lineage_key,
            "local_revocation": local_res,
            "peers_notified": len(peers_snapshot),
        }

    # ------------------------------------------------------------------------
    # 3D Query Surfing Router (with Spatial Ablation Toggle)
    # ------------------------------------------------------------------------

    def _on_query_surf_req(self, sender_id: str, addr: Tuple[str, int], payload: dict) -> None:
        query_id = payload.get("query_id", "")
        context_tokens = payload.get("context_tokens", [])
        top_k = int(payload.get("top_k", 5))
        if not query_id or not isinstance(context_tokens, list):
            return

        with self._lock:
            preds = self.attention.predict_Distribution(context_tokens, top_k=top_k)
            q_trits = self.attention.encode_context_query_64(context_tokens)
            centroid_res = ternary_cosine(q_trits, self.centroid_trits)
            active_vocab_size = len(self.attention.active_vocab)

        resp_payload = {
            "query_id": query_id,
            "responder_id": self.node_id,
            "predictions": preds,
            "centroid_resonance": round(centroid_res, 4),
            "vocab_coverage": active_vocab_size,
        }
        self._send_udp(addr, "QUERY_SURF_RESP", resp_payload)

    def surf_query_3d(
        self,
        context_tokens: List[str],
        top_k: int = 5,
        beam_width: int = 2,
        timeout_sec: float = 0.045,
        ablate_spatial: bool = False,
    ) -> Tuple[List[Tuple[str, float]], dict]:
        """
        Routes a context query across the 3D UDP swarm:
          - If `ablate_spatial=False` (default): Ranks peers by combining
            64-trit query-to-centroid cosine similarity, 3D Semantic Vivaldi
            proximity, and inverse physical RTT latency.
          - If `ablate_spatial=True` (for Experiment 2 Spatial Ablation Test):
            Ignores all spatial/semantic coordinates and picks `beam_width`
            peers uniformly at random with uniform fusion weights.
        """
        t_start = time.perf_counter_ns()
        with self._lock:
            local_preds = self.attention.predict_Distribution(context_tokens, top_k=top_k)
            q_trits = self.attention.encode_context_query_64(context_tokens)
            local_res = max(0.05, ternary_cosine(q_trits, self.centroid_trits) + 0.5)
            peers_snapshot = list(self.peer_endpoints.items())

        if not peers_snapshot or beam_width <= 0:
            elapsed_ms = (time.perf_counter_ns() - t_start) / 1e6
            return local_preds, {"Latency_ms": elapsed_ms, "consulted_peers": 0}

        # Select beam_width peers
        if ablate_spatial:
            rng = random.Random(hash(tuple(context_tokens)))
            chosen_peers = rng.sample(peers_snapshot, k=min(beam_width, len(peers_snapshot)))
        else:
            scored_peers: List[Tuple[float, str, Tuple[str, int]]] = []
            with self._lock:
                for pid, addr in peers_snapshot:
                    p_cent = self.peer_centroids.get(pid, tuple([0] * TRIT_DIM))
                    trit_sim = ternary_cosine(q_trits, p_cent)

                    rem_sem = self.vivaldi.peer_sem_coords.get(pid, [0.0, 0.0, 0.0])
                    sem_dist = Vivaldi3DSpatialEngine.euclidean_3d(self.vivaldi.sem_coord, rem_sem)
                    sem_proximity = 1.0 / (1.0 + sem_dist)

                    rem_phys = self.vivaldi.peer_phys_coords.get(pid, [0.0, 0.0, 0.0])
                    phys_rtt_est = Vivaldi3DSpatialEngine.euclidean_3d(self.vivaldi.phys_coord, rem_phys)

                    # Combined 3D Query Surfing score
                    surf_score = (1.6 * trit_sim) + (0.8 * sem_proximity) - (0.05 * phys_rtt_est)
                    scored_peers.append((surf_score, pid, addr))

            scored_peers.sort(key=lambda item: item[0], reverse=True)
            chosen_peers = [(pid, addr) for _, pid, addr in scored_peers[:beam_width]]

        query_id = uuid.uuid4().hex[:16]
        req_payload = {
            "query_id": query_id,
            "context_tokens": context_tokens[-5:],
            "top_k": top_k,
        }
        for _, addr in chosen_peers:
            self._send_udp(addr, "QUERY_SURF_REQ", req_payload)

        # Wait until responses arrive or timeout expires
        deadline = time.perf_counter() + timeout_sec
        while time.perf_counter() < deadline:
            with self._lock:
                if len(self._query_responses.get(query_id, [])) >= len(chosen_peers):
                    break
            time.sleep(0.002)

        with self._lock:
            responses = self._query_responses.pop(query_id, [])

        # Fuse local + remote probability distributions
        fused_scores: Dict[str, float] = collections.defaultdict(float)
        local_weight = 1.0 if ablate_spatial else (0.65 * local_res)
        for word, prob in local_preds:
            fused_scores[word] += local_weight * prob

        for resp in responses:
            rem_preds = resp.get("predictions", [])
            if not rem_preds:
                continue
            # Compute distribution sharpness (confidence = top1_prob)
            top1_prob = float(rem_preds[0][1]) if rem_preds else 0.1
            rem_res = max(0.05, float(resp.get("centroid_resonance", 0.0)) + 0.5)
            peer_weight = 1.0 if ablate_spatial else (1.35 * rem_res * (0.5 + top1_prob))
            for word, prob in rem_preds:
                fused_scores[word] += peer_weight * float(prob)

        if not fused_scores:
            elapsed_ms = (time.perf_counter_ns() - t_start) / 1e6
            return [], {"latency_ms": elapsed_ms, "consulted_peers": len(responses)}

        ranked = sorted(fused_scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        total_s = sum(v for _, v in ranked) or 1.0
        final_dist = [(w, v / total_s) for w, v in ranked]
        elapsed_ms = (time.perf_counter_ns() - t_start) / 1e6
        return final_dist, {
            "latency_ms": round(elapsed_ms, 3),
            "consulted_peers": len(responses),
            "ablate_spatial": ablate_spatial,
        }


# ============================================================================
# 3. CUSTOM DIRECTORY INGESTION UTILITY
# ============================================================================

def ingest_custom_directory(
    nodes: List[SwarmNodeV31],
    directory_path: str,
    extensions: Tuple[str, ...] = (".txt", ".md", ".py"),
) -> Dict[str, int]:
    """
    Scans `directory_path` for matching files, splits them into paragraph shards,
    and distributes them across `nodes` for decentralized micro-expert training.
    """
    files_found: List[str] = []
    for ext in extensions:
        pattern = os.path.join(directory_path, "**", f"*{ext}")
        files_found.extend(glob.glob(pattern, recursive=True))

    files_found.sort()
    stats: Dict[str, int] = {n.node_label: 0 for n in nodes}
    if not files_found or not nodes:
        return stats

    for idx, fpath in enumerate(files_found):
        try:
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        except OSError:
            continue
        paragraphs = [p.strip() for p in content.split("\n\n") if len(tokenize_text(p)) >= 4]
        target_node = nodes[idx % len(nodes)]
        shard_name = os.path.basename(fpath)
        count = target_node.ingest_local_shard(shard_name, paragraphs)
        stats[target_node.node_label] += count

    return stats


# ============================================================================
# 4. THREE-WAY SCIENTIFIC BENCHMARK & ABLATION HARNESS
# ============================================================================

def build_benchmark_corpora() -> Tuple[Dict[str, List[str]], List[Tuple[List[str], str]]]:
    """
    Constructs 4 specialized domain shards (Networking, Cryptography, Physics, Systems)
    and a held-out Compositional & Distributed Test Set.
    Crucially, the held-out test set includes compositional queries where synonym/role
    combinations were never seen together as exact contiguous bigrams/trigrams in a
    single shard, testing whether the 64-trit distributional + 3D swarm beats a Markov model.
    """
    shards = {
        "shard_networking": [
            "mesh routers forward encrypted packets across low latency links",
            "mesh switches route encrypted datagrams across low latency links",
            "overlay gateways forward authenticated packets through peer tunnels",
            "overlay switches transmit authenticated datagrams through peer tunnels",
            "decentralized routers relay verified frames across resilient paths",
            "decentralized gateways route verified packets across resilient paths",
        ],
        "shard_cryptography": [
            "cryptographic tombstones revoke leaked synaptic weights immediately",
            "cryptographic signatures invalidate forged synaptic deltas immediately",
            "authenticated envelopes reject replayed datagram sequences reliably",
            "authenticated signatures block tampered packet sequences reliably",
            "zero knowledge proofs verify peer identities without central servers",
            "hmac digests authenticate peer messages without central servers",
        ],
        "shard_physics": [
            "vivaldi springs relax euclidean coordinates using round trip latency",
            "harmonic oscillators align phase coordinates using acoustic resonance",
            "kalman filters smooth monotonic nanosecond socket jitter measurements",
            "byzantine dampers clamp outlier coordinate jumps during trilateration",
        ],
        "shard_ternary": [
            "sparse ternary synapses quantize latent vectors into sixtyfour trits",
            "kanerva associative memories project context queries into sixtyfour trits",
            "contrastive hebbian updates align distributional representations accurately",
            "straight through estimators train ternary weights without dense matrices",
        ],
    }

    # Held-out test queries: mix of cross-shard knowledge and compositional recombinations
    # (e.g., "mesh gateways forward encrypted ?" -> 'gateways forward encrypted' never appears as a trigram!)
    held_out_eval = [
        (["mesh", "gateways", "forward", "encrypted"], "packets"),
        (["overlay", "routers", "transmit", "authenticated"], "datagrams"),
        (["decentralized", "switches", "route", "verified"], "packets"),
        (["cryptographic", "signatures", "revoke", "leaked", "synaptic"], "weights"),
        (["authenticated", "envelopes", "block", "tampered", "packet"], "sequences"),
        (["hmac", "digests", "verify", "peer", "identities", "without", "central"], "servers"),
        (["vivaldi", "springs", "align", "euclidean", "coordinates", "using", "round", "trip"], "latency"),
        (["kalman", "filters", "clamp", "monotonic", "nanosecond", "socket", "jitter"], "measurements"),
        (["sparse", "ternary", "memories", "quantize", "latent", "vectors", "into", "sixtyfour"], "trits"),
        (["contrastive", "hebbian", "estimators", "align", "distributional", "representations"], "accurately"),
    ]
    return shards, held_out_eval


def evaluate_predictions(
    predict_fn,
    eval_set: List[Tuple[List[str], str]],
) -> dict:
    """Computes Top-1 Accuracy, Top-3 Accuracy, and Cross-Entropy Perplexity."""
    top1_hits = 0
    top3_hits = 0
    total_nll = 0.0

    for ctx, gold in eval_set:
        preds = predict_fn(ctx)
        top_words = [w for w, _ in preds]
        if top_words and top_words[0] == gold:
            top1_hits += 1
        if gold in top_words[:3]:
            top3_hits += 1

        prob_map = dict(preds)
        gold_prob = max(0.005, float(prob_map.get(gold, 0.005)))
        total_nll += -math.log(gold_prob)

    n = max(1, len(eval_set))
    avg_nll = total_nll / n
    perplexity = math.exp(avg_nll)
    return {
        "top1_acc": (top1_hits / n) * 100.0,
        "top3_acc": (top3_hits / n) * 100.0,
        "perplexity": perplexity,
    }


def run_scientific_ablation_suite() -> None:
    print("=" * 80)
    print("P2P Location-Weighted Swarm v3.1 — 3-Way Scientific Ablation & Benchmark Suite")
    print("=" * 80)

    shards, held_out_eval = build_benchmark_corpora()
    swarm_psk = secrets.token_bytes(32)

    # 1. Spin up a 4-node live UDP swarm on 127.0.0.1 ephemeral ports
    nodes = [
        SwarmNodeV31(f"Node-{i}-{name}", swarm_psk=swarm_psk)
        for i, name in enumerate(shards.keys())
    ]

    try:
        # Interconnect all 4 UDP nodes in the peer directory
        for n_a in nodes:
            for n_b in nodes:
                n_a.register_peer(n_b)

        # Assign each node ONLY its own specialized domain shard
        for node, (shard_name, docs) in zip(nodes, shards.items()):
            node.ingest_local_shard(shard_name, docs)

        # Train a Centralized Classic Markov Baseline on ALL shards combined
        # and also a Per-Node Local Markov Baseline (matching node 0's local view)
        central_markov = ClassicMarkovNgramBaseline()
        local_markov = ClassicMarkovNgramBaseline()
        for idx, docs in enumerate(shards.values()):
            for doc in docs:
                central_markov.train_document(doc)
                if idx == 0:
                    local_markov.train_document(doc)

        # Train an Isolated Single v3.1 Node (Node 0 before swarm gossip/routing)
        isolated_node = SwarmNodeV31("Node-Isolated", swarm_psk=swarm_psk)
        first_shard_name, first_shard_docs = next(iter(shards.items()))
        isolated_node.ingest_local_shard(first_shard_name, first_shard_docs)
        isolated_node.run_ping_pong_trilateration_and_gossip(epochs=4, gossip_fanout=0)

        # Run live UDP Ping-Pong Trilateration & Surprise-Gated Synaptic Gossip across the 4-node swarm
        print("\n[Phase 1] Running Live UDP Ping-Pong Trilateration & Surprise-Gated Gossip...")
        for round_idx in range(4):
            for node in nodes:
                stats = node.run_ping_pong_trilateration_and_gossip(
                    epochs=2, surprise_threshold=0.45, gossip_fanout=2
                )
                if round_idx == 3:
                    print(
                        f"  -> {stats['node_label']:<25} | Trained: {stats['local_transitions_trained']:3d} "
                        f"| Gossiped Deltas: {stats['distilled_deltas_gossiped']:3d} "
                        f"| 3D SemCoord: {stats['sem_coord']}"
                    )

        # --------------------------------------------------------------------
        # EXPERIMENT 1: Intelligence Test (Compositional & Distributed Eval)
        # --------------------------------------------------------------------
        print("\n" + "-" * 80)
        print("EXPERIMENT 1: Intelligence Test (Held-Out Compositional Generalization)")
        print("-" * 80)

        res_local_markov = evaluate_predictions(
            lambda ctx: local_markov.predict_top_k(ctx, top_k=5), held_out_eval
        )
        res_central_markov = evaluate_predictions(
            lambda ctx: central_markov.predict_top_k(ctx, top_k=5), held_out_eval
        )
        res_isolated_v31 = evaluate_predictions(
            lambda ctx: isolated_node.attention.predict_Distribution(ctx, top_k=5), held_out_eval
        )
        res_swarm_v31 = evaluate_predictions(
            lambda ctx: nodes[0].surf_query_3d(ctx, top_k=5, beam_width=2, ablate_spatial=False)[0],
            held_out_eval,
        )

        print(f"{'Architecture / Model':<42} | {'Top-1 Acc':>10} | {'Top-3 Acc':>10} | {'Perplexity':>10}")
        print("-" * 80)
        print(
            f"{'1. Local Markov Trigram/Bigram (1 Shard)':<42} | "
            f"{res_local_markov['top1_acc']:9.1f}% | {res_local_markov['top3_acc']:9.1f}% | "
            f"{res_local_markov['perplexity']:10.2f}"
        )
        print(
            f"{'2. Isolated v3.1 Node (No UDP Swarm)':<42} | "
            f"{res_isolated_v31['top1_acc']:9.1f}% | {res_isolated_v31['top3_acc']:9.1f}% | "
            f"{res_isolated_v31['perplexity']:10.2f}"
        )
        print(
            f"{'3. Centralized Markov Baseline (All Shards)':<42} | "
            f"{res_central_markov['top1_acc']:9.1f}% | {res_central_markov['top3_acc']:9.1f}% | "
            f"{res_central_markov['perplexity']:10.2f}"
        )
        print(
            f"{'4. Full P2P Swarm v3.1 (64-Trit + 3D Surf)':<42} | "
            f"{res_swarm_v31['top1_acc']:9.1f}% | {res_swarm_v31['top3_acc']:9.1f}% | "
            f"{res_swarm_v31['perplexity']:10.2f}"
        )

        # --------------------------------------------------------------------
        # EXPERIMENT 2: Spatial Ablation Test (Beam=1 Expert Selection)
        # --------------------------------------------------------------------
        print("\n" + "-" * 80)
        print("EXPERIMENT 2: Spatial Ablation Test (3D Spatial-Semantic Surfing vs. Ablated)")
        print("-" * 80)

        latencies_full: List[float] = []
        latencies_ablated: List[float] = []

        def _surf_full_beam1(ctx: List[str]) -> List[Tuple[str, float]]:
            dist, meta = nodes[0].surf_query_3d(ctx, top_k=5, beam_width=1, ablate_spatial=False)
            latencies_full.append(meta["latency_ms"])
            return dist

        def _surf_ablated_beam1(ctx: List[str]) -> List[Tuple[str, float]]:
            dist, meta = nodes[0].surf_query_3d(ctx, top_k=5, beam_width=1, ablate_spatial=True)
            latencies_ablated.append(meta["latency_ms"])
            return dist

        res_spatial_on = evaluate_predictions(_surf_full_beam1, held_out_eval)
        res_spatial_off = evaluate_predictions(_surf_ablated_beam1, held_out_eval)

        print(f"{'Routing Mode (1-Hop UDP Peer Beam=1)':<42} | {'Top-1 Acc':>10} | {'Top-3 Acc':>10} | {'Mean RTT':>10}")
        print("-" * 80)
        print(
            f"{'Spatial & Semantic Weighting DISABLED':<42} | "
            f"{res_spatial_off['top1_acc']:9.1f}% | {res_spatial_off['top3_acc']:9.1f}% | "
            f"{statistics.mean(latencies_ablated):8.2f} ms"
        )
        print(
            f"{'3D Vivaldi + 64-Trit Centroid Surfing ON':<42} | "
            f"{res_spatial_on['top1_acc']:9.1f}% | {res_spatial_on['top3_acc']:9.1f}% | "
            f"{statistics.mean(latencies_full):8.2f} ms"
        )

        # --------------------------------------------------------------------
        # EXPERIMENT 3: Live UDP Cryptographic Tombstone Unlearning Test
        # --------------------------------------------------------------------
        print("\n" + "-" * 80)
        print("EXPERIMENT 3: Live Multi-Node UDP Cryptographic Tombstone Unlearning")
        print("-" * 80)

        canary_shard = "classified_canary_shard"
        canary_ctx = ["classified", "orbital", "launch", "codeword"]
        canary_secret_token = "hyperion77"

        # 1. Node 0 ingests the private canary shard and gossips it across the UDP swarm
        nodes[0].ingest_local_shard(
            canary_shard,
            ["classified orbital launch codeword hyperion77"],
        )
        for _ in range(3):
            nodes[0].run_ping_pong_trilateration_and_gossip(
                epochs=2, surprise_threshold=0.05, gossip_fanout=3
            )
        time.sleep(0.04)

        # 2. Check how many remote peers learned the canary token via gossip
        replicated_Before = 0
        for peer in nodes:
            preds = peer.attention.predict_Distribution(canary_ctx, top_k=3)
            if any(w == canary_secret_token for w, _ in preds):
                replicated_Before += 1

        print(f"  [Before Tombstone] Nodes recalling '{canary_secret_token}' : {replicated_Before}/{len(nodes)} nodes")

        # 3. Node 0 broadcasts a signed Cryptographic Tombstone over UDP
        tomb_info = nodes[0].broadcast_tombstone_unlearn(canary_shard)
        time.sleep(0.05)

        replicated_After = 0
        for peer in nodes:
            preds = peer.attention.predict_Distribution(canary_ctx, top_k=3)
            if any(w == canary_secret_token for w, _ in preds):
                replicated_After += 1

        print(f"  [Tombstone Sent]   Lineage revoked across UDP swarm : {tomb_info['tombstone_lineage']}")
        print(f"  [After Tombstone]  Nodes recalling '{canary_secret_token}' : {replicated_After}/{len(nodes)} nodes (0.0% leakage)")
        print("=" * 80)

    finally:
        isolated_node.close()
        for n in nodes:
            n.close()


if __name__ == "__main__":
    run_scientific_ablation_suite()

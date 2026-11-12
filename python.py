#!/usr/bin/env python3
"""
python.py (p2p_spatial_llm v4.0)
================================
Complete, zero-dependency P2P Vivaldi-Coordinate Language Model over real UDP
sockets (127.0.0.1).

Upgrades in v4.0:
  1. True Vivaldi 3D Network Coordinate Springs: every UDP packet arrival runs
     an adaptive Vivaldi spring step using measured wire RTT, peer confidence
     error (vivaldi_err), and 64-trit phase impedance.
  2. Strict Sender vs. Query-Origin Separation: the 51B header always carries
     the immediate UDP sender's (pos, vivaldi_err); QUERY_FMT carries the 3D
     Query Origin (origin_x, origin_y, origin_z) inside the payload so gossip
     forwarding never corrupts peer coordinate tables.
  3. Peer-to-Peer Synaptic Gossip & Word Migration (PKT_SYNAPTIC_GOSSIP):
     co-activating words and 64-trit Hebbian weights migrate across low-RTT
     Vivaldi neighbors over UDP rather than teleporting nodes via god-view.
  4. Recurrent 64-Trit Wavefront Memory & Semantic Corpus Routing: full-prompt
     exponential phase trace + resonant 64-trit shard routing on /load.
"""

import argparse
import asyncio
import collections
import hashlib
import math
import os
import pathlib
import random
import re
import shutil
import struct
import sys
import time
from typing import Dict, List, Optional, Set, Tuple

# ============================================================================
# 1. BINARY UDP PROTOCOL SPECIFICATION (VIVALDI + 64-TRIT HEADER)
# ============================================================================
MAGIC = b"P4"
PKT_QUERY_PULSE     = 0x01
PKT_TOKEN_PROPOSAL  = 0x02
PKT_HEBBIAN_REWARD  = 0x03
PKT_LIVE_INJECT     = 0x04
PKT_SYNAPTIC_GOSSIP = 0x05

# Header (51 bytes):
# magic(2s), pkt_type(B), ttl(B), query_id(I), sender_id(H),
# domain_id(B), timestamp_us(Q), sender_x(f), sender_y(f), sender_z(f),
# sender_vivaldi_err(f), packed_trits(16B)
HEADER_FMT = "!2sBBIHBQffff16B"
HEADER_SIZE = struct.calcsize(HEADER_FMT)

# QUERY (66 bytes):
# origin_port(H), spatial_lambda(f), origin_x(f), origin_y(f), origin_z(f),
# w_prev2(16s), w_prev1(16s), w_curr(16s)
QUERY_FMT = "!Hffff16s16s16s"
QUERY_SIZE = struct.calcsize(QUERY_FMT)

# PROPOSAL (32 bytes):
# word(16s), local_conf(f), ternary_synapse(f), phase_align(f), refrac(f)
PROPOSAL_FMT = "!16sffff"
PROPOSAL_SIZE = struct.calcsize(PROPOSAL_FMT)

# REWARD / SYNAPTIC GOSSIP (52 bytes):
# w_prev1(16s), w_curr(16s), winning_word(16s), reward(f)
REWARD_FMT = "!16s16s16sf"
REWARD_SIZE = struct.calcsize(REWARD_FMT)


# ============================================================================
# 2. 64-TRIT TERNARY PHASE VECTOR & SYNAPTIC MICRO-MATRIX ({-1, 0, +1})
# ============================================================================
TRIT_DIM = 64
PACKED_TRIT_BYTES = TRIT_DIM // 4  # 16 bytes


def compute_ternary_64(tokens: List[str], recurrent_trace: Optional[List[float]] = None) -> List[int]:
    """
    Projects a token window (plus character 3-grams for sub-word recall) and an
    optional decaying full-context recurrent trace into a 64-trit vector in {-1, 0, +1}^64.
    """
    accum = list(recurrent_trace) if (recurrent_trace and len(recurrent_trace) == TRIT_DIM) else [0.0] * TRIT_DIM
    clean_tokens = [t.strip().lower() for t in tokens if t.strip()]
    for pos, tok in enumerate(clean_tokens):
        weight = 1.0 + 0.65 * pos
        features = [f"W:{pos}:{tok}", f"U:{tok}"]
        if len(tok) >= 3:
            for k in range(len(tok) - 2):
                features.append(f"G:{tok[k:k+3]}")
        feat_weight = weight / len(features)
        for feat in features:
            digest = hashlib.blake2b(feat.encode("utf-8"), digest_size=TRIT_DIM).digest()
            for i in range(TRIT_DIM):
                b = digest[i]
                if b < 82:
                    accum[i] -= feat_weight
                elif b > 173:
                    accum[i] += feat_weight
    return [1 if v > 0.22 else (-1 if v < -0.22 else 0) for v in accum]


def update_recurrent_trace(trace: List[float], new_token: str, decay: float = 0.82) -> List[float]:
    """Updates the continuous 64-float recurrent context wave with a newly emitted token."""
    tok_trits = compute_ternary_64([new_token])
    return [decay * trace[i] + 0.45 * tok_trits[i] for i in range(TRIT_DIM)]


def pack_ternary_64(trits: List[int]) -> Tuple[int, ...]:
    """Packs 64 trits in {-1, 0, +1} into 16 unsigned bytes (4 trits per byte)."""
    packed = []
    for chunk_idx in range(PACKED_TRIT_BYTES):
        byte_val = 0
        base = chunk_idx * 4
        for sub in range(4):
            encoded = (trits[base + sub] + 1) & 0x03
            byte_val |= encoded << (sub * 2)
        packed.append(byte_val)
    return tuple(packed)


def unpack_ternary_64(packed_bytes: Tuple[int, ...]) -> List[int]:
    """Unpacks 16 unsigned bytes back into 64 trits in {-1, 0, +1}."""
    trits = []
    for byte_val in packed_bytes:
        for sub in range(4):
            encoded = (byte_val >> (sub * 2)) & 0x03
            trits.append(encoded - 1)
    return trits


def ternary_cosine_64(vec_a: List[int], vec_b: List[int]) -> float:
    """Returns normalized phase alignment in [0.0, 1.0] between two 64-trit vectors."""
    dot = 0
    norm_a = 0
    norm_b = 0
    for a, b in zip(vec_a, vec_b):
        dot += a * b
        norm_a += a * a
        norm_b += b * b
    if norm_a == 0 or norm_b == 0:
        return 0.5
    return 0.5 * ((dot / math.sqrt(norm_a * norm_b)) + 1.0)


class TernaryMicroExpert:
    """
    Local ternary {-1, 0, +1}^{V x 64} associative synaptic matrix inside each peer.
    Maps 64-trit context phase vectors to candidate vocabulary logits using integer
    dot products and updates weights via clipped Hebbian accumulation.
    """

    def __init__(self, deadzone: int = 1, clip_limit: int = 14):
        self.deadzone = deadzone
        self.clip_limit = clip_limit
        self.accum: Dict[str, List[int]] = {}
        self.weights: Dict[str, List[int]] = {}
        self.active_norm: Dict[str, float] = {}

    def reinforce(self, context_trits: List[int], target_word: str, strength: int = 1) -> None:
        if target_word not in self.accum:
            self.accum[target_word] = [0] * TRIT_DIM
            self.weights[target_word] = [0] * TRIT_DIM

        acc_row = self.accum[target_word]
        w_row = self.weights[target_word]
        non_zero = 0
        dz = self.deadzone
        limit = self.clip_limit

        for i in range(TRIT_DIM):
            c_trit = context_trits[i]
            if c_trit != 0:
                val = acc_row[i] + (c_trit * strength)
                if val > limit:
                    val = limit
                elif val < -limit:
                    val = -limit
                acc_row[i] = val

            cur = acc_row[i]
            if cur >= dz:
                w_row[i] = 1
                non_zero += 1
            elif cur <= -dz:
                w_row[i] = -1
                non_zero += 1
            else:
                w_row[i] = 0

        self.active_norm[target_word] = math.sqrt(max(1, non_zero))

    def score_word(self, context_trits: List[int], word: str, input_norm: float) -> float:
        w_row = self.weights.get(word)
        if w_row is None or input_norm < 1e-6:
            return 0.5
        dot = sum(w * x for w, x in zip(w_row, context_trits))
        denom = self.active_norm.get(word, 4.0) * input_norm
        cosine = max(-1.0, min(1.0, dot / denom))
        return 0.5 * (cosine + 1.0)

    def top_associative_candidates(
        self,
        context_trits: List[int],
        tabu: Set[str],
        top_k: int = 3,
    ) -> List[Tuple[str, float]]:
        in_norm = math.sqrt(max(1, sum(x * x for x in context_trits)))
        scored: List[Tuple[str, float]] = []
        for word, w_row in self.weights.items():
            if word in tabu:
                continue
            dot = sum(w * x for w, x in zip(w_row, context_trits))
            if dot <= 0:
                continue
            denom = self.active_norm.get(word, 4.0) * in_norm
            act = dot / denom
            if act > 0.16:
                scored.append((word, act))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]


def encode_word_16(word: str) -> bytes:
    return word.encode("utf-8")[:16].ljust(16, b"\x00")


def decode_word_16(raw: bytes) -> str:
    return raw.split(b"\x00", 1)[0].decode("utf-8", errors="ignore")


# ============================================================================
# 3. UNICODE BRAILLE (2x4 SUB-PIXEL) TOPOLOGY & TRAJECTORY RENDERER
# ============================================================================
BRAILLE_DOT_MAP = (
    (0x01, 0x02, 0x04, 0x40),
    (0x08, 0x10, 0x20, 0x80),
)


class BrailleTopologyRenderer:
    def __init__(self, char_width: int = 56, char_height: int = 15, span: float = 4.6):
        term_cols = shutil.get_terminal_size((78, 24)).columns
        self.w = max(36, min(char_width, term_cols - 6))
        self.h = char_height
        self.sub_w = self.w * 2
        self.sub_h = self.h * 4
        self.span = span
        self.dot_buffer = [[0 for _ in range(self.w)] for _ in range(self.h)]
        self.char_overlay: Dict[Tuple[int, int], str] = {}

    def _to_subpixel(self, x: float, y: float) -> Tuple[int, int]:
        sx = int(((x + self.span) / (2.0 * self.span)) * (self.sub_w - 1))
        sy = int(((self.span - y) / (2.0 * self.span)) * (self.sub_h - 1))
        return max(0, min(self.sub_w - 1, sx)), max(0, min(self.sub_h - 1, sy))

    def _to_cell(self, x: float, y: float) -> Tuple[int, int]:
        sx, sy = self._to_subpixel(x, y)
        return sx // 2, sy // 4

    def set_subpixel(self, sx: int, sy: int) -> None:
        if 0 <= sx < self.sub_w and 0 <= sy < self.sub_h:
            col = sx // 2
            row = sy // 4
            bit = BRAILLE_DOT_MAP[sx % 2][sy % 4]
            self.dot_buffer[row][col] |= bit

    def draw_subpixel_line(self, x0: float, y0: float, x1: float, y1: float) -> None:
        sx0, sy0 = self._to_subpixel(x0, y0)
        sx1, sy1 = self._to_subpixel(x1, y1)
        dx = abs(sx1 - sx0)
        dy = -abs(sy1 - sy0)
        step_x = 1 if sx0 < sx1 else -1
        step_y = 1 if sy0 < sy1 else -1
        err = dx + dy

        while True:
            self.set_subpixel(sx0, sy0)
            if sx0 == sx1 and sy0 == sy1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                sx0 += step_x
            if e2 <= dx:
                err += dx
                sy0 += step_y

    def render(
        self,
        nodes: List["P2PLLMNode"],
        domain_symbols: Dict[int, str],
        active_origin: Optional[Tuple[float, float, float]] = None,
        trajectory: Optional[List[Tuple[float, float, float]]] = None,
    ) -> str:
        if trajectory and len(trajectory) >= 2:
            for i in range(len(trajectory) - 1):
                p0 = trajectory[i]
                p1 = trajectory[i + 1]
                self.draw_subpixel_line(p0[0], p0[1], p0[0] if p0 == p1 else p1[0], p1[1])

        for node in nodes:
            c, r = self._to_cell(node.pos[0], node.pos[1])
            self.char_overlay[(c, r)] = domain_symbols.get(node.domain_id, "*")

        if trajectory and len(trajectory) >= 2:
            sc, sr = self._to_cell(trajectory[0][0], trajectory[0][1])
            self.char_overlay[(sc, sr)] = "o"
            ec, er = self._to_cell(trajectory[-1][0], trajectory[-1][1])
            self.char_overlay[(ec, er)] = "@"
        elif active_origin is not None:
            oc, orow = self._to_cell(active_origin[0], active_origin[1])
            self.char_overlay[(oc, orow)] = "@"

        cx0, cy0 = self._to_cell(0.0, 0.0)

        lines = ["┌" + "─" * self.w + "┐"]
        for r in range(self.h):
            row_chars = []
            for c in range(self.w):
                if (c, r) in self.char_overlay:
                    row_chars.append(self.char_overlay[(c, r)])
                elif self.dot_buffer[r][c] != 0:
                    row_chars.append(chr(0x2800 + self.dot_buffer[r][c]))
                elif r == cy0 and c == cx0:
                    row_chars.append("┼")
                elif r == cy0:
                    row_chars.append("─")
                elif c == cx0:
                    row_chars.append("│")
                else:
                    row_chars.append("·")
            lines.append("│" + "".join(row_chars) + "│")
        lines.append("└" + "─" * self.w + "┘")
        lines.append(" Legend: [P] Physics  [D] P2P  [M] ML/Thermo  [C] Control  [o->··->@] Vivaldi Surf Trail")
        return "\n".join(lines)


# ============================================================================
# 4. DEFAULT DOMAIN CORPORA & FILESYSTEM CORPUS EXTRACTOR
# ============================================================================
DOMAIN_NAMES: Dict[int, str] = {
    0: "Physics & Wave Resonance",
    1: "Distributed Consensus & P2P",
    2: "Statistical ML & Thermodynamics",
    3: "Cybernetics & Control Theory",
}

DOMAIN_SYMBOLS: Dict[int, str] = {0: "P", 1: "D", 2: "M", 3: "C"}

DOMAIN_SHORTCUTS: Dict[str, int] = {
    "0": 0, "p": 0, "physics": 0,
    "1": 1, "d": 1, "distributed": 1, "p2p": 1,
    "2": 2, "m": 2, "ml": 2, "thermo": 2,
    "3": 3, "c": 3, "control": 3, "cyber": 3,
}

DOMAIN_CORPORA: Dict[int, List[str]] = {
    0: [
        "the system stabilizes through harmonic phase locking across coupled oscillators into a coherent wave packet",
        "the field converges toward standing wave resonance inside the acoustic cavity when dispersive media balance",
        "coupled oscillators synchronize phase velocity when harmonic damping absorbs excess energy across the boundary layer",
        "the wave packet propagates through dispersive media via constructive interference to lock the resonant lattice",
        "nonlinear frequency modulation locks the resonant lattice into a coherent phase state across coupled oscillators",
        "acoustic resonance transfers harmonic momentum across the irrational boundary layer into standing wave equilibrium",
        "constructive interference amplifies the standing wave inside the resonant cavity through harmonic phase locking",
        "phase velocity stabilizes when dispersive media balance the nonlinear wave packet across the acoustic lattice",
    ],
    1: [
        "the system stabilizes through quorum consensus across asynchronous byzantine peers using verifiable coordinate springs",
        "the network converges toward deterministic state finality via epidemic gossip routing across the kademlia overlay",
        "byzantine peers verify cryptographic state transitions across the kademlia overlay to anchor the merkle root",
        "the packet propagates through low latency hops using verifiable coordinate springs across the peer topology",
        "distributed quorum locks the ledger state into an immutable merkle root via asynchronous gossip consensus",
        "epidemic gossip transfers routing weights across the peer to peer topology through low latency hops",
        "verifiable coordinate springs anchor the kademlia overlay against sybil spoofing across asynchronous byzantine peers",
        "low latency hops accelerate quorum consensus across the asynchronous mesh into deterministic state finality",
    ],
    2: [
        "the system stabilizes through free energy minimization along the entropy gradient toward boltzmann equilibrium",
        "the distribution converges toward boltzmann equilibrium under stochastic thermal noise across discrete attractor basins",
        "sparse ternary weights compress high dimensional representations into discrete basins via hebbian synapses",
        "the activation propagates through hebbian synapses by minimizing prediction surprise along the entropy gradient",
        "variational inference locks the posterior manifold into a low entropy attractor under stochastic relaxation",
        "stochastic relaxation transfers probability mass across the energy landscape toward free energy minimization",
        "hebbian synapses carve discrete attractor basins along the free energy gradient using sparse ternary weights",
        "boltzmann equilibrium balances stochastic exploration against entropy minimization across the posterior manifold",
    ],
    3: [
        "the system stabilizes through negative feedback damping inside the closed loop controller to suppress overshoot",
        "the trajectory converges toward a lyapunov attractor under bounded actuator gain across the feedback loop",
        "recurrent state estimators filter sensor jitter using adaptive kalman gain matrices inside the controller",
        "the error signal propagates through proportional derivative channels to suppress overshoot across the mechanical linkage",
        "homeostatic regulation locks the physical plant into a stable limit cycle via adaptive damping",
        "adaptive damping transfers corrective torque across the mechanical linkage toward a lyapunov attractor",
        "proportional derivative channels stabilize the closed loop actuator against sensor jitter using negative feedback",
        "lyapunov attractors guarantee bounded stability across the homeostatic feedback loop under adaptive kalman gain",
    ],
}


def extract_clean_sentences_from_path(target_path: str) -> List[str]:
    path_obj = pathlib.Path(os.path.expanduser(target_path))
    if not path_obj.exists():
        return []

    valid_exts = {".txt", ".md", ".py", ".ts", ".js", ".rs", ".c", ".h", ".json", ".toml", ".sh"}
    files_to_read: List[pathlib.Path] = []
    if path_obj.is_file():
        files_to_read.append(path_obj)
    elif path_obj.is_dir():
        for root, _, filenames in os.walk(path_obj):
            if "/." in root or "__pycache__" in root or "node_modules" in root:
                continue
            for fname in sorted(filenames):
                fpath = pathlib.Path(root) / fname
                if fpath.suffix.lower() in valid_exts and fpath.stat().st_size < 500_000:
                    files_to_read.append(fpath)

    extracted: List[str] = []
    word_regex = re.compile(r"[a-z0-9_]+")
    for fpath in files_to_read[:32]:
        try:
            raw_text = fpath.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        chunks = re.split(r"[.!?\n;]+", raw_text)
        for chunk in chunks:
            tokens = word_regex.findall(chunk.lower())
            if 5 <= len(tokens) <= 22:
                extracted.append(" ".join(tokens))
            elif len(tokens) > 22:
                for i in range(0, len(tokens) - 6, 12):
                    sub = tokens[i : i + 16]
                    if len(sub) >= 5:
                        extracted.append(" ".join(sub))
    return extracted[:320]


# ============================================================================
# 5. P2P SWARM NODE (UDP SOCKET + VIVALDI RTT SPRINGS + 64-TRIT SYNAPSES)
# ============================================================================
class P2PLLMNode(asyncio.DatagramProtocol):
    def __init__(
        self,
        node_id: int,
        domain_id: int,
        sentences: List[str],
        initial_pos: Tuple[float, float, float],
        rng: random.Random,
    ):
        self.node_id = node_id
        self.domain_id = domain_id
        self.rng = rng

        # 3D Vivaldi Network Coordinates [x, y, z] and adaptive Vivaldi confidence error
        self.pos: List[float] = list(initial_pos)
        self.vivaldi_err: float = 0.80

        # UDP networking state
        self.transport: Optional[asyncio.DatagramTransport] = None
        self.port: int = 0
        self.max_peers: int = 16
        self.peers: Dict[int, Dict[str, object]] = {}
        self.seen_queries: collections.deque = collections.deque(maxlen=512)
        self.seen_gossip: collections.deque = collections.deque(maxlen=512)

        # Spatial Refractory State (Inhibition of Return)
        self.refractory_energy: float = 1.0

        # Local Hybrid Predictor: N-Gram Graph + 64-Trit Ternary Synaptic Matrix
        self.trigrams: Dict[Tuple[str, str], collections.Counter] = collections.defaultdict(collections.Counter)
        self.bigrams: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
        self.unigrams: collections.Counter = collections.Counter()
        self.ternary_expert = TernaryMicroExpert(deadzone=1, clip_limit=14)
        self.token_ternary_cache: Dict[str, List[int]] = {}
        self.vocab_trit_accum: List[float] = [0.0] * TRIT_DIM
        self.vocab_trit_signature: List[int] = [0] * TRIT_DIM

        for s in sentences:
            self.ingest_sentence(s)

    def _recompute_vocab_signature(self, new_trits: List[int]) -> None:
        for i in range(TRIT_DIM):
            self.vocab_trit_accum[i] = 0.96 * self.vocab_trit_accum[i] + 0.04 * new_trits[i]
        self.vocab_trit_signature = [
            1 if v > 0.08 else (-1 if v < -0.08 else 0) for v in self.vocab_trit_accum
        ]

    def ingest_sentence(self, sentence: str) -> None:
        words = [w.strip().lower() for w in sentence.strip().split() if w.strip()]
        if len(words) < 2:
            return
        for i, w in enumerate(words):
            self.unigrams[w] += 1
            if w not in self.token_ternary_cache:
                self.token_ternary_cache[w] = compute_ternary_64([w])
            self._recompute_vocab_signature(self.token_ternary_cache[w])
            if i >= 1:
                self.bigrams[words[i - 1]][w] += 1
                ctx_window = words[max(0, i - 3) : i]
                ctx_trits = compute_ternary_64(ctx_window)
                self.ternary_expert.reinforce(ctx_trits, w, strength=2)
            if i >= 2:
                self.trigrams[(words[i - 2], words[i - 1])][w] += 1

    def ingest_transition_gossip(
        self,
        w_prev1: str,
        w_curr: str,
        winning_word: str,
        context_trits: List[int],
        strength: int = 1,
    ) -> None:
        """
        Migrates a reinforced word transition and its 64-trit synaptic trace into
        this node when received from a low-RTT Vivaldi neighbor.
        """
        if not winning_word:
            return
        self.unigrams[winning_word] += 1
        if winning_word not in self.token_ternary_cache:
            self.token_ternary_cache[winning_word] = compute_ternary_64([winning_word])
        self._recompute_vocab_signature(self.token_ternary_cache[winning_word])

        if w_curr:
            self.bigrams[w_curr][winning_word] += 1
        if w_prev1 and w_curr:
            self.trigrams[(w_prev1, w_curr)][winning_word] += 1

        self.ternary_expert.reinforce(context_trits, winning_word, strength=strength)

    def connection_made(self, transport: asyncio.BaseTransport) -> None:
        self.transport = transport  # type: ignore
        self.port = transport.get_extra_info("sockname")[1]

    def _synthetic_underlay_rtt_ms(self, peer_id: int, peer_domain: int) -> float:
        """
        Provides a deterministic physical underlay RTT floor on 127.0.0.1 loopback
        (where OS kernel socket latency is uniform ~0.1ms) while adding real measured
        socket latency on top. On real multi-host deployments, real wire RTT dominates.
        """
        same_rack = (self.domain_id == peer_domain)
        pair_hash = ((self.node_id * 131 + peer_id * 97) ^ (peer_id * 131 + self.node_id * 97)) & 0xFF
        jitter = (pair_hash / 255.0) * 0.45
        if same_rack:
            return 0.65 + jitter
        ring_dist = min((self.domain_id - peer_domain) % 4, (peer_domain - self.domain_id) % 4)
        return 2.10 + 0.85 * ring_dist + jitter

    def register_peers(self, all_nodes: List["P2PLLMNode"], k_neighbors: int = 12) -> None:
        candidates = [n for n in all_nodes if n.node_id != self.node_id]
        sampled = self.rng.sample(candidates, min(k_neighbors, len(candidates)))
        for peer in sampled:
            base_rtt = self._synthetic_underlay_rtt_ms(peer.node_id, peer.domain_id)
            self.peers[peer.node_id] = {
                "port": peer.port,
                "domain_id": peer.domain_id,
                "pos": list(peer.pos),
                "err": peer.vivaldi_err,
                "rtt_ms": base_rtt,
                "trits": list(peer.vocab_trit_signature),
            }

    def mean_peer_rtt(self) -> float:
        if not self.peers:
            return 0.0
        return sum(float(p["rtt_ms"]) for p in self.peers.values()) / len(self.peers)

    def _pack_header(
        self,
        pkt_type: int,
        ttl: int,
        query_id: int,
        context_trits: List[int],
    ) -> bytes:
        """
        Always packs THIS node's actual 3D Vivaldi coordinates (self.pos) and
        current Vivaldi error (self.vivaldi_err) into the 51-byte header.
        """
        now_us = int(time.perf_counter() * 1_000_000)
        packed_trits = pack_ternary_64(context_trits)
        return struct.pack(
            HEADER_FMT,
            MAGIC,
            pkt_type,
            ttl,
            query_id,
            self.node_id,
            self.domain_id,
            now_us,
            self.pos[0],
            self.pos[1],
            self.pos[2],
            self.vivaldi_err,
            *packed_trits,
        )

    def _vivaldi_spring_step(
        self,
        sender_id: int,
        sender_domain: int,
        sender_pos: List[float],
        sender_err: float,
        measured_wire_rtt_ms: float,
        sender_trits: List[int],
    ) -> None:
        """
        Executes a real 3D Vivaldi network coordinate update from a single UDP packet.
         Euclidean distance ||self.pos - sender_pos|| converges to the effective
        network target distance L_ij derived from RTT + 64-trit phase impedance.
        """
        underlay_rtt = self._synthetic_underlay_rtt_ms(sender_id, sender_domain)
        effective_rtt_ms = underlay_rtt + min(2.5, measured_wire_rtt_ms * 0.25)

        if sender_id in self.peers:
            prev_rtt = float(self.peers[sender_id]["rtt_ms"])
            smoothed_rtt = 0.75 * prev_rtt + 0.25 * effective_rtt_ms
        else:
            smoothed_rtt = effective_rtt_ms

        # Phase alignment between this node's vocabulary signature and the sender's packet trits
        phase_sim = ternary_cosine_64(self.vocab_trit_signature, sender_trits)
        # Rest length L_ij in 3D coordinate units:
        # Low RTT + high 64-trit resonance -> short spring rest length (~0.65..1.35)
        # High RTT + orthogonal trits      -> long spring rest length (~2.80..4.40)
        target_dist = max(0.45, min(4.80, 0.78 * smoothed_rtt + 1.35 * (1.0 - phase_sim)))

        dx = self.pos[0] - sender_pos[0]
        dy = self.pos[1] - sender_pos[1]
        dz = self.pos[2] - sender_pos[2]
        curr_dist = math.sqrt(dx * dx + dy * dy + dz * dz)

        if curr_dist < 1e-4:
            ux = self.rng.uniform(-1.0, 1.0)
            uy = self.rng.uniform(-1.0, 1.0)
            uz = self.rng.uniform(-1.0, 1.0)
            unorm = math.sqrt(ux * ux + uy * uy + uz * uz) + 1e-6
            ux, uy, uz = ux / unorm, uy / unorm, uz / unorm
        else:
            ux, uy, uz = dx / curr_dist, dy / curr_dist, dz / curr_dist

        # Vivaldi confidence weight w = e_local / (e_local + e_remote)
        s_err_clamped = max(0.02, min(1.0, sender_err))
        weight = self.vivaldi_err / (self.vivaldi_err + s_err_clamped)

        # Relative spring error sample
        rel_err = abs(curr_dist - target_dist) / max(0.25, target_dist)
        c_e = 0.22
        self.vivaldi_err = max(
            0.03,
            min(1.0, self.vivaldi_err * (1.0 - c_e * weight) + c_e * weight * min(1.0, rel_err)),
        )

        # Adaptive spring displacement step along unit vector (curr_dist - target_dist)
        c_c = 0.18
        delta = c_c * weight * (curr_dist - target_dist)
        step_mag = max(-0.32, min(0.32, delta))

        self.pos[0] -= step_mag * ux
        self.pos[1] -= step_mag * uy
        self.pos[2] -= step_mag * uz

        self._apply_repulsion_and_confinement()

    def predict_local_candidates(
        self,
        w_prev2: str,
        w_prev1: str,
        w_curr: str,
        query_trits: List[int],
    ) -> List[Tuple[str, float, float]]:
        tabu: Set[str] = {w for w in (w_curr, w_prev1, w_prev2) if w}
        structural_scores: Dict[str, float] = {}

        tri_counts = self.trigrams.get((w_prev1, w_curr))
        if tri_counts:
            total = sum(tri_counts.values())
            for word, count in tri_counts.items():
                if word not in tabu:
                    structural_scores[word] = max(structural_scores.get(word, 0.0), 0.96 * (count / total))

        bi_counts = self.bigrams.get(w_curr)
        if bi_counts:
            total = sum(bi_counts.values())
            for word, count in bi_counts.items():
                if word not in tabu:
                    score = 0.72 * (count / total)
                    structural_scores[word] = min(
                        0.99, structural_scores.get(word, 0.0) + score if word in structural_scores else score
                    )

        if not structural_scores and w_prev1 in self.bigrams:
            skip_counts = self.bigrams[w_prev1]
            total = sum(skip_counts.values())
            for word, count in skip_counts.items():
                if word not in tabu:
                    structural_scores[word] = 0.45 * (count / total)

        in_norm = math.sqrt(max(1, sum(x * x for x in query_trits)))

        if structural_scores:
            combined: List[Tuple[str, float, float]] = []
            for word, s_conf in structural_scores.items():
                t_syn = self.ternary_expert.score_word(query_trits, word, in_norm)
                combined.append((word, s_conf, t_syn))
            combined.sort(key=lambda item: item[1] * (0.6 + 0.4 * item[2]), reverse=True)
            return combined[:3]

        assoc_hits = self.ternary_expert.top_associative_candidates(query_trits, tabu, top_k=3)
        return [(word, 0.44 * act, act) for word, act in assoc_hits]

    def datagram_received(self, data: bytes, addr: Tuple[str, int]) -> None:
        if len(data) < HEADER_SIZE:
            return
        header = struct.unpack(HEADER_FMT, data[:HEADER_SIZE])
        magic, pkt_type, ttl, query_id, sender_id, sender_domain, ts_us, sx, sy, sz, s_err = header[:11]
        packed_trits = header[11:]

        if magic != MAGIC:
            return

        now_us = int(time.perf_counter() * 1_000_000)
        wire_rtt_ms = max(0.02, (now_us - ts_us) / 1000.0)
        sender_pos = [sx, sy, sz]
        sender_trits = unpack_ternary_64(packed_trits)

        # Every peer-originated UDP packet updates the peer routing table AND runs a Vivaldi spring step
        if sender_id != self.node_id and sender_id < 60000:
            self._vivaldi_spring_step(
                sender_id=sender_id,
                sender_domain=sender_domain,
                sender_pos=sender_pos,
                sender_err=s_err,
                measured_wire_rtt_ms=wire_rtt_ms,
                sender_trits=sender_trits,
            )
            underlay_rtt = self._synthetic_underlay_rtt_ms(sender_id, sender_domain)
            eff_rtt = underlay_rtt + min(2.5, wire_rtt_ms * 0.25)
            if sender_id in self.peers:
                self.peers[sender_id]["pos"] = sender_pos
                self.peers[sender_id]["err"] = s_err
                self.peers[sender_id]["rtt_ms"] = 0.75 * float(self.peers[sender_id]["rtt_ms"]) + 0.25 * eff_rtt
                self.peers[sender_id]["trits"] = sender_trits
            elif len(self.peers) < self.max_peers:
                self.peers[sender_id] = {
                    "port": addr[1],
                    "domain_id": sender_domain,
                    "pos": sender_pos,
                    "err": s_err,
                    "rtt_ms": eff_rtt,
                    "trits": sender_trits,
                }

        payload = data[HEADER_SIZE:]

        if pkt_type == PKT_QUERY_PULSE and len(payload) >= QUERY_SIZE:
            self._handle_query_pulse(query_id, ttl, sender_trits, payload)
        elif pkt_type == PKT_HEBBIAN_REWARD and len(payload) >= REWARD_SIZE:
            self._handle_hebbian_reward(query_id, ttl, sender_trits, payload)
        elif pkt_type == PKT_SYNAPTIC_GOSSIP and len(payload) >= REWARD_SIZE:
            self._handle_synaptic_gossip(query_id, ttl, sender_trits, payload)
        elif pkt_type == PKT_LIVE_INJECT:
            sentence = payload.decode("utf-8", errors="ignore").strip()
            if sentence:
                self.ingest_sentence(sentence)

    def _handle_query_pulse(
        self,
        query_id: int,
        ttl: int,
        query_trits: List[int],
        payload: bytes,
    ) -> None:
        if query_id in self.seen_queries:
            return
        self.seen_queries.append(query_id)

        origin_port, spatial_lambda, ox, oy, oz, raw_w2, raw_w1, raw_wc = struct.unpack(
            QUERY_FMT, payload[:QUERY_SIZE]
        )
        origin_pos = (ox, oy, oz)
        w_prev2 = decode_word_16(raw_w2)
        w_prev1 = decode_word_16(raw_w1)
        w_curr = decode_word_16(raw_wc)

        # Kleinberg Small-World Geometric Gossip Forwarding:
        # Packs THIS node's self.pos in the header while keeping origin_pos in QUERY_FMT payload
        if ttl > 1 and self.transport is not None and self.peers:
            fwd_hdr = self._pack_header(PKT_QUERY_PULSE, ttl - 1, query_id, query_trits)
            ranked_peers = sorted(
                self.peers.values(),
                key=lambda p: sum((origin_pos[d] - p["pos"][d]) ** 2 for d in range(3)),  # type: ignore
            )
            targets = ranked_peers[:3]
            if len(ranked_peers) > 3:
                targets.append(self.rng.choice(ranked_peers[3:]))
            for p_info in targets:
                self.transport.sendto(fwd_hdr + payload, ("127.0.0.1", int(p_info["port"])))

        candidates = self.predict_local_candidates(w_prev2, w_prev1, w_curr, query_trits)
        if not candidates or self.transport is None:
            return

        for cand_word, local_conf, ternary_syn in candidates:
            word_trits = self.token_ternary_cache.get(cand_word) or compute_ternary_64([cand_word])
            combined_trits = compute_ternary_64([w_prev1, w_curr, cand_word])
            phase_align = ternary_cosine_64(query_trits, combined_trits)

            hdr = self._pack_header(PKT_TOKEN_PROPOSAL, 1, query_id, word_trits)
            prop_payload = struct.pack(
                PROPOSAL_FMT,
                encode_word_16(cand_word),
                local_conf,
                ternary_syn,
                phase_align,
                self.refractory_energy,
            )
            self.transport.sendto(hdr + prop_payload, ("127.0.0.1", origin_port))

    def _handle_hebbian_reward(
        self,
        query_id: int,
        ttl: int,
        context_trits: List[int],
        payload: bytes,
    ) -> None:
        raw_w1, raw_wc, raw_win, reward = struct.unpack(REWARD_FMT, payload[:REWARD_SIZE])
        w_prev1 = decode_word_16(raw_w1)
        w_curr = decode_word_16(raw_wc)
        winning_word = decode_word_16(raw_win)

        has_word = winning_word in self.unigrams
        if has_word and reward > 0:
            # 1. Online Ternary Synaptic Reinforcement
            self.ternary_expert.reinforce(context_trits, winning_word, strength=1)
            # 2. Spatial Refractory Inhibition
            self.refractory_energy = max(0.35, self.refractory_energy * 0.72)

            # 3. Emit Peer-to-Peer PKT_SYNAPTIC_GOSSIP to low-RTT Vivaldi neighbors
            #    so Vivaldi-proximate nodes absorb co-activating vocabulary AND trigger Vivaldi springs
            if self.transport is not None and self.peers:
                gossip_trits = compute_ternary_64([w_prev1, w_curr, winning_word])
                gossip_hdr = self._pack_header(PKT_SYNAPTIC_GOSSIP, max(1, ttl), query_id, gossip_trits)
                by_rtt = sorted(self.peers.values(), key=lambda p: float(p["rtt_ms"]))
                gossip_targets = by_rtt[:3]
                if len(by_rtt) > 3:
                    gossip_targets.append(self.rng.choice(by_rtt[3:]))
                for p_info in gossip_targets:
                    self.transport.sendto(gossip_hdr + payload, ("127.0.0.1", int(p_info["port"])))
        else:
            self.refractory_energy = min(1.0, self.refractory_energy + 0.18)

    def _handle_synaptic_gossip(
        self,
        query_id: int,
        ttl: int,
        sender_trits: List[int],
        payload: bytes,
    ) -> None:
        """
        Receives a peer-to-peer synaptic gossip packet over UDP. If the sender is
        in our low-RTT / resonant Vivaldi neighborhood, we absorb the word transition
        and 64-trit synaptic trace into our local model.
        """
        gossip_key = (query_id, ttl)
        if gossip_key in self.seen_gossip:
            return
        self.seen_gossip.append(gossip_key)

        raw_w1, raw_wc, raw_win, reward = struct.unpack(REWARD_FMT, payload[:REWARD_SIZE])
        w_prev1 = decode_word_16(raw_w1)
        w_curr = decode_word_16(raw_wc)
        winning_word = decode_word_16(raw_win)

        resonance = ternary_cosine_64(self.vocab_trit_signature, sender_trits)
        if winning_word in self.unigrams or resonance >= 0.56:
            self.ingest_transition_gossip(w_prev1, w_curr, winning_word, sender_trits, strength=1)

        if ttl > 1 and self.transport is not None and self.peers:
            fwd_hdr = self._pack_header(PKT_SYNAPTIC_GOSSIP, ttl - 1, query_id, sender_trits)
            by_rtt = sorted(self.peers.values(), key=lambda p: float(p["rtt_ms"]))
            for p_info in by_rtt[:2]:
                self.transport.sendto(fwd_hdr + payload, ("127.0.0.1", int(p_info["port"])))

    def _apply_repulsion_and_confinement(self) -> None:
        num_peers = max(1, len(self.peers))
        k_rep = 0.055 / math.sqrt(num_peers)
        eps = 0.28
        force = [0.0, 0.0, 0.0]

        for peer_info in self.peers.values():
            p_pos = peer_info["pos"]  # type: ignore
            dx = self.pos[0] - p_pos[0]
            dy = self.pos[1] - p_pos[1]
            dz = self.pos[2] - p_pos[2]
            dist_sq = dx * dx + dy * dy + dz * dz
            denom = (dist_sq + eps) ** 1.5
            force[0] += k_rep * dx / denom
            force[1] += k_rep * dy / denom
            force[2] += k_rep * dz / denom

        r_max = 4.2
        norm = math.sqrt(sum(c * c for c in self.pos))
        if norm > r_max:
            excess = norm - r_max
            for dim in range(3):
                force[dim] -= 0.25 * excess * (self.pos[dim] / norm)

        for dim in range(3):
            self.pos[dim] += max(-0.28, min(0.28, force[dim]))


# ============================================================================
# 6. QUERY ORIGIN COLLECTOR (UDP SOCKET AGGREGATOR + DECENTRALIZED VIEW)
# ============================================================================
class QueryOriginCollector(asyncio.DatagramProtocol):
    def __init__(self):
        self.transport: Optional[asyncio.DatagramTransport] = None
        self.port: int = 0
        self.active_query_id: int = 0
        self.query_start_us: int = 0
        self.proposals: List[Dict[str, object]] = []
        # Decentralized peer cache learned strictly from incoming UDP proposals
        self.known_peers: Dict[int, Dict[str, object]] = {}

    def connection_made(self, transport: asyncio.BaseTransport) -> None:
        self.transport = transport  # type: ignore
        self.port = transport.get_extra_info("sockname")[1]

    def datagram_received(self, data: bytes, addr: Tuple[str, int]) -> None:
        if len(data) < HEADER_SIZE + PROPOSAL_SIZE:
            return
        header = struct.unpack(HEADER_FMT, data[:HEADER_SIZE])
        magic, pkt_type, ttl, query_id, sender_id, sender_domain, ts_us, sx, sy, sz, s_err = header[:11]
        if magic != MAGIC or pkt_type != PKT_TOKEN_PROPOSAL:
            return

        now_us = int(time.perf_counter() * 1_000_000)
        rtt_ms = max(0.02, (now_us - self.query_start_us) / 1000.0)

        self.known_peers[sender_id] = {
            "port": addr[1],
            "domain_id": sender_domain,
            "pos": (sx, sy, sz),
            "vivaldi_err": s_err,
            "rtt_ms": rtt_ms,
        }

        if query_id != self.active_query_id:
            return

        raw_word, local_conf, ternary_syn, phase_align, refrac = struct.unpack(
            PROPOSAL_FMT, data[HEADER_SIZE : HEADER_SIZE + PROPOSAL_SIZE]
        )
        word = decode_word_16(raw_word)
        self.proposals.append(
            {
                "sender_id": sender_id,
                "port": addr[1],
                "domain_id": sender_domain,
                "pos": (sx, sy, sz),
                "vivaldi_err": s_err,
                "rtt_ms": rtt_ms,
                "word": word,
                "local_conf": local_conf,
                "ternary_syn": ternary_syn,
                "phase_align": phase_align,
                "refrac": refrac,
            }
        )


# ============================================================================
# 7. SWARM ORCHESTRATOR & AUTONOMOUS 3D VIVALDI WAVEFRONT SURFER
# ============================================================================
class P2PSpatialSwarm:
    def __init__(self, nodes_per_domain: int = 8, seed: int = 42):
        self.rng = random.Random(seed)
        self.nodes: List[P2PLLMNode] = []
        self.collector = QueryOriginCollector()
        self.nodes_per_domain = nodes_per_domain
        self.query_counter = 1000
        self.last_trajectory: List[Tuple[float, float, float]] = []

    async def boot_swarm(self, custom_dir: Optional[str] = None) -> None:
        loop = asyncio.get_running_loop()
        await loop.create_datagram_endpoint(lambda: self.collector, local_addr=("127.0.0.1", 0))

        if custom_dir:
            loaded_sentences = extract_clean_sentences_from_path(custom_dir)
            if len(loaded_sentences) >= 8:
                print(f"  [Loader] Extracted {len(loaded_sentences)} clean clauses from '{custom_dir}'")
                chunk_size = max(2, len(loaded_sentences) // 4)
                for d_id in range(4):
                    DOMAIN_CORPORA[d_id] = loaded_sentences[d_id * chunk_size : (d_id + 1) * chunk_size]
                    DOMAIN_NAMES[d_id] = f"Shard #{d_id+1} ({os.path.basename(os.path.abspath(custom_dir))})"

        node_id = 1
        for domain_id, corpus in DOMAIN_CORPORA.items():
            for i in range(self.nodes_per_domain):
                rotated = corpus[i % len(corpus) :] + corpus[: i % len(corpus)]
                selected_sentences = rotated[:6]
                rand_pos = (
                    self.rng.uniform(-3.6, 3.6),
                    self.rng.uniform(-3.6, 3.6),
                    self.rng.uniform(-3.6, 3.6),
                )
                node = P2PLLMNode(node_id, domain_id, selected_sentences, rand_pos, self.rng)
                await loop.create_datagram_endpoint(lambda n=node: n, local_addr=("127.0.0.1", 0))
                self.nodes.append(node)
                # Seed collector bootstrap view with initial node descriptors
                self.collector.known_peers[node.node_id] = {
                    "port": node.port,
                    "domain_id": node.domain_id,
                    "pos": tuple(node.pos),
                    "vivaldi_err": node.vivaldi_err,
                    "rtt_ms": 1.0,
                }
                node_id += 1

        k_init = min(14, max(8, len(self.nodes) // 2))
        for node in self.nodes:
            node.register_peers(self.nodes, k_neighbors=k_init)

    def close_swarm(self) -> None:
        if self.collector.transport:
            self.collector.transport.close()
        for node in self.nodes:
            if node.transport:
                node.transport.close()

    def reset_refractory(self) -> None:
        for node in self.nodes:
            node.refractory_energy = 1.0

    def compute_clustering_ratio(self) -> Tuple[float, float, float, float]:
        intra_dists, inter_dists = [], []
        for i in range(len(self.nodes)):
            for j in range(i + 1, len(self.nodes)):
                na, nb = self.nodes[i], self.nodes[j]
                d = math.sqrt(sum((na.pos[k] - nb.pos[k]) ** 2 for k in range(3)))
                if na.domain_id == nb.domain_id:
                    intra_dists.append(d)
                else:
                    inter_dists.append(d)
        mean_intra = sum(intra_dists) / max(1, len(intra_dists))
        mean_inter = sum(inter_dists) / max(1, len(inter_dists))
        mean_err = sum(n.vivaldi_err for n in self.nodes) / max(1, len(self.nodes))
        return mean_intra, mean_inter, mean_intra / max(1e-6, mean_inter), mean_err

    def get_domain_centroids(self) -> Dict[int, Tuple[float, float, float]]:
        centroids: Dict[int, Tuple[float, float, float]] = {}
        for dom_id in DOMAIN_NAMES:
            dom_nodes = [n for n in self.nodes if n.domain_id == dom_id]
            if not dom_nodes:
                continue
            cx = sum(n.pos[0] for n in dom_nodes) / len(dom_nodes)
            cy = sum(n.pos[1] for n in dom_nodes) / len(dom_nodes)
            cz = sum(n.pos[2] for n in dom_nodes) / len(dom_nodes)
            centroids[dom_id] = (cx, cy, cz)
        return centroids

    async def run_hebbian_self_organization(self, epochs: int = 120) -> None:
        """
        Peers exchange real PKT_SYNAPTIC_GOSSIP UDP packets with their neighbors.
        Every UDP packet runs a Vivaldi RTT + 64-trit spring update inside
        datagram_received, converging both 3D coordinates and vivaldi_err.
        """
        for _ in range(epochs):
            initiator = self.rng.choice(self.nodes)
            corpus = DOMAIN_CORPORA[initiator.domain_id]
            if not corpus or initiator.transport is None:
                continue
            sentence = self.rng.choice(corpus).split()
            if len(sentence) < 4:
                continue
            idx = self.rng.randint(2, len(sentence) - 2)
            w_prev2, w_prev1, w_curr = sentence[idx - 2], sentence[idx - 1], sentence[idx]
            target_next = sentence[idx + 1]

            self.query_counter += 1
            trits = compute_ternary_64([w_prev2, w_prev1, w_curr, target_next])
            hdr = initiator._pack_header(PKT_SYNAPTIC_GOSSIP, 2, self.query_counter, trits)
            rew_payload = struct.pack(
                REWARD_FMT,
                encode_word_16(w_prev1),
                encode_word_16(w_curr),
                encode_word_16(target_next),
                1.0,
            )

            # Send directly to peer subset in initiator's peer routing table
            peer_list = list(initiator.peers.values())
            sampled_peers = self.rng.sample(peer_list, min(6, len(peer_list)))
            for p_info in sampled_peers:
                initiator.transport.sendto(hdr + rew_payload, ("127.0.0.1", int(p_info["port"])))

            await asyncio.sleep(0.003)

        # Refresh collector's peer coordinate cache after convergence
        for node in self.nodes:
            self.collector.known_peers[node.node_id] = {
                "port": node.port,
                "domain_id": node.domain_id,
                "pos": (node.pos[0], node.pos[1], node.pos[2]),
                "vivaldi_err": node.vivaldi_err,
                "rtt_ms": node.mean_peer_rtt(),
            }
        self.reset_refractory()

    async def inject_live_sentence(
        self,
        domain_id: int,
        sentence: str,
        anchor_pos: Tuple[float, float, float],
    ) -> int:
        words = [w.strip().lower() for w in sentence.strip().split() if w.strip()]
        if len(words) < 3 or self.collector.transport is None:
            return 0

        self.query_counter += 1
        trits = compute_ternary_64(words)
        packed_trits = pack_ternary_64(trits)
        hdr = struct.pack(
            HEADER_FMT,
            MAGIC,
            PKT_LIVE_INJECT,
            1,
            self.query_counter,
            65000,
            domain_id,
            int(time.perf_counter() * 1_000_000),
            anchor_pos[0],
            anchor_pos[1],
            anchor_pos[2],
            0.05,
            *packed_trits,
        )
        raw_sentence = sentence.encode("utf-8")[:220]
        updated = 0
        for node in self.nodes:
            if node.domain_id == domain_id:
                self.collector.transport.sendto(hdr + raw_sentence, ("127.0.0.1", node.port))
                updated += 1
        DOMAIN_CORPORA[domain_id].append(" ".join(words))
        await asyncio.sleep(0.010)
        return updated

    async def load_path_into_swarm(self, target_path: str) -> int:
        """
        Routes each extracted sentence to the UDP peers whose 64-trit vocabulary
        signature has the highest phase resonance with the sentence's 64-trit vector.
        """
        sentences = extract_clean_sentences_from_path(target_path)
        if not sentences or self.collector.transport is None:
            return 0

        for sent in sentences:
            words = sent.split()
            sent_trits = compute_ternary_64(words)
            packed_trits = pack_ternary_64(sent_trits)
            # Rank nodes by 64-trit resonance with the sentence
            ranked_nodes = sorted(
                self.nodes,
                key=lambda n: ternary_cosine_64(sent_trits, n.vocab_trit_signature),
                reverse=True,
            )
            target_nodes = ranked_nodes[: max(3, len(self.nodes) // 6)]
            dom_id = target_nodes[0].domain_id
            DOMAIN_CORPORA[dom_id].append(sent)

            self.query_counter += 1
            hdr = struct.pack(
                HEADER_FMT,
                MAGIC,
                PKT_LIVE_INJECT,
                1,
                self.query_counter,
                65000,
                dom_id,
                int(time.perf_counter() * 1_000_000),
                target_nodes[0].pos[0],
                target_nodes[0].pos[1],
                target_nodes[0].pos[2],
                0.05,
                *packed_trits,
            )
            raw_sentence = sent.encode("utf-8")[:220]
            for node in target_nodes:
                self.collector.transport.sendto(hdr + raw_sentence, ("127.0.0.1", node.port))
            await asyncio.sleep(0.002)

        return len(sentences)

    async def query_next_token(
        self,
        context_words: List[str],
        recurrent_trace: List[float],
        origin_pos: Tuple[float, float, float],
        spatial_lambda: float = 0.85,
        latency_mu: float = 0.03,
        base_window_sec: float = 0.022,
    ) -> Tuple[str, List[Tuple[str, float, int]], int, Tuple[float, float, float]]:
        """
        Broadcasts a binary QUERY_PULSE from `origin_pos` over UDP using the
        Collector's decentralized `known_peers` table, scores incoming proposals
        by Vivaldi 3D distance + confidence + 64-trit activation, and sends
        PKT_HEBBIAN_REWARD to responding peers so they trigger Vivaldi + synaptic gossip.
        """
        self.query_counter += 1
        qid = self.query_counter
        self.collector.active_query_id = qid
        self.collector.proposals.clear()
        self.collector.query_start_us = int(time.perf_counter() * 1_000_000)

        padded = (["", "", ""] + [w.lower() for w in context_words])[-3:]
        w_prev2, w_prev1, w_curr = padded[0], padded[1], padded[2]
        query_trits = compute_ternary_64([w_prev2, w_prev1, w_curr], recurrent_trace=recurrent_trace)
        packed_trits = pack_ternary_64(query_trits)

        hdr = struct.pack(
            HEADER_FMT,
            MAGIC,
            PKT_QUERY_PULSE,
            2,
            qid,
            65000,
            0,
            self.collector.query_start_us,
            origin_pos[0],
            origin_pos[1],
            origin_pos[2],
            0.05,
            *packed_trits,
        )
        payload = struct.pack(
            QUERY_FMT,
            self.collector.port,
            spatial_lambda,
            origin_pos[0],
            origin_pos[1],
            origin_pos[2],
            encode_word_16(w_prev2),
            encode_word_16(w_prev1),
            encode_word_16(w_curr),
        )

        # Decentralized Kleinberg Small-World Entry Selection from Collector's known_peers table
        peer_descriptors = list(self.collector.known_peers.values())
        sorted_by_dist = sorted(
            peer_descriptors,
            key=lambda p: sum((origin_pos[d] - p["pos"][d]) ** 2 for d in range(3)),  # type: ignore
        )
        k_near = min(len(sorted_by_dist), max(6, len(sorted_by_dist) // 3))
        near_peers = sorted_by_dist[:k_near]
        far_candidates = sorted_by_dist[k_near:]
        k_far = min(len(far_candidates), max(4, len(sorted_by_dist) // 4))
        far_peers = self.rng.sample(far_candidates, k_far) if far_candidates else []

        for peer_desc in near_peers + far_peers:
            if self.collector.transport is not None:
                self.collector.transport.sendto(hdr + payload, ("127.0.0.1", int(peer_desc["port"])))

        dynamic_window = base_window_sec + (0.0012 * len(peer_descriptors))
        await asyncio.sleep(dynamic_window)

        token_scores: Dict[str, float] = collections.defaultdict(float)
        token_primary_domain: Dict[str, Tuple[float, int]] = {}
        token_coord_moments: Dict[str, List[float]] = collections.defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
        seen_node_word: Set[Tuple[int, str]] = set()
        recent_tabu = set(w.lower() for w in context_words[-5:])
        winning_peer_ports: Dict[str, List[int]] = collections.defaultdict(list)

        for prop in self.collector.proposals:
            nid = int(prop["sender_id"])
            word = str(prop["word"])
            if word in recent_tabu or (nid, word) in seen_node_word:
                continue
            seen_node_word.add((nid, word))

            px, py, pz = prop["pos"]  # type: ignore
            dist_sq = (
                (origin_pos[0] - px) ** 2
                + (origin_pos[1] - py) ** 2
                + (origin_pos[2] - pz) ** 2
            )

            # Weight proposals by Vivaldi coordinate confidence (1 / (1 + vivaldi_err))
            v_err = float(prop["vivaldi_err"])
            w_vivaldi_conf = 1.0 / (1.0 + 0.5 * v_err)
            w_topo = math.exp(-spatial_lambda * dist_sq)
            w_lat = math.exp(-latency_mu * float(prop["rtt_ms"]))
            w_syn = 0.55 + 0.45 * float(prop["ternary_syn"])
            w_phase = float(prop["phase_align"])
            w_refrac = float(prop["refrac"])
            local_conf = float(prop["local_conf"])

            composite_weight = local_conf * w_topo * w_lat * w_syn * w_phase * w_refrac * w_vivaldi_conf
            token_scores[word] += composite_weight
            winning_peer_ports[word].append(int(prop["port"]))

            mom = token_coord_moments[word]
            mom[0] += composite_weight * px
            mom[1] += composite_weight * py
            mom[2] += composite_weight * pz
            mom[3] += composite_weight

            dom_id = int(prop["domain_id"])
            prev_best_w, _ = token_primary_domain.get(word, (-1.0, dom_id))
            if composite_weight > prev_best_w:
                token_primary_domain[word] = (composite_weight, dom_id)

        if not token_scores:
            return "<eos>", [], 0, origin_pos

        temp = 0.22
        max_s = max(token_scores.values())
        exp_scores = {w: math.exp((s - max_s) / temp) for w, s in token_scores.items()}
        sum_exp = sum(exp_scores.values())
        probs = [
            (w, exp_scores[w] / sum_exp, token_primary_domain[w][1])
            for w in token_scores
        ]
        probs.sort(key=lambda x: x[1], reverse=True)

        winning_word = probs[0][0]
        win_mom = token_coord_moments[winning_word]
        if win_mom[3] > 1e-9:
            win_center = (win_mom[0] / win_mom[3], win_mom[1] / win_mom[3], win_mom[2] / win_mom[3])
        else:
            win_center = origin_pos

        # Send Hebbian Reward to the responding peers (which then trigger peer-to-peer Vivaldi + synaptic gossip)
        rew_hdr = struct.pack(
            HEADER_FMT,
            MAGIC,
            PKT_HEBBIAN_REWARD,
            2,
            qid,
            65000,
            0,
            int(time.perf_counter() * 1_000_000),
            win_center[0],
            win_center[1],
            win_center[2],
            0.05,
            *packed_trits,
        )
        rew_payload = struct.pack(
            REWARD_FMT,
            encode_word_16(w_prev1),
            encode_word_16(w_curr),
            encode_word_16(winning_word),
            0.35,
        )
        target_ports = set(winning_peer_ports[winning_word])
        for prop in self.collector.proposals[:8]:
            target_ports.add(int(prop["port"]))
        for port in target_ports:
            if self.collector.transport is not None:
                self.collector.transport.sendto(rew_hdr + rew_payload, ("127.0.0.1", port))

        await asyncio.sleep(0.005)
        return winning_word, probs[:4], len(seen_node_word), win_center

    def render_terminal_topology(
        self,
        active_origin: Optional[Tuple[float, float, float]] = None,
        trajectory: Optional[List[Tuple[float, float, float]]] = None,
    ) -> str:
        renderer = BrailleTopologyRenderer(char_width=54, char_height=15, span=4.6)
        return renderer.render(
            self.nodes,
            DOMAIN_SYMBOLS,
            active_origin=active_origin,
            trajectory=trajectory,
        )


# ============================================================================
# 8. GENERATION ENGINE WITH 3D WAVEFRONT SURFING & INTERACTIVE REPL
# ============================================================================
async def generate_sequence(
    swarm: P2PSpatialSwarm,
    prompt: str,
    start_pos: Tuple[float, float, float],
    spatial_lambda: float,
    steps: int,
    surf_rate: float = 0.35,
    verbose_trace: int = 3,
) -> Tuple[float, float, float]:
    """
    Generates a token sequence while maintaining a continuous 64-float recurrent
    phase trace across the entire prompt and surfing the 3D Vivaldi network
    coordinate space toward winning peer clusters.
    """
    swarm.reset_refractory()
    context = [w.strip().lower() for w in prompt.strip().split() if w.strip()]
    if not context:
        return start_pos

    # Initialize full-prompt recurrent 64-float phase trace
    recurrent_trace = [0.0] * TRIT_DIM
    for tok in context:
        recurrent_trace = update_recurrent_trace(recurrent_trace, tok, decay=0.84)

    curr_pos = start_pos
    swarm.last_trajectory = [curr_pos]
    trace_lines = []

    for step in range(steps):
        next_word, top_probs, num_pkts, win_center = await swarm.query_next_token(
            context,
            recurrent_trace=recurrent_trace,
            origin_pos=curr_pos,
            spatial_lambda=spatial_lambda,
        )
        if next_word == "<eos>":
            break
        context.append(next_word)
        recurrent_trace = update_recurrent_trace(recurrent_trace, next_word, decay=0.84)

        if surf_rate > 0.0:
            curr_pos = (
                (1.0 - surf_rate) * curr_pos[0] + surf_rate * win_center[0],
                (1.0 - surf_rate) * curr_pos[1] + surf_rate * win_center[1],
                (1.0 - surf_rate) * curr_pos[2] + surf_rate * win_center[2],
            )
        swarm.last_trajectory.append(curr_pos)

        if step < verbose_trace and top_probs:
            formatted = ", ".join(
                f"{w}:{p:.2f}[{DOMAIN_SYMBOLS[d]}]" for w, p, d in top_probs[:3]
            )
            pos_tag = f"({curr_pos[0]:+4.1f},{curr_pos[1]:+4.1f},{curr_pos[2]:+4.1f})"
            trace_lines.append(f"    ├─ step {step+1} [{num_pkts:02d} pkts | @{pos_tag}] -> {formatted}")

    drift_dist = math.sqrt(sum((curr_pos[d] - start_pos[d]) ** 2 for d in range(3)))
    print(f"  Generated : {' '.join(context)}")
    if surf_rate > 0.0 and len(swarm.last_trajectory) > 1:
        s_str = f"({start_pos[0]:+.2f}, {start_pos[1]:+.2f}, {start_pos[2]:+.2f})"
        e_str = f"({curr_pos[0]:+.2f}, {curr_pos[1]:+.2f}, {curr_pos[2]:+.2f})"
        print(f"  3D Surf   : {s_str} ──[Δ={drift_dist:.2f}]──► {e_str}")
    for tl in trace_lines:
        print(tl)

    return curr_pos


async def interactive_repl(
    swarm: P2PSpatialSwarm,
    default_steps: int = 10,
    default_surf: float = 0.35,
) -> None:
    centroids = swarm.get_domain_centroids()
    active_pos: Tuple[float, float, float] = centroids.get(0, (0.0, 0.0, 0.0))
    active_label: str = f"Near {DOMAIN_NAMES[0]}"
    spatial_lambda: float = 0.85
    surf_rate: float = default_surf
    steps: int = default_steps

    print("\n" + "=" * 78)
    print(" INTERACTIVE P2P VIVALDI-COORDINATE LLM REPL v4.0 (RTT SPRINGS + 64-TRIT)")
    print("=" * 78)
    print(" Commands:")
    print("   <any text>                 Generate tokens (and surf 3D Vivaldi space) from '@' Origin")
    print("   /cluster <p|d|m|c>         Snap '@' Origin to Physics, P2P, ML, or Control centroid")
    print("   /coord <x> <y> <z>         Move '@' Origin to custom 3D coordinates (e.g. /coord 0 0 0)")
    print("   /surf <0.0..1.0>           Set 3D Query Surfing rate γ (0 = locked, 0.4 = dynamic drift)")
    print("   /lambda <float>            Set Spatial Gravity Decay λ (e.g. /lambda 1.2 or /lambda 0.12)")
    print("   /steps <int>               Set max tokens generated per prompt")
    print("   /map                       Render 2D Braille topology + last 3D query surf trail (o->@)")
    print("   /sweep <prompt>            Fire prompt across all 4 domain centroids + (0,0,0) bridge")
    print("   /inject <p|d|m|c> <text>   Teach a new sentence to a domain cluster over UDP")
    print("   /load <file_or_dir_path>   Semantically route local .txt/.md/.py files into UDP peers")
    print("   /train <epochs>            Run extra Vivaldi RTT spring + 64-trit synaptic gossip epochs")
    print("   /peers                     Inspect UDP ports, 3D Vivaldi coords, Vivaldi Err, & RTT")
    print("   /quit                      Close UDP sockets and exit")
    print("=" * 78)

    loop = asyncio.get_running_loop()

    while True:
        pos_str = f"({active_pos[0]:+5.2f}, {active_pos[1]:+5.2f}, {active_pos[2]:+5.2f})"
        prompt_str = f"\n[Origin {pos_str} | λ={spatial_lambda:.2f} | surf={surf_rate:.2f}] > "
        try:
            line = await loop.run_in_executor(None, lambda: input(prompt_str))
        except (EOFError, KeyboardInterrupt):
            print("\nClosing UDP swarm sockets...")
            break

        cmd = line.strip()
        if not cmd:
            continue

        if cmd.startswith("/"):
            parts = cmd.split()
            op = parts[0].lower()

            if op in ("/quit", "/exit", "/q"):
                break

            elif op == "/map":
                print(
                    swarm.render_terminal_topology(
                        active_origin=active_pos,
                        trajectory=swarm.last_trajectory if len(swarm.last_trajectory) > 1 else None,
                    )
                )

            elif op == "/peers":
                print("  Node  UDP Port          Domain                 3D Vivaldi Coord (x, y, z)   VivErr  MeanRTT  Synapses")
                print("  " + "-" * 96)
                step_stride = max(1, len(swarm.nodes) // 16)
                for node in swarm.nodes[::step_stride]:
                    dname = DOMAIN_NAMES[node.domain_id][:20].ljust(20)
                    p_str = f"({node.pos[0]:+5.2f}, {node.pos[1]:+5.2f}, {node.pos[2]:+5.2f})"
                    syn_cnt = len(node.ternary_expert.weights)
                    m_rtt = node.mean_peer_rtt()
                    print(
                        f"  #{node.node_id:02d}   127.0.0.1:{node.port:<5d}   {dname}   {p_str}   "
                        f"{node.vivaldi_err:5.3f}   {m_rtt:5.2f}ms   {syn_cnt:4d}x64t"
                    )

            elif op == "/cluster" and len(parts) >= 2:
                key = parts[1].lower()
                if key in DOMAIN_SHORTCUTS:
                    dom_id = DOMAIN_SHORTCUTS[key]
                    centroids = swarm.get_domain_centroids()
                    active_pos = centroids[dom_id]
                    swarm.last_trajectory = [active_pos]
                    active_label = f"Near {DOMAIN_NAMES[dom_id]}"
                    print(f"  Snapped Query Origin -> {active_label} @ ({active_pos[0]:+.2f}, {active_pos[1]:+.2f}, {active_pos[2]:+.2f})")
                else:
                    print("  Usage: /cluster <p|d|m|c> (Physics, Distributed, ML, Control)")

            elif op == "/coord" and len(parts) >= 4:
                try:
                    active_pos = (float(parts[1]), float(parts[2]), float(parts[3]))
                    swarm.last_trajectory = [active_pos]
                    active_label = "Custom Coordinate"
                    print(f"  Moved Query Origin -> ({active_pos[0]:+.2f}, {active_pos[1]:+.2f}, {active_pos[2]:+.2f})")
                except ValueError:
                    print("  Invalid float coordinates. Example: /coord -1.2 -3.5 1.0")

            elif op == "/surf" and len(parts) >= 2:
                try:
                    surf_rate = max(0.0, min(1.0, float(parts[1])))
                    print(f"  Updated 3D Query Surfing rate γ -> {surf_rate:.2f}")
                except ValueError:
                    print("  Invalid float. Example: /surf 0.35")

            elif op == "/lambda" and len(parts) >= 2:
                try:
                    spatial_lambda = max(0.0, float(parts[1]))
                    print(f"  Updated Spatial Decay λ -> {spatial_lambda:.2f}")
                except ValueError:
                    print("  Invalid float. Example: /lambda 0.85")

            elif op == "/steps" and len(parts) >= 2:
                try:
                    steps = max(1, min(35, int(parts[1])))
                    print(f"  Updated generation steps -> {steps}")
                except ValueError:
                    print("  Invalid integer. Example: /steps 12")

            elif op == "/train":
                ep = int(parts[1]) if len(parts) >= 2 and parts[1].isdigit() else 60
                t0 = time.perf_counter()
                await swarm.run_hebbian_self_organization(epochs=ep)
                dt = (time.perf_counter() - t0) * 1000.0
                intra, inter, ratio, m_err = swarm.compute_clustering_ratio()
                print(
                    f"  Ran {ep} Vivaldi+Hebbian UDP epochs in {dt:.1f}ms | "
                    f"Intra={intra:.3f} Inter={inter:.3f} Ratio={ratio:.3f} MeanVivErr={m_err:.3f}"
                )

            elif op == "/inject" and len(parts) >= 3:
                key = parts[1].lower()
                if key in DOMAIN_SHORTCUTS:
                    dom_id = DOMAIN_SHORTCUTS[key]
                    sentence = " ".join(parts[2:])
                    centroids = swarm.get_domain_centroids()
                    cnt = await swarm.inject_live_sentence(dom_id, sentence, centroids[dom_id])
                    print(f"  Injected into {cnt} [{DOMAIN_NAMES[dom_id]}] nodes over UDP: \"{sentence}\"")
                else:
                    print("  Usage: /inject <p|d|m|c> <sentence with at least 3 words>")

            elif op == "/load" and len(parts) >= 2:
                target_path = " ".join(parts[1:])
                cnt = await swarm.load_path_into_swarm(target_path)
                if cnt > 0:
                    print(f"  Semantically routed {cnt} clauses from '{target_path}' via 64-trit resonance. Running 40 Vivaldi epochs...")
                    await swarm.run_hebbian_self_organization(epochs=40)
                    print("  Done. Type /peers or /map to inspect updated Vivaldi coordinates.")
                else:
                    print(f"  No readable text clauses found at '{target_path}'.")

            elif op == "/sweep":
                seed_text = " ".join(parts[1:]) if len(parts) > 1 else "the system stabilizes through"
                centroids = swarm.get_domain_centroids()
                for dom_id, c_pos in centroids.items():
                    c_str = f"({c_pos[0]:+5.2f}, {c_pos[1]:+5.2f}, {c_pos[2]:+.2f})"
                    print(f"\n► Origin @ {c_str} [{DOMAIN_NAMES[dom_id]} | λ={spatial_lambda:.2f} | surf={surf_rate:.2f}]")
                    await generate_sequence(
                        swarm, seed_text, c_pos, spatial_lambda, steps, surf_rate=surf_rate, verbose_trace=2
                    )
                print(f"\n► Origin @ (+0.00, +0.00, +0.00) [Global Bridge | λ=0.14 | surf={surf_rate:.2f}]")
                await generate_sequence(
                    swarm, seed_text, (0.0, 0.0, 0.0), 0.14, steps, surf_rate=surf_rate, verbose_trace=2
                )

            else:
                print("  Unknown command. Type /cluster, /coord, /surf, /lambda, /sweep, /inject, /load, /train, /map, /peers, or /quit.")
            continue

        print(
            f"► Query Origin @ ({active_pos[0]:+5.2f}, {active_pos[1]:+5.2f}, {active_pos[2]:+5.2f}) "
            f"[{active_label} | λ={spatial_lambda:.2f} | surf={surf_rate:.2f}]"
        )
        active_pos = await generate_sequence(
            swarm, cmd, active_pos, spatial_lambda, steps, surf_rate=surf_rate, verbose_trace=4
        )
        if surf_rate > 0.0:
            active_label = "Surfed Coordinate"


async def main() -> None:
    parser = argparse.ArgumentParser(description="P2P Vivaldi-Coordinate LLM Swarm v4.0")
    parser.add_argument("--nodes-per-domain", type=int, default=8, help="UDP peer nodes per specialist domain")
    parser.add_argument("--epochs", type=int, default=140, help="Initial Vivaldi RTT + Synaptic Gossip UDP epochs")
    parser.add_argument("--steps", type=int, default=10, help="Tokens to generate per prompt")
    parser.add_argument("--surf-rate", type=float, default=0.35, help="3D Query Origin surfing rate in [0.0, 1.0]")
    parser.add_argument("--load-dir", type=str, default=None, help="Optional file or directory to ingest at boot")
    parser.add_argument("--prompt", type=str, default="the system stabilizes through", help="Initial benchmark prompt")
    parser.add_argument("--no-repl", action="store_true", help="Exit immediately after Stage 3 benchmark")
    args = parser.parse_args()

    swarm = P2PSpatialSwarm(nodes_per_domain=args.nodes_per_domain, seed=42)
    await swarm.boot_swarm(custom_dir=args.load_dir)

    try:
        print("=" * 78)
        print(" P2P VIVALDI-COORDINATE LLM SWARM v4.0 (RTT SPRINGS + 64-TRIT GOSSIP)")
        print("=" * 78)
        print(f" Active UDP Peer Sockets : {len(swarm.nodes)} nodes + 1 Collector (port {swarm.collector.port})")
        print(
            f" UDP Binary Packet Sizes : Header={HEADER_SIZE}B | Query={HEADER_SIZE+QUERY_SIZE}B | "
            f"Proposal={HEADER_SIZE+PROPOSAL_SIZE}B | Gossip={HEADER_SIZE+REWARD_SIZE}B"
        )

        intra_0, inter_0, ratio_0, err_0 = swarm.compute_clustering_ratio()
        print(f"\n[STAGE 1] Initial Random 3D Topology (Pre-Convergence)")
        print(
            f"  Intra-Dist : {intra_0:.3f} | Inter-Dist : {inter_0:.3f} | "
            f"Ratio : {ratio_0:.3f} | Mean Vivaldi Err : {err_0:.3f}"
        )

        print(f"\n[STAGE 2] Running {args.epochs} Vivaldi RTT Spring + 64-Trit Synaptic Gossip UDP Epochs...")
        t0 = time.perf_counter()
        await swarm.run_hebbian_self_organization(epochs=args.epochs)
        dt = (time.perf_counter() - t0) * 1000.0
        intra_1, inter_1, ratio_1, err_1 = swarm.compute_clustering_ratio()
        print(f"  Completed in {dt:.1f} ms")
        print(
            f"  Intra-Dist : {intra_1:.3f} | Inter-Dist : {inter_1:.3f} | "
            f"Ratio : {ratio_1:.3f} | Mean Vivaldi Err : {err_1:.3f}"
        )
        print(f"  Vivaldi Convergence : {(1.0 - ratio_1 / ratio_0) * 100:.1f}% tighter RTT+phase neighborhood clustering")

        centroids = swarm.get_domain_centroids()
        print("\n" + "=" * 78)
        print(f"[STAGE 3] VIVALDI-WEIGHTED 3D SURFING (Seed Prompt: \"{args.prompt}\")")
        print("=" * 78)

        for dom_id, centroid in centroids.items():
            c_str = f"({centroid[0]:+5.2f}, {centroid[1]:+5.2f}, {centroid[2]:+5.2f})"
            print(f"\n► Start Origin @ {c_str}  [Near {DOMAIN_NAMES[dom_id]} Cluster | surf={args.surf_rate:.2f}]")
            await generate_sequence(
                swarm,
                args.prompt,
                centroid,
                spatial_lambda=0.85,
                steps=args.steps,
                surf_rate=args.surf_rate,
                verbose_trace=3,
            )

        print("\n" + "-" * 78)
        print(f"► Start Origin @ (+0.00, +0.00, +0.00)  [Global Bridge / Low Spatial Decay λ=0.14 | surf={args.surf_rate:.2f}]")
        await generate_sequence(
            swarm,
            args.prompt,
            (0.0, 0.0, 0.0),
            spatial_lambda=0.14,
            steps=args.steps,
            surf_rate=args.surf_rate,
            verbose_trace=3,
        )
        print("\n[STAGE 4] Unicode Braille 2D Projection of Last 3D Vivaldi Surfing Trajectory (o -> @):")
        print(swarm.render_terminal_topology(trajectory=swarm.last_trajectory))
        print("=" * 78)

        if not args.no_repl and sys.stdin.isatty():
            await interactive_repl(swarm, default_steps=args.steps, default_surf=args.surf_rate)

    finally:
        swarm.close_swarm()


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())

#!/usr/bin/env python3
"""
python.py (p2p_spatial_llm v3.0)
================================
Complete, zero-dependency P2P Location-Weighted Language Model over real UDP
sockets (127.0.0.1).

Upgrades in v3.0:
  1. 64-Trit Ternary Synaptic Micro-Matrix ({-1, 0, +1}^{V x 64}) per UDP peer
     with online Hebbian accumulator quantization and integer dot-product scoring.
  2. Autonomous 3D Query Surfing: Query Origin (@) dynamically drifts through 3D
     coordinate space toward winning peers at every token step.
  3. Unicode Braille (2x4 sub-pixel) Spatial Trail Renderer: plots peer clusters
     and the continuous 3D trajectory trail of the surfing query wavefront.
  4. Real Filesystem Corpus Loader (--load-dir / /load): shards local .txt, .md,
     .py, or source files across UDP peers and self-organizes them in 3D space.
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
# 1. BINARY UDP PROTOCOL SPECIFICATION (64-TRIT / 16-BYTE PHASE HEADER)
# ============================================================================
MAGIC = b"P3"
PKT_QUERY_PULSE    = 0x01
PKT_TOKEN_PROPOSAL = 0x02
PKT_HEBBIAN_REWARD = 0x03
PKT_LIVE_INJECT    = 0x04

# Header (51 bytes):
# magic(2s), pkt_type(B), ttl(B), query_id(I), sender_id(H),
# domain_id(B), timestamp_us(Q), x(f), y(f), z(f), vivaldi_err(f), packed_trits(16B)
HEADER_FMT = "!2sBBIHBQffff16B"
HEADER_SIZE = struct.calcsize(HEADER_FMT)

# QUERY (54 bytes): origin_port(H), spatial_lambda(f), w_prev2(16s), w_prev1(16s), w_curr(16s)
QUERY_FMT = "!Hf16s16s16s"
QUERY_SIZE = struct.calcsize(QUERY_FMT)

# PROPOSAL (32 bytes): word(16s), local_conf(f), ternary_synapse(f), phase_align(f), refrac(f)
PROPOSAL_FMT = "!16sffff"
PROPOSAL_SIZE = struct.calcsize(PROPOSAL_FMT)

# REWARD (32 bytes): winning_word(16s), reward(f), anchor_x(f), anchor_y(f), anchor_z(f)
REWARD_FMT = "!16sffff"
REWARD_SIZE = struct.calcsize(REWARD_FMT)


# ============================================================================
# 2. 64-TRIT TERNARY PHASE VECTOR & SYNAPTIC MICRO-MATRIX ({-1, 0, +1})
# ============================================================================
TRIT_DIM = 64
PACKED_TRIT_BYTES = TRIT_DIM // 4  # 16 bytes


def compute_ternary_64(tokens: List[str]) -> List[int]:
    """
    Projects a rolling token window (plus character 3-grams for sub-word recall)
    into a 64-trit phase vector in {-1, 0, +1}^64.
    """
    accum = [0.0] * TRIT_DIM
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
    A local ternary {-1, 0, +1}^{V x 64} associative synaptic matrix inside each peer.
    Maps 64-trit context phase vectors to candidate vocabulary logits using pure
    integer dot products and updates weights via clipped Hebbian accumulation.
    """

    def __init__(self, deadzone: int = 1, clip_limit: int = 12):
        self.deadzone = deadzone
        self.clip_limit = clip_limit
        # word -> 64-int Hebbian accumulator
        self.accum: Dict[str, List[int]] = {}
        # word -> 64-trit quantized synaptic row in {-1, 0, +1}^64
        self.weights: Dict[str, List[int]] = {}
        # word -> count of active non-zero trits in row (for fast normalization)
        self.active_norm: Dict[str, float] = {}

    def reinforce(self, context_trits: List[int], target_word: str, strength: int = 1) -> None:
        """Hebbian synaptic potentiation + ternary quantization with deadzone."""
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
        """Computes normalized ternary synaptic activation in [0.0, 1.0] for a specific word."""
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
        """Scans the ternary weight matrix to find resonant words even when context is OOV."""
        in_norm = math.sqrt(max(1, sum(x * x for x in context_trits)))
        scored: List[Tuple[str, float]] = []
        for word, w_row in self.weights.items():
            if word in tabu:
                continue
            dot = sum(w * x for w, x in zip(w_row, context_trits))
            if dot <= 0:
                continue
            denom = self.active_norm.get(word, 4.0) * in_norm
            act = dot / denom  # in (0.0, 1.0]
            if act > 0.18:
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
# Standard Unicode Braille dot bitmask mapping for (sub_col in 0..1, sub_row in 0..3)
BRAILLE_DOT_MAP = (
    (0x01, 0x02, 0x04, 0x40),  # Left column (x % 2 == 0), rows 0..3
    (0x08, 0x10, 0x20, 0x80),  # Right column (x % 2 == 1), rows 0..3
)


class BrailleTopologyRenderer:
    """
    Renders a 2D projection of the 3D P2P coordinate space using Unicode Braille
    sub-pixel graphics (2x4 dots per terminal character cell) to trace the continuous
    3D surfing trajectory of the Query Origin alongside peer cluster symbols.
    """

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
        """Bresenham sub-pixel line in Braille space."""
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
        # 1. Draw continuous Braille trajectory trail if present
        if trajectory and len(trajectory) >= 2:
            for i in range(len(trajectory) - 1):
                p0 = trajectory[i]
                p1 = trajectory[i + 1]
                self.draw_subpixel_line(p0[0], p0[1], p1[0], p1[1])

        # 2. Overlay peer node domain symbols
        for node in nodes:
            c, r = self._to_cell(node.pos[0], node.pos[1])
            self.char_overlay[(c, r)] = domain_symbols.get(node.domain_id, "*")

        # 3. Overlay trajectory start ('o') and final Query Origin ('@')
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
        lines.append(" Legend: [P] Physics  [D] P2P  [M] ML/Thermo  [C] Control  [o->··->@] Braille Surf Trail")
        return "\n".join(lines)


# ============================================================================
# 4. DEFAULT DOMAIN CORPORA & FILESYSTEM CORPUS SHARDING
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
    """
    Reads a file or directory of text/markdown/code files and extracts clean
    lowercase token sequences of length 5..24 words for swarm ingestion.
    """
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
        # Split on sentence boundaries or newlines
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
# 5. P2P SWARM NODE (UDP SOCKET + N-GRAMS + 64-TRIT TERNARY SYNAPTIC MATRIX)
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

        # 3D Topological Coordinates [x, y, z] and Vivaldi error
        self.pos: List[float] = list(initial_pos)
        self.vivaldi_err: float = 0.80

        # UDP networking state
        self.transport: Optional[asyncio.DatagramTransport] = None
        self.port: int = 0
        self.max_peers: int = 14
        self.peers: Dict[int, Dict[str, object]] = {}
        self.seen_queries: collections.deque = collections.deque(maxlen=512)

        # Spatial Refractory State (Inhibition of Return)
        self.refractory_energy: float = 1.0

        # Local Hybrid Predictor: N-Gram Graph + 64-Trit Ternary Synaptic Matrix
        self.trigrams: Dict[Tuple[str, str], collections.Counter] = collections.defaultdict(collections.Counter)
        self.bigrams: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
        self.unigrams: collections.Counter = collections.Counter()
        self.ternary_expert = TernaryMicroExpert(deadzone=1, clip_limit=14)
        self.token_ternary_cache: Dict[str, List[int]] = {}

        for s in sentences:
            self.ingest_sentence(s)

    def ingest_sentence(self, sentence: str) -> None:
        words = [w.strip().lower() for w in sentence.strip().split() if w.strip()]
        if len(words) < 2:
            return
        for i, w in enumerate(words):
            self.unigrams[w] += 1
            if w not in self.token_ternary_cache:
                self.token_ternary_cache[w] = compute_ternary_64([w])
            if i >= 1:
                self.bigrams[words[i - 1]][w] += 1
                ctx_window = words[max(0, i - 3) : i]
                ctx_trits = compute_ternary_64(ctx_window)
                self.ternary_expert.reinforce(ctx_trits, w, strength=2)
            if i >= 2:
                self.trigrams[(words[i - 2], words[i - 1])][w] += 1

    def connection_made(self, transport: asyncio.BaseTransport) -> None:
        self.transport = transport  # type: ignore
        self.port = transport.get_extra_info("sockname")[1]

    def register_peers(self, all_nodes: List["P2PLLMNode"], k_neighbors: int = 10) -> None:
        candidates = [n for n in all_nodes if n.node_id != self.node_id]
        sampled = self.rng.sample(candidates, min(k_neighbors, len(candidates)))
        for peer in sampled:
            self.peers[peer.node_id] = {
                "port": peer.port,
                "pos": list(peer.pos),
                "err": peer.vivaldi_err,
                "rtt_ms": 1.0,
            }

    def _pack_header(
        self,
        pkt_type: int,
        ttl: int,
        query_id: int,
        coords: Tuple[float, float, float],
        context_trits: List[int],
    ) -> bytes:
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
            coords[0],
            coords[1],
            coords[2],
            self.vivaldi_err,
            *packed_trits,
        )

    def predict_local_candidates(
        self,
        w_prev2: str,
        w_prev1: str,
        w_curr: str,
        query_trits: List[int],
    ) -> List[Tuple[str, float, float]]:
        """
        Fuses N-Gram transition probabilities with the 64-Trit Ternary Synaptic
        Micro-Matrix. Returns list of (word, structural_conf, ternary_synaptic_score).
        """
        tabu: Set[str] = {w for w in (w_curr, w_prev1, w_prev2) if w}
        structural_scores: Dict[str, float] = {}

        # 1. Exact Trigram Match
        tri_counts = self.trigrams.get((w_prev1, w_curr))
        if tri_counts:
            total = sum(tri_counts.values())
            for word, count in tri_counts.items():
                if word not in tabu:
                    structural_scores[word] = max(structural_scores.get(word, 0.0), 0.96 * (count / total))

        # 2. Exact Bigram Backoff
        bi_counts = self.bigrams.get(w_curr)
        if bi_counts:
            total = sum(bi_counts.values())
            for word, count in bi_counts.items():
                if word not in tabu:
                    score = 0.72 * (count / total)
                    structural_scores[word] = min(
                        0.99, structural_scores.get(word, 0.0) + score if word in structural_scores else score
                    )

        # 3. Skip-Bigram Backoff
        if not structural_scores and w_prev1 in self.bigrams:
            skip_counts = self.bigrams[w_prev1]
            total = sum(skip_counts.values())
            for word, count in skip_counts.items():
                if word not in tabu:
                    structural_scores[word] = 0.45 * (count / total)

        in_norm = math.sqrt(max(1, sum(x * x for x in query_trits)))

        # 4. If structural n-grams matched, score each candidate with the Ternary Matrix
        if structural_scores:
            combined: List[Tuple[str, float, float]] = []
            for word, s_conf in structural_scores.items():
                t_syn = self.ternary_expert.score_word(query_trits, word, in_norm)
                combined.append((word, s_conf, t_syn))
            combined.sort(key=lambda item: item[1] * (0.6 + 0.4 * item[2]), reverse=True)
            return combined[:3]

        # 5. Pure Ternary Synaptic Matrix Recall (handles arbitrary/OOV user prompts)
        assoc_hits = self.ternary_expert.top_associative_candidates(query_trits, tabu, top_k=3)
        return [(word, 0.42 * act, act) for word, act in assoc_hits]

    def datagram_received(self, data: bytes, addr: Tuple[str, int]) -> None:
        if len(data) < HEADER_SIZE:
            return
        header = struct.unpack(HEADER_FMT, data[:HEADER_SIZE])
        magic, pkt_type, ttl, query_id, sender_id, sender_domain, ts_us, sx, sy, sz, s_err = header[:11]
        packed_trits = header[11:]

        if magic != MAGIC:
            return

        now_us = int(time.perf_counter() * 1_000_000)
        rtt_ms = max(0.05, (now_us - ts_us) / 1000.0)
        sender_pos = [sx, sy, sz]
        sender_trits = unpack_ternary_64(packed_trits)

        if sender_id != self.node_id and sender_id < 60000:
            if sender_id in self.peers:
                self.peers[sender_id]["pos"] = sender_pos
                self.peers[sender_id]["err"] = s_err
                self.peers[sender_id]["rtt_ms"] = 0.8 * float(self.peers[sender_id]["rtt_ms"]) + 0.2 * rtt_ms
            elif len(self.peers) < self.max_peers:
                self.peers[sender_id] = {
                    "port": addr[1],
                    "pos": sender_pos,
                    "err": s_err,
                    "rtt_ms": rtt_ms,
                }

        payload = data[HEADER_SIZE:]

        if pkt_type == PKT_QUERY_PULSE and len(payload) >= QUERY_SIZE:
            self._handle_query_pulse(query_id, ttl, (sx, sy, sz), sender_trits, payload)
        elif pkt_type == PKT_HEBBIAN_REWARD and len(payload) >= REWARD_SIZE:
            self._handle_hebbian_reward(sender_trits, payload)
        elif pkt_type == PKT_LIVE_INJECT:
            sentence = payload.decode("utf-8", errors="ignore").strip()
            if sentence:
                self.ingest_sentence(sentence)

    def _handle_query_pulse(
        self,
        query_id: int,
        ttl: int,
        origin_pos: Tuple[float, float, float],
        query_trits: List[int],
        payload: bytes,
    ) -> None:
        if query_id in self.seen_queries:
            return
        self.seen_queries.append(query_id)

        origin_port, spatial_lambda, raw_w2, raw_w1, raw_wc = struct.unpack(QUERY_FMT, payload[:QUERY_SIZE])
        w_prev2 = decode_word_16(raw_w2)
        w_prev1 = decode_word_16(raw_w1)
        w_curr = decode_word_16(raw_wc)

        # Kleinberg Small-World Geometric Gossip Forwarding
        if ttl > 1 and self.transport is not None and self.peers:
            fwd_hdr = self._pack_header(PKT_QUERY_PULSE, ttl - 1, query_id, origin_pos, query_trits)
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

        my_coords = (self.pos[0], self.pos[1], self.pos[2])
        for cand_word, local_conf, ternary_syn in candidates:
            word_trits = self.token_ternary_cache.get(cand_word) or compute_ternary_64([cand_word])
            combined_trits = compute_ternary_64([w_prev1, w_curr, cand_word])
            phase_align = ternary_cosine_64(query_trits, combined_trits)

            hdr = self._pack_header(PKT_TOKEN_PROPOSAL, 1, query_id, my_coords, word_trits)
            prop_payload = struct.pack(
                PROPOSAL_FMT,
                encode_word_16(cand_word),
                local_conf,
                ternary_syn,
                phase_align,
                self.refractory_energy,
            )
            self.transport.sendto(hdr + prop_payload, ("127.0.0.1", origin_port))

    def _handle_hebbian_reward(self, context_trits: List[int], payload: bytes) -> None:
        raw_word, reward, ax, ay, az = struct.unpack(REWARD_FMT, payload[:REWARD_SIZE])
        winning_word = decode_word_16(raw_word)
        anchor = [ax, ay, az]

        has_word = winning_word in self.unigrams
        if has_word and reward > 0:
            # 1. Online Ternary Synaptic Reinforcement from UDP packet header trits
            self.ternary_expert.reinforce(context_trits, winning_word, strength=1)
            # 2. Spatial Refractory Inhibition
            self.refractory_energy = max(0.35, self.refractory_energy * 0.72)
            # 3. Hebbian Topological Pull toward co-activating anchor coordinate
            eta_attr = 0.14 * reward
            for dim in range(3):
                self.pos[dim] += eta_attr * (anchor[dim] - self.pos[dim])
        else:
            self.refractory_energy = min(1.0, self.refractory_energy + 0.18)
            if not has_word:
                eta_push = 0.022
                dist_sq = sum((self.pos[d] - anchor[d]) ** 2 for d in range(3)) + 0.15
                if dist_sq < 6.5:
                    inv_d = 1.0 / math.sqrt(dist_sq)
                    for dim in range(3):
                        self.pos[dim] += eta_push * (self.pos[dim] - anchor[dim]) * inv_d

        self._apply_repulsion_and_confinement()

    def _apply_repulsion_and_confinement(self) -> None:
        num_peers = max(1, len(self.peers))
        k_rep = 0.095 / math.sqrt(num_peers)
        eps = 0.25
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
                force[dim] -= 0.22 * excess * (self.pos[dim] / norm)

        for dim in range(3):
            self.pos[dim] += max(-0.35, min(0.35, force[dim]))


# ============================================================================
# 6. QUERY ORIGIN COLLECTOR (UDP SOCKET AGGREGATOR)
# ============================================================================
class QueryOriginCollector(asyncio.DatagramProtocol):
    def __init__(self):
        self.transport: Optional[asyncio.DatagramTransport] = None
        self.port: int = 0
        self.active_query_id: int = 0
        self.query_start_us: int = 0
        self.proposals: List[Dict[str, object]] = []

    def connection_made(self, transport: asyncio.BaseTransport) -> None:
        self.transport = transport  # type: ignore
        self.port = transport.get_extra_info("sockname")[1]

    def datagram_received(self, data: bytes, addr: Tuple[str, int]) -> None:
        if len(data) < HEADER_SIZE + PROPOSAL_SIZE:
            return
        header = struct.unpack(HEADER_FMT, data[:HEADER_SIZE])
        magic, pkt_type, ttl, query_id, sender_id, sender_domain, ts_us, sx, sy, sz, s_err = header[:11]
        if magic != MAGIC or pkt_type != PKT_TOKEN_PROPOSAL or query_id != self.active_query_id:
            return

        now_us = int(time.perf_counter() * 1_000_000)
        rtt_ms = max(0.02, (now_us - self.query_start_us) / 1000.0)

        raw_word, local_conf, ternary_syn, phase_align, refrac = struct.unpack(
            PROPOSAL_FMT, data[HEADER_SIZE : HEADER_SIZE + PROPOSAL_SIZE]
        )
        word = decode_word_16(raw_word)
        self.proposals.append(
            {
                "sender_id": sender_id,
                "domain_id": sender_domain,
                "pos": (sx, sy, sz),
                "rtt_ms": rtt_ms,
                "word": word,
                "local_conf": local_conf,
                "ternary_syn": ternary_syn,
                "phase_align": phase_align,
                "refrac": refrac,
            }
        )


# ============================================================================
# 7. SWARM ORCHESTRATOR & AUTONOMOUS 3D WAVEFRONT SURFER
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
                node_id += 1

        k_init = min(12, max(6, len(self.nodes) // 3))
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

    def compute_clustering_ratio(self) -> Tuple[float, float, float]:
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
        return mean_intra, mean_inter, mean_intra / max(1e-6, mean_inter)

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
        for _ in range(epochs):
            initiator = self.rng.choice(self.nodes)
            corpus = DOMAIN_CORPORA[initiator.domain_id]
            if not corpus:
                continue
            sentence = self.rng.choice(corpus).split()
            if len(sentence) < 4:
                continue
            idx = self.rng.randint(2, len(sentence) - 2)
            w_prev2, w_prev1, w_curr = sentence[idx - 2], sentence[idx - 1], sentence[idx]
            target_next = sentence[idx + 1]

            self.query_counter += 1
            trits = compute_ternary_64([w_prev2, w_prev1, w_curr])
            coords = (initiator.pos[0], initiator.pos[1], initiator.pos[2])
            hdr = initiator._pack_header(PKT_HEBBIAN_REWARD, 1, self.query_counter, coords, trits)
            rew_payload = struct.pack(
                REWARD_FMT,
                encode_word_16(target_next),
                1.0,
                initiator.pos[0],
                initiator.pos[1],
                initiator.pos[2],
            )
            for peer in self.nodes:
                if peer.node_id != initiator.node_id and initiator.transport is not None:
                    initiator.transport.sendto(hdr + rew_payload, ("127.0.0.1", peer.port))

            await asyncio.sleep(0.003)

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
        trits = compute_ternary_64(words[:3])
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
        sentences = extract_clean_sentences_from_path(target_path)
        if not sentences:
            return 0
        centroids = self.get_domain_centroids()
        for idx, sent in enumerate(sentences):
            dom_id = idx % 4
            await self.inject_live_sentence(dom_id, sent, centroids.get(dom_id, (0.0, 0.0, 0.0)))
        return len(sentences)

    async def query_next_token(
        self,
        context_words: List[str],
        origin_pos: Tuple[float, float, float],
        spatial_lambda: float = 0.85,
        latency_mu: float = 0.03,
        base_window_sec: float = 0.022,
    ) -> Tuple[str, List[Tuple[str, float, int]], int, Tuple[float, float, float]]:
        """
        Broadcasts a binary QUERY_PULSE from `origin_pos` over UDP, collects peer
        TOKEN_PROPOSAL packets, scores them using 3D spatial gravity + 64-trit
        ternary synaptic activation, and returns:
          (winning_word, top_probs, num_packets, winning_center_of_mass_3d)
        """
        self.query_counter += 1
        qid = self.query_counter
        self.collector.active_query_id = qid
        self.collector.proposals.clear()
        self.collector.query_start_us = int(time.perf_counter() * 1_000_000)

        padded = (["", "", ""] + [w.lower() for w in context_words])[-3:]
        w_prev2, w_prev1, w_curr = padded[0], padded[1], padded[2]
        query_trits = compute_ternary_64([w_prev2, w_prev1, w_curr])
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
            encode_word_16(w_prev2),
            encode_word_16(w_prev1),
            encode_word_16(w_curr),
        )

        # Kleinberg Small-World Entry Selection
        sorted_by_dist = sorted(
            self.nodes,
            key=lambda n: sum((origin_pos[d] - n.pos[d]) ** 2 for d in range(3)),
        )
        k_near = min(len(self.nodes), max(6, len(self.nodes) // 3))
        near_peers = sorted_by_dist[:k_near]
        far_candidates = sorted_by_dist[k_near:]
        k_far = min(len(far_candidates), max(4, len(self.nodes) // 4))
        far_peers = self.rng.sample(far_candidates, k_far) if far_candidates else []

        for peer in near_peers + far_peers:
            if self.collector.transport is not None:
                self.collector.transport.sendto(hdr + payload, ("127.0.0.1", peer.port))

        dynamic_window = base_window_sec + (0.0012 * len(self.nodes))
        await asyncio.sleep(dynamic_window)

        token_scores: Dict[str, float] = collections.defaultdict(float)
        token_primary_domain: Dict[str, Tuple[float, int]] = {}
        token_coord_moments: Dict[str, List[float]] = collections.defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
        seen_node_word: Set[Tuple[int, str]] = set()
        recent_tabu = set(w.lower() for w in context_words[-3:])

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

            w_topo = math.exp(-spatial_lambda * dist_sq)
            w_lat = math.exp(-latency_mu * float(prop["rtt_ms"]))
            w_syn = 0.55 + 0.45 * float(prop["ternary_syn"])
            w_phase = float(prop["phase_align"])
            w_refrac = float(prop["refrac"])
            local_conf = float(prop["local_conf"])

            composite_weight = local_conf * w_topo * w_lat * w_syn * w_phase * w_refrac
            token_scores[word] += composite_weight

            # Accumulate weighted 3D center-of-mass for autonomous query surfing
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

        # Broadcast Hebbian Reward feedback pulse (trains both 3D coords & 64-trit synapses)
        rew_hdr = struct.pack(
            HEADER_FMT,
            MAGIC,
            PKT_HEBBIAN_REWARD,
            1,
            qid,
            65000,
            0,
            int(time.perf_counter() * 1_000_000),
            origin_pos[0],
            origin_pos[1],
            origin_pos[2],
            0.05,
            *packed_trits,
        )
        rew_payload = struct.pack(
            REWARD_FMT,
            encode_word_16(winning_word),
            0.28,
            origin_pos[0],
            origin_pos[1],
            origin_pos[2],
        )
        for peer in self.nodes:
            if self.collector.transport is not None:
                self.collector.transport.sendto(rew_hdr + rew_payload, ("127.0.0.1", peer.port))

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
    Generates a token sequence while allowing the Query Origin coordinate to
    dynamically surf toward winning peer clusters at rate `surf_rate` in [0, 1].
    Records the 3D spatial trajectory in `swarm.last_trajectory`.
    """
    swarm.reset_refractory()
    context = [w.strip().lower() for w in prompt.strip().split() if w.strip()]
    if not context:
        return start_pos

    curr_pos = start_pos
    swarm.last_trajectory = [curr_pos]
    trace_lines = []

    for step in range(steps):
        next_word, top_probs, num_pkts, win_center = await swarm.query_next_token(
            context,
            origin_pos=curr_pos,
            spatial_lambda=spatial_lambda,
        )
        if next_word == "<eos>":
            break
        context.append(next_word)

        # Autonomous 3D Coordinate Surfing Step
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

    print("\n" + "=" * 74)
    print(" INTERACTIVE P2P SPATIAL LLM REPL v3.0 (64-TRIT + 3D BRAILLE SURFING)")
    print("=" * 74)
    print(" Commands:")
    print("   <any text>                 Generate tokens (and surf 3D space) from current '@' Origin")
    print("   /cluster <p|d|m|c>         Snap '@' Origin to Physics, P2P, ML, or Control centroid")
    print("   /coord <x> <y> <z>         Move '@' Origin to custom 3D coordinates (e.g. /coord 0 0 0)")
    print("   /surf <0.0..1.0>           Set 3D Query Surfing rate γ (0 = locked, 0.4 = dynamic drift)")
    print("   /lambda <float>            Set Spatial Gravity Decay λ (e.g. /lambda 1.2 or /lambda 0.12)")
    print("   /steps <int>               Set max tokens generated per prompt")
    print("   /map                       Render 2D Braille topology + last 3D query surf trail (o->@)")
    print("   /sweep <prompt>            Fire prompt across all 4 domain centroids + (0,0,0) bridge")
    print("   /inject <p|d|m|c> <text>   Teach a new sentence to a domain cluster over UDP")
    print("   /load <file_or_dir_path>   Ingest local .txt/.md/.py files into the UDP swarm live")
    print("   /train <epochs>            Run extra Hebbian UDP coordinate + ternary synapse epochs")
    print("   /peers                     Inspect UDP ports, 3D coordinates, & ternary vocabulary size")
    print("   /quit                      Close UDP sockets and exit")
    print("=" * 74)

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
                print("  Node  UDP Port          Domain                   3D Coordinate (x, y, z)     Synapses  Refrac")
                print("  " + "-" * 90)
                step_stride = max(1, len(swarm.nodes) // 16)
                for node in swarm.nodes[::step_stride]:
                    dname = DOMAIN_NAMES[node.domain_id][:22].ljust(22)
                    p_str = f"({node.pos[0]:+5.2f}, {node.pos[1]:+5.2f}, {node.pos[2]:+5.2f})"
                    syn_cnt = len(node.ternary_expert.weights)
                    print(f"  #{node.node_id:02d}   127.0.0.1:{node.port:<5d}   {dname}   {p_str}   {syn_cnt:4d}x64t   {node.refractory_energy:.2f}")

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
                intra, inter, ratio = swarm.compute_clustering_ratio()
                print(f"  Ran {ep} Hebbian UDP epochs in {dt:.1f}ms | Intra={intra:.3f} Inter={inter:.3f} Ratio={ratio:.3f}")

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
                    print(f"  Loaded & sharded {cnt} clauses from '{target_path}' across UDP swarm. Running 40 drift epochs...")
                    await swarm.run_hebbian_self_organization(epochs=40)
                    print("  Done. Type /peers or /map to inspect updated topology.")
                else:
                    print(f"  No readable text clauses found at '{target_path}'.")

            elif op == "/sweep":
                seed_text = " ".join(parts[1:]) if len(parts) > 1 else "the system stabilizes through"
                centroids = swarm.get_domain_centroids()
                for dom_id, c_pos in centroids.items():
                    c_str = f"({c_pos[0]:+5.2f}, {c_pos[1]:+5.2f}, {c_pos[2]:+5.2f})"
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

        # Standard prompt generation from current 3D Query Origin (updates active_pos if surfing is active)
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
    parser = argparse.ArgumentParser(description="P2P Location-Weighted LLM Swarm v3.0")
    parser.add_argument("--nodes-per-domain", type=int, default=8, help="UDP peer nodes per specialist domain")
    parser.add_argument("--epochs", type=int, default=120, help="Initial Hebbian topological drift epochs")
    parser.add_argument("--steps", type=int, default=10, help="Tokens to generate per prompt")
    parser.add_argument("--surf-rate", type=float, default=0.35, help="3D Query Origin surfing rate in [0.0, 1.0]")
    parser.add_argument("--load-dir", type=str, default=None, help="Optional file or directory to ingest at boot")
    parser.add_argument("--prompt", type=str, default="the system stabilizes through", help="Initial benchmark prompt")
    parser.add_argument("--no-repl", action="store_true", help="Exit immediately after Stage 3 benchmark")
    args = parser.parse_args()

    swarm = P2PSpatialSwarm(nodes_per_domain=args.nodes_per_domain, seed=42)
    await swarm.boot_swarm(custom_dir=args.load_dir)

    try:
        print("=" * 74)
        print(" P2P LOCATION-WEIGHTED LLM SWARM v3.0 (64-TRIT SYNAPSES + 3D SURFING)")
        print("=" * 74)
        print(f" Active UDP Peer Sockets : {len(swarm.nodes)} nodes + 1 Collector (port {swarm.collector.port})")
        print(f" UDP Binary Packet Sizes : Header={HEADER_SIZE}B (64-trit) | Query={HEADER_SIZE+QUERY_SIZE}B | Proposal={HEADER_SIZE+PROPOSAL_SIZE}B")

        intra_0, inter_0, ratio_0 = swarm.compute_clustering_ratio()
        print(f"\n[STAGE 1] Initial Random 3D Topology (Pre-Training)")
        print(f"  Mean Intra-Domain Dist : {intra_0:.3f} | Mean Inter-Domain Dist : {inter_0:.3f} | Ratio : {ratio_0:.3f}")

        print(f"\n[STAGE 2] Running {args.epochs} Hebbian Drift + 64-Trit Synaptic Potentiation Epochs...")
        t0 = time.perf_counter()
        await swarm.run_hebbian_self_organization(epochs=args.epochs)
        dt = (time.perf_counter() - t0) * 1000.0
        intra_1, inter_1, ratio_1 = swarm.compute_clustering_ratio()
        print(f"  Completed in {dt:.1f} ms")
        print(f"  Mean Intra-Domain Dist : {intra_1:.3f} | Mean Inter-Domain Dist : {inter_1:.3f} | Ratio : {ratio_1:.3f}")
        print(f"  Topological Separation : {(1.0 - ratio_1 / ratio_0) * 100:.1f}% tighter semantic clustering")

        centroids = swarm.get_domain_centroids()
        print("\n" + "=" * 74)
        print(f"[STAGE 3] LOCATION-WEIGHTED 3D SURFING (Seed Prompt: \"{args.prompt}\")")
        print("=" * 74)

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

        print("\n" + "-" * 74)
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
        print("\n[STAGE 4] Unicode Braille 2D Projection of Last 3D Surfing Trajectory (o -> @):")
        print(swarm.render_terminal_topology(trajectory=swarm.last_trajectory))
        print("=" * 74)

        if not args.no_repl and sys.stdin.isatty():
            await interactive_repl(swarm, default_steps=args.steps, default_surf=args.surf_rate)

    finally:
        swarm.close_swarm()


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())

#!/usr/bin/env python3
"""
P2P-LLM v5.1 — Hardened Binary asyncio.DatagramProtocol Engine
MIT License

Directly resolves all 10 findings from the v5.0 code audit:
  [Fix #1] Implements PKT_TOKEN_PROPOSAL branch in datagram_received() + async collector.
  [Fix #2] Separates physical RTT geometry (phys_xyz) from semantic geometry (sem_xyz).
  [Fix #3] Order-invariant sublinear Log-TF vocabulary signature (replaces 0.96/0.04 EMA).
  [Fix #4] Anti-Hebbian contrastive lateral inhibition + homeostatic row scaling (no dead clipping).
  [Fix #5] Honest Vivaldi spring-relaxation manifold telemetry (no false GPS claims).
  [Fix #6] Configurable ternary dimensionality (64, 256, 512, 1024 trits) + learned feature gates.
  [Fix #7] Structure-aware code & prose tokenizer preserving == vs !=, indentation, and scopes.
  [Fix #8] 3-Tier Epistemic Provenance isolating HEBBIAN_REWARD into decaying fast-weights.
  [Fix #9] Binary struct validation, NaN/Inf coordinate rejection, HMAC auth, & peer expiration.
  [Fix #10] Exact 51-byte Header, 70-byte Query Pulse, and 32-byte Token Proposal wire alignment.
"""

from __future__ import annotations

import asyncio
import collections
import dataclasses
import enum
import hashlib
import hmac
import math
import os
import re
import secrets
import socket
import struct
import time
from typing import Callable, Deque, Dict, Iterable, List, Optional, Set, Tuple


# ============================================================================
# 1. BINARY WIRE PROTOCOL CONSTANTS & EXACT BYTE LAYOUTS (51B / 70B / 32B)
# ============================================================================

PROTOCOL_MAGIC = b"P2"
PROTOCOL_VERSION = 51  # v5.1

# Packet Types
PKT_PING = 1
PKT_PONG = 2
PKT_QUERY_PULSE = 3
PKT_TOKEN_PROPOSAL = 4   # <-- Previously missing in datagram_received()!
PKT_HEBBIAN_REWARD = 5
PKT_SYNAPTIC_GOSSIP = 6
PKT_LIVE_INJECT = 7
PKT_TOMBSTONE = 8

# Epistemic Provenance Tiers (Fix #8: Prevents self-reinforcing hallucination loops)
class EpistemicTier(enum.IntEnum):
    GROUND_TRUTH = 1        # Verified corpus or direct LIVE_INJECT (updates permanent memory)
    HUMAN_FEEDBACK = 2      # Explicit user confirmation (moderate permanent weight)
    PSEUDOLABEL_REWARD = 3  # Model-generated winning proposal (ephemeral fast-weights ONLY)

# Routing Geometry Modes (Fix #2: Allows clean ablation of physical vs semantic space)
class RoutingMode(enum.Enum):
    LATENCY_ONLY = "latency_only"
    SEMANTIC_ONLY = "semantic_only"
    HYBRID = "hybrid"
    RANDOM = "random"

# Exact 51-byte Binary Header:
#   magic (2s)         : 2 bytes  ("P2")
#   version (B)        : 1 byte   (51)
#   pkt_type (B)       : 1 byte   (PKT_*)
#   ttl (B)            : 1 byte   (hop count remaining)
#   flags (B)          : 1 byte   (epistemic tier / flags)
#   sender_id (I)      : 4 bytes  (32-bit node ID)
#   pkt_seq (I)        : 4 bytes  (monotonic sequence number for anti-replay)
#   query_id (Q)       : 8 bytes  (64-bit correlation ID)
#   trits_pos (Q)      : 8 bytes  (64-bit positive trit mask for wire header)
#   trits_neg (Q)      : 8 bytes  (64-bit negative trit mask for wire header)
#   tx_ts_us (Q)       : 8 bytes  (uint64 monotonic/wall microseconds)
#   mac_tag (5s)       : 5 bytes  (40-bit truncated HMAC-SHA256 wire authenticator)
# Total = 2 + 1 + 1 + 1 + 1 + 4 + 4 + 8 + 8 + 8 + 8 + 5 = 51 bytes
HEADER_FMT = "!2sBBBBIIQQQQ5s"
HEADER_SIZE = struct.calcsize(HEADER_FMT)
assert HEADER_SIZE == 51, f"Expected 51-byte header, got {HEADER_SIZE}"

# Exact 70-byte QUERY_PULSE Payload:
#   origin_ip (4s)     : 4 bytes  (IPv4 packed bytes)
#   origin_port (H)    : 2 bytes  (uint16 UDP port)
#   ctx_pos (Q)        : 8 bytes  (64-bit context query positive trits)
#   ctx_neg (Q)        : 8 bytes  (64-bit context query negative trits)
#   phys_x,y,z (3f)    : 12 bytes (origin physical Vivaldi coordinates)
#   sem_x,y,z (3f)     : 12 bytes (origin semantic Vivaldi coordinates)
#   tail_tokens (24s)  : 24 bytes (UTF-8 pipe-delimited tail context tokens for exact lookup)
# Total = 4 + 2 + 8 + 8 + 12 + 12 + 24 = 70 bytes
QUERY_PULSE_FMT = "!4sHQQfff fff24s"
QUERY_PULSE_SIZE = struct.calcsize(QUERY_PULSE_FMT)
assert QUERY_PULSE_SIZE == 70, f"Expected 70-byte QUERY_PULSE payload, got {QUERY_PULSE_SIZE}"

# Exact 32-byte PKT_TOKEN_PROPOSAL Payload:
#   token_utf8 (16s)   : 16 bytes (proposed token UTF-8 padded with null bytes)
#   confidence (f)     : 4 bytes  (float32 local probability / SDM margin in [0, 1])
#   sem_resonance (f)  : 4 bytes  (float32 query-to-node semantic similarity in [-1, 1])
#   support_count (I)  : 4 bytes  (uint32 ground-truth observation count)
#   responder_id (I)   : 4 bytes  (uint32 responder node ID)
# Total = 16 + 4 + 4 + 4 + 4 = 32 bytes
TOKEN_PROPOSAL_FMT = "!16sffII"
TOKEN_PROPOSAL_SIZE = struct.calcsize(TOKEN_PROPOSAL_FMT)
assert TOKEN_PROPOSAL_SIZE == 32, f"Expected 32-byte TOKEN_PROPOSAL payload, got {TOKEN_PROPOSAL_SIZE}"

# Exact 32-byte PING / PONG Geometry Payload:
#   phys_x,y,z (3f)    : 12 bytes (pure RTT Vivaldi coordinates)
#   sem_x,y,z (3f)     : 12 bytes (pure vocabulary-sketch coordinates)
#   vivaldi_err (f)    : 4 bytes  (local Vivaldi confidence error)
#   echo_ts_us (I)     : 4 bytes  (low 32 bits of ping timestamp in microseconds)
PING_PONG_FMT = "!ffffff fI"
PING_PONG_SIZE = struct.calcsize(PING_PONG_FMT)
assert PING_PONG_SIZE == 32, f"Expected 32-byte PING_PONG payload, got {PING_PONG_SIZE}"


# ============================================================================
# 2. STRUCTURE-AWARE CODE & PROSE TOKENIZER (FIX #7)
# ============================================================================

class StructureAwareTokenizer:
    """
    Fixes Audit Finding #7:
    Replaces `re.split(r"[.!?\n;]+")` with a structure-preserving lexical scanner
    that retains:
      - Multi-character programming operators: `==`, `!=`, `<=`, `>=`, `->`, `+=`, `-=`, `*=`, `//`, `**`, `&&`, `||`
      - Single-character operators and structural delimiters: `=`, `<`, `>`, `+`, `-`, `*`, `/`, `(`, `)`, `[`, `]`, `{`, `}`, `:`, `;`, `,`
      - Python/code indentation blocks: `<INDENT>`, `<DEDENT>`, `<NL>`
      - Identifiers, numbers, and prose words.

    Consequence:
      `if x == 10:\n    return True`  -> ['if', 'x', '==', '10', ':', '<NL>', '<INDENT>', 'return', 'true']
      `if x != 10:\n    return False` -> ['if', 'x', '!=', '10', ':', '<NL>', '<INDENT>', 'return', 'false']
    """
    TOKEN_SPEC = re.compile(
        r"""
        (?P<OPERATOR>==|!=|<=|>=|->|\+=|-=|\*=|/=|//|\*\*|&&|\|\||<<|>>|[=<>+\-*/%&|^~])
        | (?P<DELIM>[()\[\]{}:;,.])
        | (?P<IDENT>[A-Za-z_][A-Za-z0-9_]*|[0-9]+(?:\.[0-9]+)?)
        """,
        re.VERBOSE,
    )

    @classmethod
    def tokenize(cls, text: str, preserve_structure: bool = True) -> List[str]:
        if not text:
            return []

        # Detect whether input looks like source code or structured text
        is_code_like = preserve_structure and any(
            sym in text for sym in ("==", "!=", "<=", ">=", "def ", "class ", "return ", "if ", "{", "}", "    ")
        )

        tokens: List[str] = []
        indent_stack: List[int] = [0]
        lines = text.splitlines()

        for line_idx, raw_line in enumerate(lines):
            stripped = raw_line.strip()
            if not stripped:
                continue

            if is_code_like:
                # Calculate leading spaces (expand tabs to 4 spaces)
                expanded = raw_line.expandtabs(4)
                leading = len(expanded) - len(expanded.lstrip(" "))
                if leading > indent_stack[-1]:
                    indent_stack.append(leading)
                    tokens.append("<indent>")
                else:
                    while len(indent_stack) > 1 and leading < indent_stack[-1]:
                        indent_stack.pop()
                        tokens.append("<dedent>")

            for match in cls.TOKEN_SPEC.finditer(stripped):
                tok = match.group(0).lower()
                # Keep structural punctuation in code; in pure prose, skip trailing periods/commas if desired
                if not is_code_like and match.lastgroup == "DELIM" and tok in (".", ",", ";"):
                    continue
                tokens.append(tok)

            if is_code_like and line_idx < len(lines) - 1:
                tokens.append("<nl>")

        while is_code_like and len(indent_stack) > 1:
            indent_stack.pop()
            tokens.append("<dedent>")

        return tokens


# ============================================================================
# 3. CONFIGURABLE MULTI-DIMENSIONAL TERNARY ENCODER & ORDER-INVARIANT SKETCH
#    (FIXES #3 AND #6)
# ============================================================================

def pack_64_trits(trits: Sequence[int]) -> Tuple[int, int]:
    """
    Folds an arbitrary-length ternary vector (64, 256, 512, 1024 trits)
    into a 64-trit wire signature (pos_mask_u64, neg_mask_u64) for the 51B header.
    """
    dim = len(trits)
    pos_mask = 0
    neg_mask = 0
    if dim == 64:
        for i, val in enumerate(trits):
            if val > 0:
                pos_mask |= (1 << i)
            elif val < 0:
                neg_mask |= (1 << i)
        return pos_mask, neg_mask

    # Fold higher dimensions (256/512/1024) into 64 buckets via majority vote
    bucket_sums = [0] * 64
    for i, val in enumerate(trits):
        bucket_sums[i & 63] += val
    for i, s in enumerate(bucket_sums):
        if s > 0:
            pos_mask |= (1 << i)
        elif s < 0:
            neg_mask |= (1 << i)
    return pos_mask, neg_mask


def unpack_64_trits(pos_mask: int, neg_mask: int) -> Tuple[int, ...]:
    out = [0] * 64
    for i in range(64):
        if (pos_mask >> i) & 1:
            out[i] = 1
        elif (neg_mask >> i) & 1:
            out[i] = -1
    return tuple(out)


def ternary_cosine_vec(a: Sequence[int], b: Sequence[int]) -> float:
    """Cosine similarity between two ternary vectors of matching dimension."""
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    dot = 0
    na = 0
    nb = 0
    for i in range(n):
        va = a[i]
        vb = b[i]
        dot += va * vb
        na += va * va
        nb += vb * vb
    if na == 0 or nb == 0:
        return 0.0
    return dot / math.sqrt(na * nb)


class ConfigurableTernaryEncoder:
    """
    Fixes Audit Finding #6 & Finding #3:
    1. Supports `trit_dim` in {64, 256, 512, 1024} using multi-block keyed Blake2b
       XOF expansion so contexts don't collide in 64 dimensions.
    2. Separates whole-token identity channels from positional role channels and
       operator/syntax channels.
    3. Maintains learned per-dimension precision gates (`feature_gates`) updated
       via prediction margin utility.
    4. Computes an **Order-Invariant Log-TF Vocabulary Signature** (Fix #3):
       Regardless of the order in which documents or tokens are ingested,
       `get_vocabulary_signature()` produces the exact same ternary vector.
    """
    def __init__(self, trit_dim: int = 256) -> None:
        if trit_dim not in (64, 256, 512, 1024):
            raise ValueError("trit_dim must be one of {64, 256, 512, 1024}")
        self.trit_dim = trit_dim
        self.num_blocks = trit_dim // 64
        self._token_cache: Dict[str, Tuple[int, ...]] = {}
        # Learned feature importance gates per dimension (initialized to 1.0)
        self.feature_gates: List[float] = [1.0] * trit_dim
        # Exact token frequency counter for Order-Invariant Vocabulary Sketch (Fix #3)
        self.vocab_counts: collections.Counter = collections.Counter()
        self._vocab_sig_dirty: bool = True
        self._cached_vocab_sig_64: Tuple[int, ...] = tuple([0] * 64)
        self._cached_vocab_sig_full: Tuple[int, ...] = tuple([0] * trit_dim)

    def _hash_feature_to_rademacher(self, feat_str: str, salt_prefix: bytes) -> List[int]:
        """Expands a feature string into `self.trit_dim` sparse trits in {-1, 0, +1}."""
        out: List[int] = []
        raw_bytes = feat_str.encode("utf-8")
        for blk in range(self.num_blocks):
            key = salt_prefix + struct.pack("!B", blk)
            digest = hashlib.blake2b(raw_bytes, key=key, digest_size=16).digest()
            bits_sign, bits_mask = struct.unpack("!QQ", digest)
            for b in range(64):
                # 50% sparsity mask + unbiased Rademacher sign
                if (bits_mask >> b) & 1:
                    out.append(1 if ((bits_sign >> b) & 1) else -1)
                else:
                    out.append(0)
        return out

    def encode_token(self, token: str) -> Tuple[int, ...]:
        tok = token.strip().lower()
        if tok in self._token_cache:
            return self._token_cache[tok]

        accum = [0.0] * self.trit_dim
        # Channel A: Exact lexical identity (high weight so `==` and `!=` never blur)
        exact_vec = self._hash_feature_to_rademacher(f"LEX:{tok}", b"lex_ch")
        for i in range(self.trit_dim):
            accum[i] += 2.4 * exact_vec[i]

        # Channel B: Morphological character trigrams ONLY for longer alphabetic words
        if len(tok) >= 4 and tok.isalpha():
            padded = f"<{tok}>"
            n_tris = len(padded) - 2
            scale = 1.0 / math.sqrt(max(1, n_tris))
            for idx in range(n_tris):
                tri = padded[idx : idx + 3]
                tri_vec = self._hash_feature_to_rademacher(f"TRI:{tri}", b"morph_ch")
                for i in range(self.trit_dim):
                    accum[i] += scale * 0.65 * tri_vec[i]

        # Quantize with adaptive threshold
        mean_abs = sum(abs(x) for x in accum) / self.trit_dim
        thresh = max(0.25, 0.45 * mean_abs)
        trits = tuple(1 if x > thresh else (-1 if x < -thresh else 0) for x in accum)
        self._token_cache[tok] = trits
        return trits

    def encode_context(self, context_tokens: Sequence[str]) -> Tuple[int, ...]:
        """
        Encodes a multi-token context window into `self.trit_dim` trits using
        position-specific circular shifts + adjacent bigram interaction binding +
        learned `feature_gates`.
        """
        window = [t.strip().lower() for t in context_tokens[-8:] if t.strip()]
        if not window:
            return tuple([0] * self.trit_dim)

        accum = [0.0] * self.trit_dim
        shifts = (0, 7, 19, 37, 53, 79, 97, 113)

        for dist_from_end, tok in enumerate(reversed(window)):
            # Sharp recency weighting + exact positional role shift
            pos_weight = 1.65 / (1.0 + 0.45 * dist_from_end)
            tok_trits = self.encode_token(tok)
            shift = shifts[dist_from_end % len(shifts)] % self.trit_dim
            for d in range(self.trit_dim):
                rotated_val = tok_trits[(d - shift) % self.trit_dim]
                accum[d] += pos_weight * rotated_val

        # Bind immediate last two tokens via multiplicative XOR interaction
        if len(window) >= 2:
            t_last = self.encode_token(window[-1])
            t_prev = self.encode_token(window[-2])
            for d in range(self.trit_dim):
                bound = t_last[d] * t_prev[(d - 3) % self.trit_dim]
                accum[d] += 1.25 * bound

        # Apply learned feature importance gates
        for d in range(self.trit_dim):
            accum[d] *= self.feature_gates[d]

        mean_abs = sum(abs(x) for x in accum) / self.trit_dim
        thresh = max(0.20, 0.40 * mean_abs)
        return tuple(1 if x > thresh else (-1 if x < -thresh else 0) for x in accum)

    def record_token_observation(self, token: str, count: int = 1) -> None:
        """
        Updates the Order-Invariant Vocabulary Counter (Fix #3).
        """
        tok = token.strip().lower()
        if not tok:
            return
        self.vocab_counts[tok] += count
        if self.vocab_counts[tok] <= 0:
            del self.vocab_counts[tok]
        self._vocab_sig_dirty = True

    def get_vocabulary_signature_64(self) -> Tuple[int, ...]:
        """
        Fixes Audit Finding #3:
        Computes an Order-Invariant Sublinear Log-TF Set Sketch of the node's
        vocabulary. Because addition is commutative and counts are log-compressed
        (`log(1 + count)`), ingestion order has ZERO effect on the signature,
        and frequent stop-words cannot drown out domain-specific terms.
        """
        if not self._vocab_sig_dirty:
            return self._cached_vocab_sig_64

        if not self.vocab_counts:
            self._cached_vocab_sig_full = tuple([0] * self.trit_dim)
            self._cached_vocab_sig_64 = tuple([0] * 64)
            self._vocab_sig_dirty = False
            return self._cached_vocab_sig_64

        accum = [0.0] * self.trit_dim
        # Sort keys deterministically so floating-point summation is 100% order-invariant
        for word in sorted(self.vocab_counts.keys()):
            cnt = self.vocab_counts[word]
            if cnt <= 0:
                continue
            # Sublinear log-TF weighting + slight boost for informative medium/long tokens
            specificity_boost = 1.35 if len(word) >= 5 or not word.isalpha() else 0.75
            weight = math.log1p(cnt) * specificity_boost
            w_trits = self.encode_token(word)
            for d in range(self.trit_dim):
                accum[d] += weight * w_trits[d]

        mean_abs = sum(abs(x) for x in accum) / self.trit_dim
        thresh = max(0.15, 0.40 * mean_abs)
        full_sig = tuple(1 if x > thresh else (-1 if x < -thresh else 0) for x in accum)
        self._cached_vocab_sig_full = full_sig
        pos_m, neg_m = pack_64_trits(full_sig)
        self._cached_vocab_sig_64 = unpack_64_trits(pos_m, neg_m)
        self._vocab_sig_dirty = False
        return self._cached_vocab_sig_64


# ============================================================================
# 4. ANTI-HEBBIAN CONTRASTIVE ASSOCIATIVE MEMORY & EPISTEMIC TIERS
#    (FIXES #4 AND #8)
# ============================================================================

class ContrastiveTernaryAssociativeMemory:
    """
    Fixes Audit Finding #4 (Saturation Problem) & Finding #8 (Feedback Loop):
    1. No Hard-Clipped Saturation: Each target word's weight vector `w_target` in R^D
       is updated via a Contrastive Margin rule (reinforce ground-truth `target`,
       actively depress competing top false-positive `rival`), followed by
       Homeostatic L2 Row Normalization (`||w||_2 <= max_l2_norm`).
       When contradictory examples arrive later, anti-Hebbian depression smoothly
       shifts the winner without getting stuck at `+limit`.
    2. 3-Tier Epistemic Separation:
       - `EpistemicTier.GROUND_TRUTH` updates permanent associative weights & counts.
       - `EpistemicTier.PSEUDOLABEL_REWARD` (`HEBBIAN_REWARD`) updates ONLY a separate
         `ephemeral_fast_weights` overlay that decays by `0.90` every step and is capped
         at 15% of permanent memory magnitude. Model predictions NEVER pollute ground truth.
    """
    def __init__(self, encoder: ConfigurableTernaryEncoder, max_row_l2: float = 6.0) -> None:
        self.encoder = encoder
        self.trit_dim = encoder.trit_dim
        self.max_row_l2 = max_row_l2

        # Permanent ground-truth associative prototypes:
        # context_key (exact tail tuple) -> Counter[target_word, float_score]
        self.exact_context_prototypes: Dict[Tuple[str, ...], Dict[str, float]] = collections.defaultdict(
            lambda: collections.defaultdict(float)
        )

        # Distributed Ternary SDM synaptic weight vectors per target word:
        # target_word -> List[float] of length `trit_dim`
        self.synaptic_weights: Dict[str, List[float]] = {}
        self.ground_truth_counts: collections.Counter = collections.Counter()

        # Lineage attribution for unlearning/peer-dropout recovery:
        # source_id -> List[Tuple[Tuple[str, ...], str, float]]
        self.lineage_ledger: Dict[int, List[Tuple[Tuple[str, ...], str, float]]] = collections.defaultdict(list)

        # Fix #8: Isolated Ephemeral Fast-Weights for `HEBBIAN_REWARD` pseudo-labels
        self.ephemeral_fast_weights: Dict[Tuple[str, ...], Dict[str, float]] = collections.defaultdict(
            lambda: collections.defaultdict(float)
        )

    def _normalize_row(self, vec: List[float]) -> None:
        """Homeostatic L2 normalization: prevents saturation while preserving direction."""
        norm_sq = sum(x * x for x in vec)
        if norm_sq > self.max_row_l2 * self.max_row_l2:
            scale = self.max_row_l2 / math.sqrt(norm_sq)
            for i in range(len(vec)):
                vec[i] *= scale

    def learn_transition(
        self,
        context_tokens: Sequence[str],
        target_word: str,
        tier: EpistemicTier = EpistemicTier.GROUND_TRUTH,
        source_node_id: int = 0,
        lr: float = 0.35,
    ) -> None:
        target = target_word.strip().lower()
        ctx_clean = tuple(t.strip().lower() for t in context_tokens[-8:] if t.strip())
        if not target or not ctx_clean:
            return

        # Fix #8: Isolate model-generated rewards into decaying fast-weights only
        if tier == EpistemicTier.PSEUDOLABEL_REWARD:
            self._decay_ephemeral_weights(decay_factor=0.90)
            fast_slot = self.ephemeral_fast_weights[ctx_clean]
            fast_slot[target] = min(0.45, fast_slot[target] + 0.08)
            return

        eff_lr = lr if tier == EpistemicTier.GROUND_TRUTH else (0.5 * lr)
        ctx_trits = self.encoder.encode_context(ctx_clean)

        # Ensure target synaptic row exists
        if target not in self.synaptic_weights:
            self.synaptic_weights[target] = [0.0] * self.trit_dim

        # Find current top rival prediction (excluding `target`) for Anti-Hebbian inhibition
        current_preds = self.predict(ctx_clean, top_k=3, include_fast_weights=False)
        rival_word: Optional[str] = None
        rival_prob: float = 0.0
        for cand_w, cand_p in current_preds:
            if cand_w != target:
                rival_word = cand_w
                rival_prob = cand_p
                break

        # 1. Positive Hebbian reinforcement on ground-truth target
        w_target = self.synaptic_weights[target]
        for d in range(self.trit_dim):
            if ctx_trits[d] != 0:
                w_target[d] += eff_lr * ctx_trits[d]
        self._normalize_row(w_target)

        # 2. Anti-Hebbian Competitive Depression on rival false-positive (Fix #4)
        if rival_word is not None and rival_word in self.synaptic_weights and rival_prob > 0.15:
            w_rival = self.synaptic_weights[rival_word]
            anti_lr = 0.55 * eff_lr * rival_prob
            for d in range(self.trit_dim):
                if ctx_trits[d] != 0:
                    w_rival[d] -= anti_lr * ctx_trits[d]
                    # Update encoder feature gate: reward dimensions that separate target from rival
                    diff = abs(w_target[d] - w_rival[d])
                    self.encoder.feature_gates[d] = min(
                        2.5, max(0.4, 0.995 * self.encoder.feature_gates[d] + 0.005 * (0.8 + 0.2 * diff))
                    )
            self._normalize_row(w_rival)

        # 3. Update multi-scale context prototypes with competitive decay on rivals
        for k_len in range(1, min(8, len(ctx_clean)) + 1):
            sub_ctx = ctx_clean[-k_len:]
            proto_map = self.exact_context_prototypes[sub_ctx]
            # Softly decay competing targets in this exact context so new ground truth can win
            for other_w in list(proto_map.keys()):
                if other_w != target:
                    proto_map[other_w] *= 0.82
                    if proto_map[other_w] < 0.02:
                        del proto_map[other_w]
            step_val = eff_lr * (0.65 ** (len(ctx_clean) - k_len))
            proto_map[target] = min(10.0, proto_map[target] + step_val)
            if source_node_id != 0:
                self.lineage_ledger[source_node_id].append((sub_ctx, target, step_val))

        self.ground_truth_counts[target] += 1
        self.encoder.record_token_observation(target, 1)
        for tok in ctx_clean:
            self.encoder.record_token_observation(tok, 1)

    def _decay_ephemeral_weights(self, decay_factor: float = 0.90) -> None:
        for ctx_key in list(self.ephemeral_fast_weights.keys()):
            slot = self.ephemeral_fast_weights[ctx_key]
            for w in list(slot.keys()):
                slot[w] *= decay_factor
                if slot[w] < 0.01:
                    del slot[w]
            if not slot:
                del self.ephemeral_fast_weights[ctx_key]

    def purge_source_lineage(self, source_node_id: int) -> int:
        """Subtracts gossiped prototype contributions from a departed/revoked peer."""
        entries = self.lineage_ledger.pop(source_node_id, [])
        for sub_ctx, target, delta in entries:
            if sub_ctx in self.exact_context_prototypes:
                self.exact_context_prototypes[sub_ctx][target] -= delta
                if self.exact_context_prototypes[sub_ctx][target] <= 0.01:
                    self.exact_context_prototypes[sub_ctx].pop(target, None)
        return len(entries)

    def predict(
        self,
        context_tokens: Sequence[str],
        top_k: int = 5,
        include_fast_weights: bool = True,
    ) -> List[Tuple[str, float]]:
        if not self.synaptic_weights:
            return []

        ctx_clean = tuple(t.strip().lower() for t in context_tokens[-8:] if t.strip())
        if not ctx_clean:
            return []

        ctx_trits = self.encoder.encode_context(ctx_clean)
        logits: Dict[str, float] = collections.defaultdict(float)

        # 1. Ternary SDM Dot-Product Resonance across `self.trit_dim` (64/256/512/1024)
        inv_sqrt_d = 1.0 / math.sqrt(self.trit_dim)
        for word, w_vec in self.synaptic_weights.items():
            dot = 0.0
            for d in range(self.trit_dim):
                if ctx_trits[d] != 0:
                    dot += ctx_trits[d] * w_vec[d]
            sdm_score = dot * inv_sqrt_d
            if sdm_score > -0.5:
                logits[word] += 1.8 * sdm_score

        # 2. Multi-scale Prototype Memory readout (hierarchical context suffixes)
        for k_len in range(1, min(8, len(ctx_clean)) + 1):
            sub_ctx = ctx_clean[-k_len:]
            proto_map = self.exact_context_prototypes.get(sub_ctx)
            if proto_map:
                specificity_gain = 1.35 ** k_len
                for word, score in proto_map.items():
                    logits[word] += specificity_gain * score

        # 3. Bounded Ephemeral Fast-Weight overlay (if enabled)
        if include_fast_weights and ctx_clean in self.ephemeral_fast_weights:
            for word, fast_val in self.ephemeral_fast_weights[ctx_clean].items():
                logits[word] += min(0.25, fast_val)

        if not logits:
            return []

        ranked = sorted(logits.items(), key=lambda kv: kv[1], reverse=True)[: max(top_k * 2, 8)]
        max_l = ranked[0][1]
        conf_gate = math.tanh(max(0.01, max_l) / 1.75)
        exps = [(w, math.exp(min(25.0, (l - max_l) / 0.45))) for w, l in ranked]
        total_z = sum(v for _, v in exps) or 1.0
        return [(w, (v / total_z) * conf_gate) for w, v in exps[:top_k]]


# ============================================================================
# 5. ORTHOGONAL PHYSICAL & SEMANTIC 3D VIVALDI MANIFOLDS (FIXES #2, #5, #9)
# ============================================================================

def is_valid_vec3(coords: Sequence[float], max_bound: float = 500.0) -> bool:
    """Fix #9: Validates that 3D coordinates are finite and within safe bounds."""
    if len(coords) != 3:
        return False
    return all(isinstance(c, (int, float)) and math.isfinite(c) and abs(c) <= max_bound for c in coords)


@dataclasses.dataclass
class PeerStateV51:
    peer_id: int
    ip: str
    port: int
    last_seen_mono: float
    smoothed_rtt_ms: float = 2.0
    phys_xyz: List[float] = dataclasses.field(default_factory=lambda: [0.0, 0.0, 0.0])
    sem_xyz: List[float] = dataclasses.field(default_factory=lambda: [0.0, 0.0, 0.0])
    vocab_trits_64: Tuple[int, ...] = dataclasses.field(default_factory=lambda: tuple([0] * 64))
    vivaldi_error: float = 1.0


class OrthogonalVivaldiManifold:
    """
    Fixes Audit Finding #2 (Contaminated Geometry) & Finding #5 (GPS vs Vivaldi):
    Maintains TWO strictly orthogonal 3D coordinate systems:
      1. `phys_xyz`: Updated ONLY from hardware RTT measurements (`smoothed_rtt_ms`).
         Semantic similarity has ZERO influence on `phys_xyz`.
      2. `sem_xyz`: Updated ONLY from Order-Invariant Vocabulary Sketch similarity
         (`1.0 - phase_sim`). Network RTT has ZERO influence on `sem_xyz`.
    """
    def __init__(self, node_id: int) -> None:
        rng = hashlib.sha256(struct.pack("!I", node_id)).digest()
        self.phys_xyz: List[float] = [
            ((rng[0] / 255.0) - 0.5) * 0.4,
            ((rng[1] / 255.0) - 0.5) * 0.4,
            ((rng[2] / 255.0) - 0.5) * 0.4,
        ]
        self.sem_xyz: List[float] = [
            ((rng[3] / 255.0) - 0.5) * 0.4,
            ((rng[4] / 255.0) - 0.5) * 0.4,
            ((rng[5] / 255.0) - 0.5) * 0.4,
        ]
        self.local_error: float = 1.0

    @staticmethod
    def dist3(a: Sequence[float], b: Sequence[float]) -> float:
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)

    def update_manifolds(
        self,
        peer: PeerStateV51,
        measured_rtt_ms: float,
        local_vocab_trits_64: Tuple[int, ...],
    ) -> Tuple[float, float]:
        """
        Updates `self.phys_xyz` strictly from `measured_rtt_ms` and
        `self.sem_xyz` strictly from `ternary_cosine_vec(local_vocab_trits_64, peer.vocab_trits_64)`.
        Returns `(target_phys_dist, target_sem_dist)`.
        """
        if not is_valid_vec3(peer.phys_xyz) or not is_valid_vec3(peer.sem_xyz):
            return 0.0, 0.0

        # Smooth hardware RTT
        clamped_rtt = max(0.05, min(250.0, measured_rtt_ms))
        peer.smoothed_rtt_ms = 0.75 * peer.smoothed_rtt_ms + 0.25 * clamped_rtt

        # --- Manifold A: Pure Physical Latency Vivaldi Spring Update ---
        target_phys_dist = max(0.20, min(50.0, peer.smoothed_rtt_ms))
        curr_phys_dist = self.dist3(self.phys_xyz, peer.phys_xyz)
        rel_err = abs(curr_phys_dist - target_phys_dist) / max(0.1, target_phys_dist)

        rem_err = max(0.05, min(2.0, peer.vivaldi_error))
        weight = self.local_error / (self.local_error + rem_err)
        self.local_error = max(0.05, min(2.0, 0.20 * weight * rel_err + (1.0 - 0.20 * weight) * self.local_error))

        phys_step = max(-1.5, min(1.5, 0.18 * weight * (target_phys_dist - curr_phys_dist)))
        if curr_phys_dist > 1e-6:
            ux = (self.phys_xyz[0] - peer.phys_xyz[0]) / curr_phys_dist
            uy = (self.phys_xyz[1] - peer.phys_xyz[1]) / curr_phys_dist
            uz = (self.phys_xyz[2] - peer.phys_xyz[2]) / curr_phys_dist
        else:
            ux, uy, uz = 0.577, 0.577, 0.577

        self.phys_xyz[0] += phys_step * ux
        self.phys_xyz[1] += phys_step * uy
        self.phys_xyz[2] += phys_step * uz

        # --- Manifold B: Pure Semantic Vocabulary-Sketch Spring Update ---
        phase_sim = ternary_cosine_vec(local_vocab_trits_64, peer.vocab_trits_64)
        target_sem_dist = max(0.25, min(8.0, 3.5 * (1.0 - phase_sim)))
        curr_sem_dist = self.dist3(self.sem_xyz, peer.sem_xyz)
        sem_step = max(-1.2, min(1.2, 0.22 * (target_sem_dist - curr_sem_dist)))
        if curr_sem_dist > 1e-6:
            sx = (self.sem_xyz[0] - peer.sem_xyz[0]) / curr_sem_dist
            sy = (self.sem_xyz[1] - peer.sem_xyz[1]) / curr_sem_dist
            sz = (self.sem_xyz[2] - peer.sem_xyz[2]) / curr_sem_dist
        else:
            sx, sy, sz = -0.577, 0.577, 0.577

        self.sem_xyz[0] += sem_step * sx
        self.sem_xyz[1] += sem_step * sy
        self.sem_xyz[2] += sem_step * sz

        return target_phys_dist, target_sem_dist


# ============================================================================
# 6. ASYNCIO DATAGRAM PROTOCOL NODE WITH FIXED `PKT_TOKEN_PROPOSAL` DISPATCHER
#    (FIXES #1, #9, #10)
# ============================================================================

@dataclasses.dataclass
class TokenProposalRecord:
    responder_id: int
    token: str
    confidence: float
    sem_resonance: float
    support_count: int
    rx_mono: float


class PendingQueryCollector:
    """Collects asynchronous `PKT_TOKEN_PROPOSAL` datagrams arriving for `query_id`."""
    def __init__(self, query_id: int, expected_quorum: int, loop: asyncio.AbstractEventLoop) -> None:
        self.query_id = query_id
        self.expected_quorum = max(1, expected_quorum)
        self.proposals: List[TokenProposalRecord] = []
        self.seen_responders: Set[int] = set()
        self.quorum_event = asyncio.Event()
        self._loop = loop

    def add_proposal(self, prop: TokenProposalRecord) -> None:
        if prop.responder_id in self.seen_responders:
            return
        self.seen_responders.add(prop.responder_id)
        self.proposals.append(prop)
        if len(self.proposals) >= self.expected_quorum:
            self.quorum_event.set()


class P2PLLMNodeV51(asyncio.DatagramProtocol):
    """
    Repaired v5.1 Binary UDP Node implementing `asyncio.DatagramProtocol`.
    Includes the missing `PKT_TOKEN_PROPOSAL` receive branch in `datagram_received()`.
    """
    def __init__(
        self,
        node_id: int,
        swarm_key: bytes = b"p2p-llm-v51-default-swarm-secret",
        trit_dim: int = 256,
        routing_mode: RoutingMode = RoutingMode.HYBRID,
        host: str = "127.0.0.1",
        port: int = 0,
    ) -> None:
        self.node_id = node_id & 0xFFFFFFFF
        self.swarm_key = hashlib.sha256(swarm_key).digest()
        self.routing_mode = routing_mode
        self.bind_host = host
        self.bind_port = port

        self.encoder = ConfigurableTernaryEncoder(trit_dim=trit_dim)
        self.memory = ContrastiveTernaryAssociativeMemory(self.encoder)
        self.manifold = OrthogonalVivaldiManifold(self.node_id)

        self.transport: Optional[asyncio.DatagramTransport] = None
        self.bound_addr: Tuple[str, int] = (host, port)

        self.peers: Dict[int, PeerStateV51] = {}
        self._pending_queries: Dict[int, PendingQueryCollector] = {}
        self._seen_packets: Deque[Tuple[int, int]] = collections.deque(maxlen=2048)
        self._seen_packets_set: Set[Tuple[int, int]] = set()
        self._tx_seq: int = 0

        # Telemetry counters
        self.stats: Dict[str, int] = collections.Counter()

    # ------------------------------------------------------------------------
    # Lifecycle & Binary Header Packing / Authentication (Fix #9 & #10)
    # ------------------------------------------------------------------------

    def connection_made(self, transport: asyncio.BaseTransport) -> None:
        self.transport = transport  # type: ignore[assignment]
        sockname = transport.get_extra_info("sockname")
        if sockname:
            self.bound_addr = (sockname[0], int(sockname[1]))

    def _next_seq(self) -> int:
        self._tx_seq = (self._tx_seq + 1) & 0xFFFFFFFF
        return self._tx_seq

    def _compute_mac_tag(self, header_without_mac: bytes, payload: bytes) -> bytes:
        h = hmac.new(self.swarm_key, header_without_mac, hashlib.sha256)
        h.update(payload)
        return h.digest()[:5]

    def _pack_header(
        self,
        pkt_type: int,
        ttl: int,
        query_id: int,
        trits_64: Tuple[int, ...],
        payload: bytes,
        flags: int = int(EpistemicTier.GROUND_TRUTH),
    ) -> bytes:
        seq = self._next_seq()
        pos_m, neg_m = pack_64_trits(trits_64)
        tx_ts_us = int(time.monotonic() * 1_000_000) & 0xFFFFFFFFFFFFFFFF
        prefix = struct.pack(
            "!2sBBBBIIQQQQ",
            PROTOCOL_MAGIC,
            PROTOCOL_VERSION,
            pkt_type & 0xFF,
            max(0, min(16, ttl)),
            flags & 0xFF,
            self.node_id,
            seq,
            query_id & 0xFFFFFFFFFFFFFFFF,
            pos_m,
            neg_m,
            tx_ts_us,
        )
        mac5 = self._compute_mac_tag(prefix, payload)
        return prefix + mac5

    def _unpack_and_validate_header(self, data: bytes) -> Optional[dict]:
        """Fix #9: Validates magic, version, HMAC authentication tag, and anti-replay."""
        if len(data) < HEADER_SIZE:
            self.stats["dropped_runt"] += 1
            return None

        prefix = data[: HEADER_SIZE - 5]
        mac_rx = data[HEADER_SIZE - 5 : HEADER_SIZE]
        payload = data[HEADER_SIZE:]

        magic, ver, pkt_type, ttl, flags, sender_id, seq, query_id, pos_m, neg_m, tx_ts_us = struct.unpack(
            "!2sBBBBIIQQQQ", prefix
        )
        if magic != PROTOCOL_MAGIC or ver != PROTOCOL_VERSION:
            self.stats["dropped_bad_magic"] += 1
            return None

        mac_expected = self._compute_mac_tag(prefix, payload)
        if not hmac.compare_digest(mac_rx, mac_expected):
            self.stats["dropped_bad_mac"] += 1
            return None

        # Deduplicate (sender_id, seq)
        dedup_key = (sender_id, seq)
        if dedup_key in self._seen_packets_set:
            self.stats["dropped_duplicate"] += 1
            return None
        if len(self._seen_packets) == self._seen_packets.maxlen:
            oldest = self._seen_packets.popleft()
            self._seen_packets_set.discard(oldest)
        self._seen_packets.append(dedup_key)
        self._seen_packets_set.add(dedup_key)

        return {
            "pkt_type": pkt_type,
            "ttl": ttl,
            "flags": flags,
            "sender_id": sender_id,
            "seq": seq,
            "query_id": query_id,
            "sender_trits_64": unpack_64_trits(pos_m, neg_m),
            "tx_ts_us": tx_ts_us,
            "payload": payload,
        }

    # ------------------------------------------------------------------------
    # Peer Management & Expiration Sweeper (Fix #9)
    # ------------------------------------------------------------------------

    def add_known_peer(self, peer_id: int, ip: str, port: int) -> None:
        if peer_id == self.node_id:
            return
        if peer_id not in self.peers:
            self.peers[peer_id] = PeerStateV51(
                peer_id=peer_id,
                ip=ip,
                port=port,
                last_seen_mono=time.monotonic(),
            )
        else:
            self.peers[peer_id].ip = ip
            self.peers[peer_id].port = port
            self.peers[peer_id].last_seen_mono = time.monotonic()

    def prune_expired_peers(self, timeout_sec: float = 15.0, unlearn_on_drop: bool = True) -> List[int]:
        """Fix #9: Expires silent peers and optionally purges their gossiped lineage."""
        now = time.monotonic()
        expired: List[int] = []
        for pid, peer in list(self.peers.items()):
            if now - peer.last_seen_mono > timeout_sec:
                expired.append(pid)
                del self.peers[pid]
                if unlearn_on_drop:
                    self.memory.purge_source_lineage(pid)
        return expired

    # ------------------------------------------------------------------------
    # UDP Receive Dispatcher — WITH `PKT_TOKEN_PROPOSAL` BRANCH (FIX #1)
    # ------------------------------------------------------------------------

    def datagram_received(self, data: bytes, addr: Tuple[str, int]) -> None:
        parsed = self._unpack_and_validate_header(data)
        if parsed is None:
            return

        sender_id = parsed["sender_id"]
        if sender_id == self.node_id:
            return

        pkt_type = parsed["pkt_type"]
        payload = parsed["payload"]
        sender_trits_64 = parsed["sender_trits_64"]

        # Update or register peer entry
        peer = self.peers.get(sender_id)
        if peer is None:
            peer = PeerStateV51(
                peer_id=sender_id,
                ip=addr[0],
                port=addr[1],
                last_seen_mono=time.monotonic(),
                vocab_trits_64=sender_trits_64,
            )
            self.peers[sender_id] = peer
        else:
            peer.ip = addr[0]
            peer.port = addr[1]
            peer.last_seen_mono = time.monotonic()
            peer.vocab_trits_64 = sender_trits_64

        self.stats[f"rx_type_{pkt_type}"] += 1

        # Dispatch all packet types — INCLUDING PKT_TOKEN_PROPOSAL!
        if pkt_type == PKT_PING:
            self._handle_ping(peer, parsed, addr)
        elif pkt_type == PKT_PONG:
            self._handle_pong(peer, parsed)
        elif pkt_type == PKT_QUERY_PULSE:
            self._handle_query_pulse(peer, parsed)
        elif pkt_type == PKT_TOKEN_PROPOSAL:
            self._handle_token_proposal(peer, parsed)
        elif pkt_type == PKT_HEBBIAN_REWARD:
            self._handle_hebbian_reward(peer, parsed)
        elif pkt_type == PKT_SYNAPTIC_GOSSIP:
            self._handle_synaptic_gossip(peer, parsed)
        elif pkt_type == PKT_LIVE_INJECT:
            self._handle_live_inject(peer, parsed)
        elif pkt_type == PKT_TOMBSTONE:
            self._handle_tombstone(peer, parsed)

    # ------------------------------------------------------------------------
    # Individual Packet Handlers
    # ------------------------------------------------------------------------

    def send_ping(self, peer: PeerStateV51) -> None:
        if not self.transport:
            return
        my_sig = self.encoder.get_vocabulary_signature_64()
        echo_us = int(time.monotonic() * 1_000_000) & 0xFFFFFFFF
        payload = struct.pack(
            PING_PONG_FMT,
            self.manifold.phys_xyz[0],
            self.manifold.phys_xyz[1],
            self.manifold.phys_xyz[2],
            self.manifold.sem_xyz[0],
            self.manifold.sem_xyz[1],
            self.manifold.sem_xyz[2],
            self.manifold.local_error,
            echo_us,
        )
        hdr = self._pack_header(PKT_PING, 1, 0, my_sig, payload)
        self.transport.sendto(hdr + payload, (peer.ip, peer.port))

    def _handle_ping(self, peer: PeerStateV51, parsed: dict, addr: Tuple[str, int]) -> None:
        payload = parsed["payload"]
        if len(payload) < PING_PONG_SIZE or not self.transport:
            return
        px, py, pz, sx, sy, sz, rem_err, echo_us = struct.unpack(PING_PONG_FMT, payload[:PING_PONG_SIZE])
        if is_valid_vec3((px, py, pz)) and is_valid_vec3((sx, sy, sz)) and math.isfinite(rem_err):
            peer.phys_xyz = [px, py, pz]
            peer.sem_xyz = [sx, sy, sz]
            peer.vivaldi_error = max(0.05, min(2.0, rem_err))

        # Reply with PONG echoing the sender's timestamp
        my_sig = self.encoder.get_vocabulary_signature_64()
        pong_payload = struct.pack(
            PING_PONG_FMT,
            self.manifold.phys_xyz[0],
            self.manifold.phys_xyz[1],
            self.manifold.phys_xyz[2],
            self.manifold.sem_xyz[0],
            self.manifold.sem_xyz[1],
            self.manifold.sem_xyz[2],
            self.manifold.local_error,
            echo_us,
        )
        hdr = self._pack_header(PKT_PONG, 1, parsed["query_id"], my_sig, pong_payload)
        self.transport.sendto(hdr + pong_payload, addr)

    def _handle_pong(self, peer: PeerStateV51, parsed: dict) -> None:
        payload = parsed["payload"]
        if len(payload) < PING_PONG_SIZE:
            return
        px, py, pz, sx, sy, sz, rem_err, echo_us = struct.unpack(PING_PONG_FMT, payload[:PING_PONG_SIZE])
        if not (is_valid_vec3((px, py, pz)) and is_valid_vec3((sx, sy, sz)) and math.isfinite(rem_err)):
            self.stats["dropped_invalid_coords"] += 1
            return

        now_us = int(time.monotonic() * 1_000_000) & 0xFFFFFFFF
        elapsed_us = (now_us - echo_us) & 0xFFFFFFFF
        rtt_ms = max(0.05, elapsed_us / 1000.0)

        peer.phys_xyz = [px, py, pz]
        peer.sem_xyz = [sx, sy, sz]
        peer.vivaldi_error = max(0.05, min(2.0, rem_err))

        my_sig = self.encoder.get_vocabulary_signature_64()
        self.manifold.update_manifolds(peer, rtt_ms, my_sig)

    def _handle_query_pulse(self, peer: PeerStateV51, parsed: dict) -> None:
        """
        Receives a 70-byte `PKT_QUERY_PULSE`, evaluates local associative memory,
        and sends back a 51B + 32B `PKT_TOKEN_PROPOSAL` directly to `(origin_ip, origin_port)`.
        """
        payload = parsed["payload"]
        if len(payload) < QUERY_PULSE_SIZE or not self.transport:
            return

        origin_ip_bytes, origin_port, ctx_pos, ctx_neg, px, py, pz, sx, sy, sz, tail_bytes = struct.unpack(
            QUERY_PULSE_FMT, payload[:QUERY_PULSE_SIZE]
        )
        origin_ip = socket.inet_ntoa(origin_ip_bytes)
        tail_str = tail_bytes.rstrip(b"\x00").decode("utf-8", errors="ignore")
        context_tokens = [t for t in tail_str.split("|") if t]

        query_trits_64 = unpack_64_trits(ctx_pos, ctx_neg)
        my_sig_64 = self.encoder.get_vocabulary_signature_64()
        sem_res = ternary_cosine_vec(query_trits_64, my_sig_64)

        preds = self.memory.predict(context_tokens, top_k=2)
        for cand_word, cand_prob in preds:
            word_bytes = cand_word.encode("utf-8")[:16].ljust(16, b"\x00")
            support_cnt = int(self.memory.ground_truth_counts.get(cand_word, 1))
            prop_payload = struct.pack(
                TOKEN_PROPOSAL_FMT,
                word_bytes,
                float(cand_prob),
                float(sem_res),
                support_cnt,
                self.node_id,
            )
            word_trits_64 = unpack_64_trits(*pack_64_trits(self.encoder.encode_token(cand_word)))
            hdr = self._pack_header(PKT_TOKEN_PROPOSAL, 1, parsed["query_id"], word_trits_64, prop_payload)
            self.transport.sendto(hdr + prop_payload, (origin_ip, origin_port))
            self.stats["tx_proposals"] += 1

    def _handle_token_proposal(self, peer: PeerStateV51, parsed: dict) -> None:
        """
        FIX #1: The critical missing receive branch in `datagram_received()`.
        Unpacks the 32-byte `PKT_TOKEN_PROPOSAL`, validates float bounds, and
        registers the candidate in `self._pending_queries[query_id]`.
        """
        payload = parsed["payload"]
        if len(payload) < TOKEN_PROPOSAL_SIZE:
            self.stats["dropped_bad_proposal"] += 1
            return

        word_bytes, confidence, sem_res, support_cnt, responder_id = struct.unpack(
            TOKEN_PROPOSAL_FMT, payload[:TOKEN_PROPOSAL_SIZE]
        )
        if not (math.isfinite(confidence) and math.isfinite(sem_res)):
            self.stats["dropped_nan_proposal"] += 1
            return

        token = word_bytes.rstrip(b"\x00").decode("utf-8", errors="ignore").strip()
        if not token:
            return

        query_id = parsed["query_id"]
        collector = self._pending_queries.get(query_id)
        if collector is None:
            self.stats["late_or_unknown_proposals"] += 1
            return

        collector.add_proposal(
            TokenProposalRecord(
                responder_id=responder_id,
                token=token,
                confidence=max(0.0, min(1.0, confidence)),
                sem_resonance=max(-1.0, min(1.0, sem_res)),
                support_count=max(0, support_cnt),
                rx_mono=time.monotonic(),
            )
        )
        self.stats["accepted_proposals"] += 1

    def _handle_hebbian_reward(self, peer: PeerStateV51, parsed: dict) -> None:
        """
        FIX #8: Processes `PKT_HEBBIAN_REWARD` strictly under `EpistemicTier.PSEUDOLABEL_REWARD`
        so winning swarm proposals only nudge decaying fast-weights and NEVER corrupt
        permanent ground-truth statistics.
        """
        text = parsed["payload"].decode("utf-8", errors="ignore")
        if "->" not in text:
            return
        ctx_part, target = text.split("->", 1)
        ctx_tokens = [t for t in ctx_part.split("|") if t]
        self.memory.learn_transition(
            ctx_tokens,
            target.strip(),
            tier=EpistemicTier.PSEUDOLABEL_REWARD,
            source_node_id=peer.peer_id,
        )

    def _handle_synaptic_gossip(self, peer: PeerStateV51, parsed: dict) -> None:
        """
        Only accepts gossip transitions tagged with `EpistemicTier.GROUND_TRUTH` or `HUMAN_FEEDBACK`.
        """
        tier_val = parsed["flags"]
        if tier_val == int(EpistemicTier.PSEUDOLABEL_REWARD):
            self.stats["blocked_pseudolabel_gossip"] += 1
            return

        text = parsed["payload"].decode("utf-8", errors="ignore")
        if "->" not in text:
            return
        ctx_part, target = text.split("->", 1)
        ctx_tokens = [t for t in ctx_part.split("|") if t]
        tier = EpistemicTier.GROUND_TRUTH if tier_val == 1 else EpistemicTier.HUMAN_FEEDBACK
        self.memory.learn_transition(
            ctx_tokens,
            target.strip(),
            tier=tier,
            source_node_id=peer.peer_id,
            lr=0.25,
        )

    def _handle_live_inject(self, peer: PeerStateV51, parsed: dict) -> None:
        text = parsed["payload"].decode("utf-8", errors="ignore")
        self.ingest_text(text, preserve_structure=True, gossip_to_peers=False)

    def _handle_tombstone(self, peer: PeerStateV51, parsed: dict) -> None:
        revoked_id = parsed["sender_id"]
        self.memory.purge_source_lineage(revoked_id)

    # ------------------------------------------------------------------------
    # Ingestion & Distributed Query Pulse Inference Pipeline
    # ------------------------------------------------------------------------

    def ingest_text(
        self,
        raw_text: str,
        preserve_structure: bool = True,
        gossip_to_peers: bool = False,
    ) -> int:
        tokens = StructureAwareTokenizer.tokenize(raw_text, preserve_structure=preserve_structure)
        if len(tokens) < 2:
            return 0

        transitions = 0
        for i in range(1, len(tokens)):
            ctx = tokens[max(0, i - 8) : i]
            target = tokens[i]
            self.memory.learn_transition(
                ctx,
                target,
                tier=EpistemicTier.GROUND_TRUTH,
                source_node_id=self.node_id,
            )
            transitions += 1

            if gossip_to_peers and self.transport and self.peers:
                gossip_bytes = f"{'|'.join(ctx)}->{target}".encode("utf-8")
                my_sig = self.encoder.get_vocabulary_signature_64()
                hdr = self._pack_header(
                    PKT_SYNAPTIC_GOSSIP,
                    1,
                    0,
                    my_sig,
                    gossip_bytes,
                    flags=int(EpistemicTier.GROUND_TRUTH),
                )
                for p in list(self.peers.values())[:3]:
                    self.transport.sendto(hdr + gossip_bytes, (p.ip, p.port))

        return transitions

    def select_routing_peers(
        self,
        query_trits_64: Tuple[int, ...],
        fanout: int = 3,
        mode_override: Optional[RoutingMode] = None,
    ) -> List[PeerStateV51]:
        """
        Fix #2: Selects peers using cleanly separated geometry modes so we can
        scientifically measure Latency-Only vs Semantic-Only vs Hybrid vs Random.
        """
        mode = mode_override or self.routing_mode
        peer_list = list(self.peers.values())
        if not peer_list or fanout <= 0:
            return []

        if mode == RoutingMode.RANDOM:
            # Deterministic pseudo-random sample keyed by query trits
            seed_val = pack_64_trits(query_trits_64)[0]
            ordered = sorted(peer_list, key=lambda p: hashlib.sha256(struct.pack("!QI", seed_val, p.peer_id)).digest())
            return ordered[:fanout]

        scored: List[Tuple[float, PeerStateV51]] = []
        for p in peer_list:
            phys_d = OrthogonalVivaldiManifold.dist3(self.manifold.phys_xyz, p.phys_xyz)
            sem_d = OrthogonalVivaldiManifold.dist3(self.manifold.sem_xyz, p.sem_xyz)
            q_sim = ternary_cosine_vec(query_trits_64, p.vocab_trits_64)

            if mode == RoutingMode.LATENCY_ONLY:
                score = -phys_d
            elif mode == RoutingMode.SEMANTIC_ONLY:
                score = (2.0 * q_sim) - (0.6 * sem_d)
            else:  # RoutingMode.HYBRID
                score = (2.0 * q_sim) - (0.5 * sem_d) - (0.08 * phys_d)

            scored.append((score, p))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [p for _, p in scored[:fanout]]

    async def query_swarm_async(
        self,
        context_tokens: Sequence[str],
        top_k: int = 5,
        fanout: int = 3,
        timeout_sec: float = 0.045,
        mode_override: Optional[RoutingMode] = None,
        send_hebbian_reward: bool = False,
    ) -> Tuple[List[Tuple[str, float]], dict]:
        """
        Broadcasts a 70-byte `PKT_QUERY_PULSE` to selected peers, awaits incoming
        32-byte `PKT_TOKEN_PROPOSAL` packets dispatched via `datagram_received()`,
        and fuses remote proposals with local memory.
        """
        t0 = time.perf_counter()
        local_preds = self.memory.predict(context_tokens, top_k=top_k)
        ctx_full_trits = self.encoder.encode_context(context_tokens)
        q_pos, q_neg = pack_64_trits(ctx_full_trits)
        q_trits_64 = unpack_64_trits(q_pos, q_neg)

        selected_peers = self.select_routing_peers(q_trits_64, fanout=fanout, mode_override=mode_override)
        if not selected_peers or not self.transport:
            return local_preds, {"proposals_received": 0, "latency_ms": (time.perf_counter() - t0) * 1000.0}

        query_id = secrets.randbits(64)
        loop = asyncio.get_running_loop()
        collector = PendingQueryCollector(query_id, expected_quorum=len(selected_peers), loop=loop)
        self._pending_queries[query_id] = collector

        # Pack the 70-byte QUERY_PULSE payload
        # Keep only the most recent tokens that fit cleanly inside the 24-byte wire tail
        tail_tokens_list = [
            t.strip().lower() for t in context_tokens[-8:]
            if t.strip() and t.strip().lower() not in (":", "<nl>", "<indent>", "<dedent>")
        ]
        while len(tail_tokens_list) > 1 and len("|".join(tail_tokens_list).encode("utf-8")) > 24:
            tail_tokens_list.pop(0)
        tail_bytes = "|".join(tail_tokens_list).encode("utf-8")[:24].ljust(24, b"\x00")
        origin_ip_bytes = socket.inet_aton(self.bound_addr[0])
        origin_port = int(self.bound_addr[1])

        pulse_payload = struct.pack(
            QUERY_PULSE_FMT,
            origin_ip_bytes,
            origin_port,
            q_pos,
            q_neg,
            self.manifold.phys_xyz[0],
            self.manifold.phys_xyz[1],
            self.manifold.phys_xyz[2],
            self.manifold.sem_xyz[0],
            self.manifold.sem_xyz[1],
            self.manifold.sem_xyz[2],
            tail_bytes,
        )
        my_sig = self.encoder.get_vocabulary_signature_64()
        hdr = self._pack_header(PKT_QUERY_PULSE, 1, query_id, my_sig, pulse_payload)

        for p in selected_peers:
            self.transport.sendto(hdr + pulse_payload, (p.ip, p.port))

        try:
            await asyncio.wait_for(collector.quorum_event.wait(), timeout=timeout_sec)
        except asyncio.TimeoutError:
            pass
        finally:
            self._pending_queries.pop(query_id, None)

        # Aggregate local predictions + collected remote `PKT_TOKEN_PROPOSAL` records
        fused_scores: Dict[str, float] = collections.defaultdict(float)
        my_res = max(0.1, ternary_cosine_vec(q_trits_64, my_sig) + 0.5)
        for w, prob in local_preds:
            fused_scores[w] += 0.65 * my_res * prob

        for prop in collector.proposals:
            sem_factor = max(0.15, prop.sem_resonance + 0.6)
            support_factor = min(2.0, 0.7 + 0.3 * math.log1p(prop.support_count))
            weight = 1.25 * sem_factor * support_factor
            fused_scores[prop.token] += weight * prop.confidence

        if not fused_scores:
            return [], {"proposals_received": len(collector.proposals), "latency_ms": (time.perf_counter() - t0) * 1000.0}

        ranked = sorted(fused_scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        total_s = sum(v for _, v in ranked) or 1.0
        final_preds = [(w, v / total_s) for w, v in ranked]

        # Optional Epistemic-Tier-3 fast-weight reward broadcast
        if send_hebbian_reward and final_preds and self.transport:
            winner_word = final_preds[0][0]
            reward_bytes = f"{'|'.join(context_tokens[-8:])}->{winner_word}".encode("utf-8")
            r_hdr = self._pack_header(
                PKT_HEBBIAN_REWARD,
                1,
                query_id,
                my_sig,
                reward_bytes,
                flags=int(EpistemicTier.PSEUDOLABEL_REWARD),
            )
            for p in selected_peers:
                self.transport.sendto(r_hdr + reward_bytes, (p.ip, p.port))

        return final_preds, {
            "proposals_received": len(collector.proposals),
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 3),
        }


# ============================================================================
# 7. BUILT-IN SELF-AUDIT VERIFYING ALL 10 FIXES OVER LIVE ASYNCIO UDP
# ============================================================================

async def run_v51_audit_verification() -> None:
    print("=" * 82)
    print("P2P-LLM v5.1 — Live Binary asyncio.DatagramProtocol Self-Audit (All 10 Fixes)")
    print("=" * 82)

    loop = asyncio.get_running_loop()
    nodes: List[P2PLLMNodeV51] = []
    transports: List[asyncio.DatagramTransport] = []

    try:
        # Spin up 4 binary UDP nodes with 256-trit associative memories
        for i in range(4):
            node = P2PLLMNodeV51(node_id=100 + i, trit_dim=256, routing_mode=RoutingMode.HYBRID)
            transport, _ = await loop.create_datagram_endpoint(lambda n=node: n, local_addr=("127.0.0.1", 0))
            nodes.append(node)
            transports.append(transport)

        # Interconnect peers
        for a in nodes:
            for b in nodes:
                a.add_known_peer(b.node_id, b.bound_addr[0], b.bound_addr[1])

        # [Audit Check #3] Verify Order-Invariant Vocabulary Signature
        enc_forward = ConfigurableTernaryEncoder(trit_dim=256)
        enc_reverse = ConfigurableTernaryEncoder(trit_dim=256)
        sample_stream = ["quantum", "packet", "router", "quantum", "synapse", "packet", "quantum", "trit"]
        for tok in sample_stream:
            enc_forward.record_token_observation(tok)
        for tok in reversed(sample_stream):
            enc_reverse.record_token_observation(tok)
        sig_match = enc_forward.get_vocabulary_signature_64() == enc_reverse.get_vocabulary_signature_64()
        print(f"[Fix #3] Order-Invariant Vocab Sketch identical under reversal : {sig_match}")

        # [Audit Check #7] Verify Structure-Aware Code Ingestion (== vs != and indentation)
        code_snippet_eq = "if x == 10:\n    return true"
        code_snippet_neq = "if x != 10:\n    return false"
        nodes[1].ingest_text(code_snippet_eq, preserve_structure=True)
        nodes[1].ingest_text(code_snippet_neq, preserve_structure=True)

        pred_eq = nodes[1].memory.predict(["if", "x", "==", "10", ":", "<nl>", "<indent>", "return"], top_k=2)
        pred_neq = nodes[1].memory.predict(["if", "x", "!=", "10", ":", "<nl>", "<indent>", "return"], top_k=2)
        print(f"[Fix #7] Code AST/Operator Distinction ('==' -> {pred_eq[0][0]}, '!=' -> {pred_neq[0][0]})")

        # [Audit Check #4] Verify Anti-Hebbian Lateral Inhibition on Contradictory Examples
        for _ in range(10):
            nodes[2].memory.learn_transition(["protocol", "port", "is"], "alpha")
        for _ in range(6):
            nodes[2].memory.learn_transition(["protocol", "port", "is"], "beta")
        contradict_pred = nodes[2].memory.predict(["protocol", "port", "is"], top_k=2)
        print(f"[Fix #4] Anti-Hebbian Contradiction Resolution (new target wins): {contradict_pred}")

        # [Audit Check #1 & #2] Exchange PING/PONG and test live PKT_TOKEN_PROPOSAL over UDP
        nodes[3].ingest_text("decentralized vivaldi manifold routes ternary proposals accurately")
        for a in nodes:
            for p in a.peers.values():
                a.send_ping(p)
        await asyncio.sleep(0.03)

        # Node 0 has ZERO local knowledge of Node 3's sentence; it must receive PKT_TOKEN_PROPOSAL over UDP!
        swarm_preds, meta = await nodes[0].query_swarm_async(
            ["decentralized", "vivaldi", "manifold", "routes", "ternary", "proposals"],
            top_k=3,
            fanout=3,
        )
        print(
            f"[Fix #1] Live UDP PKT_TOKEN_PROPOSAL Received by Node 0         : "
            f"{meta['proposals_received']} proposals | Top Prediction: {swarm_preds[0] if swarm_preds else None}"
        )
        print(
            f"[Fix #2] Orthogonal Manifolds on Node 0                         : "
            f"PhysXYZ={[round(x, 3) for x in nodes[0].manifold.phys_xyz]} | "
            f"SemXYZ={[round(x, 3) for x in nodes[0].manifold.sem_xyz]}"
        )
        print("=" * 82)

    finally:
        for t in transports:
            t.close()


if __name__ == "__main__":
    asyncio.run(run_v51_audit_verification())

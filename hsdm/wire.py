"""Authenticated datagram framing.

Layout:  magic(2) version(1) flags(1) type(1) sender(8) nonce(12) ts_ms(8) | body | tag(16)

Every packet is authenticated with a key derived from the swarm key.  When the
`cryptography` package is installed the body is also encrypted
(ChaCha20-Poly1305, header as associated data); without it the body is sent in
the clear under an HMAC-SHA256 tag.  A receiver rejects packets whose
timestamp is outside a two-minute window or whose (sender, nonce) it has
already seen, so a captured packet cannot be replayed.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import struct
import time
from typing import Dict, NamedTuple, Optional, Tuple

try:
    from cryptography.exceptions import InvalidTag
    from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

    HAVE_AEAD = True
except Exception:  # pragma: no cover - optional dependency
    HAVE_AEAD = False

MAGIC = b"HS"
VERSION = 6
FLAG_SEALED = 1

PING, PONG, SKETCH_REQ, SKETCH, QUERY, HIT = 1, 2, 3, 4, 5, 6

HEADER = struct.Struct("!2sBBBQ12sQ")
TAG_LEN = 16
MAX_DATAGRAM = 1400
CLOCK_WINDOW_MS = 120_000


class Packet(NamedTuple):
    type: int
    sender: int
    nonce: bytes
    body: bytes


class Codec:
    def __init__(self, swarm_key: bytes, node_id: int, seal: Optional[bool] = None) -> None:
        self.node_id = node_id
        self._mac_key = hashlib.blake2b(b"hsdm-wire-mac" + swarm_key, digest_size=32).digest()
        self.seal = HAVE_AEAD if seal is None else (seal and HAVE_AEAD)
        self._aead = (
            ChaCha20Poly1305(hashlib.blake2b(b"hsdm-wire-aead" + swarm_key, digest_size=32).digest())
            if HAVE_AEAD
            else None
        )
        self._seen: Dict[Tuple[int, bytes], int] = {}
        self._last_sweep = 0

    def pack(self, ptype: int, body: bytes = b"") -> Tuple[bytes, bytes]:
        """Returns (datagram, nonce)."""
        nonce = os.urandom(12)  # never reused: it is also the AEAD nonce
        flags = FLAG_SEALED if self.seal else 0
        head = HEADER.pack(MAGIC, VERSION, flags, ptype, self.node_id, nonce, int(time.time() * 1000))
        if self.seal:
            data = head + self._aead.encrypt(nonce, body, head)
        else:
            data = head + body + hmac.new(self._mac_key, head + body, hashlib.sha256).digest()[:TAG_LEN]
        if len(data) > MAX_DATAGRAM:
            raise ValueError(f"datagram of {len(data)} bytes exceeds {MAX_DATAGRAM}")
        return data, nonce

    def unpack(self, data: bytes) -> Optional[Packet]:
        """Returns None for anything malformed, unauthenticated, stale or replayed."""
        if len(data) < HEADER.size + TAG_LEN or len(data) > MAX_DATAGRAM:
            return None
        head = data[: HEADER.size]
        magic, version, flags, ptype, sender, nonce, ts_ms = HEADER.unpack(head)
        if magic != MAGIC or version != VERSION or sender == self.node_id:
            return None
        if flags & FLAG_SEALED:
            if self._aead is None:
                return None
            try:
                body = self._aead.decrypt(nonce, data[HEADER.size :], head)
            except InvalidTag:
                return None
        else:
            body, tag = data[HEADER.size : -TAG_LEN], data[-TAG_LEN:]
            want = hmac.new(self._mac_key, head + body, hashlib.sha256).digest()[:TAG_LEN]
            if not hmac.compare_digest(tag, want):
                return None
        now_ms = int(time.time() * 1000)
        if abs(now_ms - ts_ms) > CLOCK_WINDOW_MS:
            return None
        key = (sender, nonce)
        if key in self._seen:
            return None
        self._seen[key] = ts_ms
        if now_ms - self._last_sweep > 10_000:
            self._last_sweep = now_ms
            cutoff = now_ms - 2 * CLOCK_WINDOW_MS
            self._seen = {k: t for k, t in self._seen.items() if t >= cutoff}
        return Packet(ptype, sender, nonce, body)

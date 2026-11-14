"""A swarm peer: membership, sketch exchange, query routing and result fusion."""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import os
import random
import socket
import struct
import time
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

from . import wire
import numpy as np

from .hdc import STOPWORDS, Encoder, Trits, cosine_dense, ternarize, tokenize
from .index import SKETCH_BUCKETS, LocalIndex, pack_levels, route_scores, unpack_levels
from .vivaldi import Vivaldi

Addr = Tuple[str, int]

STATE = struct.Struct("!5fII")  # vivaldi x y z height error | passages | sketch version
PEX_ENTRY = struct.Struct("!Q4sH")
SKETCH_HEAD = struct.Struct("!IIBB")  # version | passages | part | parts
SKETCH_PART = 1024
QUERY_HEAD = struct.Struct("!BH")
HIT_HEAD = struct.Struct("!12sBBf")

MAX_HITS_PER_QUERY = 8
PEX_PER_PONG = 8


@dataclasses.dataclass
class Peer:
    id: int
    addr: Addr
    last_seen: float
    rtt_ms: Optional[float] = None
    pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    height: float = 0.0
    error: float = 1.0
    passages: int = 0
    sketch_version: int = 0
    sketch: Optional[np.ndarray] = None
    parts: Dict[int, bytes] = dataclasses.field(default_factory=dict)
    parts_version: int = -1
    have_version: int = -1
    asked_sketch_at: float = 0.0
    tokens: float = 20.0  # query rate limit bucket
    tokens_at: float = 0.0


@dataclasses.dataclass
class Result:
    score: float
    label: str
    text: str
    peer: int  # 0 means this node's own index
    addr: Optional[Addr] = None


@dataclasses.dataclass
class SearchReport:
    results: List[Result]
    asked: List[int]
    answered: List[int]
    elapsed_ms: float
    known_peers: int


class _Pending:
    def __init__(self, expected: set) -> None:
        self.waiting = set(expected)
        self.hits: List[Tuple[int, Addr, str, str]] = []
        self.counts: Counter = Counter()
        self.done = asyncio.Event()
        if not self.waiting:
            self.done.set()


class Node(asyncio.DatagramProtocol):
    def __init__(
        self,
        swarm_key: bytes,
        dim: int = 2048,
        node_id: Optional[int] = None,
        ping_interval: float = 5.0,
        seal: Optional[bool] = None,
    ) -> None:
        if not swarm_key:
            raise ValueError("a swarm key is required")
        self.id = node_id or (int.from_bytes(os.urandom(8), "big") | 1)
        self.encoder = Encoder(swarm_key, dim)
        self.index = LocalIndex(self.encoder)
        self.codec = wire.Codec(swarm_key, self.id, seal)
        self.coord = Vivaldi(self.id)
        self.peers: Dict[int, Peer] = {}
        self.ping_interval = ping_interval
        self.query_rate = 20.0  # queries per second accepted from any one peer
        self.stats: Counter = Counter()
        self.addr: Addr = ("", 0)
        self._transport: Optional[asyncio.DatagramTransport] = None
        self._pings: Dict[bytes, Tuple[float, Addr]] = {}
        self._pinged: Dict[Addr, float] = {}
        self._queries: Dict[bytes, _Pending] = {}
        self._task: Optional[asyncio.Task] = None
        self._bootstrap: List[Addr] = []

    # -- lifecycle -----------------------------------------------------------
    async def start(self, host: str = "127.0.0.1", port: int = 0, bootstrap: Sequence[Addr] = ()) -> None:
        loop = asyncio.get_running_loop()
        await loop.create_datagram_endpoint(lambda: self, local_addr=(host, port))
        self._bootstrap = [(socket.gethostbyname(h), p) for h, p in bootstrap]
        for addr in self._bootstrap:
            self._ping(addr)
        self._task = asyncio.create_task(self._maintain())

    def stop(self) -> None:
        if self._task:
            self._task.cancel()
        if self._transport:
            self._transport.close()

    def connection_made(self, transport) -> None:
        self._transport = transport
        name = transport.get_extra_info("sockname")
        self.addr = (name[0], name[1])

    async def _maintain(self) -> None:
        started = time.monotonic()
        next_ping = started + self.ping_interval
        while True:
            now = time.monotonic()
            await asyncio.sleep(0.2 if now - started < 3 else min(1.0, self.ping_interval))
            now = time.monotonic()
            for peer in self.peers.values():  # fetch sketches we are missing
                self._want_sketch(peer, now)
            if now < next_ping:
                continue
            next_ping = now + self.ping_interval * random.uniform(0.8, 1.2)
            for pid, peer in list(self.peers.items()):
                if now - peer.last_seen > 4 * self.ping_interval:
                    del self.peers[pid]
            self._pings = {n: v for n, v in self._pings.items() if now - v[0] < 10}
            self._pinged = {a: t for a, t in self._pinged.items() if now - t < 10}
            targets = list(self.peers.values())
            if len(targets) > 24:
                targets = random.sample(targets, 24)
            for peer in targets:
                self._ping(peer.addr)
            if not self.peers:
                for addr in self._bootstrap:
                    self._ping(addr)

    async def wait_for_peers(self, count: int = 1, timeout: float = 2.0) -> bool:
        """Wait until `count` peers are known and their sketches have arrived."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            ready = [p for p in self.peers.values() if p.passages == 0 or p.have_version == p.sketch_version]
            if len(ready) >= count:
                return True
            await asyncio.sleep(0.01)
        return False

    # -- sending -------------------------------------------------------------
    def _send(self, addr: Addr, ptype: int, body: bytes = b"") -> bytes:
        data, nonce = self.codec.pack(ptype, body)
        if self._transport and not self._transport.is_closing():
            self._transport.sendto(data, addr)
            self.stats["tx"] += 1
            self.stats["tx_bytes"] += len(data)
        return nonce

    def _state(self) -> bytes:
        return STATE.pack(*self.coord.pos, self.coord.height, self.coord.error, len(self.index.vectors), self.index.version)

    def _ping(self, addr: Addr) -> None:
        now = time.monotonic()
        self._pinged[addr] = now
        self._pings[self._send(addr, wire.PING, self._state())] = (now, addr)

    def _want_sketch(self, peer: Peer, now: float) -> None:
        if peer.passages and peer.have_version != peer.sketch_version and now - peer.asked_sketch_at > 0.25:
            peer.asked_sketch_at = now
            self._send(peer.addr, wire.SKETCH_REQ)

    # -- receiving -----------------------------------------------------------
    def datagram_received(self, data: bytes, addr: Addr) -> None:
        pkt = self.codec.unpack(data)
        if pkt is None:
            self.stats["dropped"] += 1
            return
        try:
            handler = {
                wire.PING: self._on_ping, wire.PONG: self._on_pong, wire.SKETCH_REQ: self._on_sketch_req,
                wire.SKETCH: self._on_sketch, wire.QUERY: self._on_query, wire.HIT: self._on_hit,
            }.get(pkt.type)
            if handler:
                handler(pkt, addr)
        except (struct.error, ValueError, UnicodeDecodeError, IndexError):
            self.stats["malformed"] += 1

    def _update(self, peer: Peer, state: bytes) -> None:
        x, y, z, height, error, passages, version = STATE.unpack(state)
        peer.last_seen = time.monotonic()
        if Vivaldi.valid((x, y, z), height, error):
            peer.pos, peer.height, peer.error = (x, y, z), height, error
        peer.passages, peer.sketch_version = passages, version
        self._want_sketch(peer, peer.last_seen)

    def _known(self, pkt: wire.Packet, addr: Addr, cost: float = 1.0) -> Optional[Peer]:
        """The sending peer, if it is in our table at this address and within its rate limit.

        A peer only enters the table by answering a ping we sent to its
        address, so everything expensive we do is done for an address that
        has proven it receives our packets - not for a forged source.
        """
        peer = self.peers.get(pkt.sender)
        if peer is None or peer.addr != addr:
            self.stats["refused_unknown"] += 1
            return None
        now = time.monotonic()
        peer.tokens = min(self.query_rate, peer.tokens + (now - peer.tokens_at) * self.query_rate)
        peer.tokens_at = now
        if peer.tokens < cost:
            self.stats["refused_rate"] += 1
            return None
        peer.tokens -= cost
        return peer

    def _on_ping(self, pkt: wire.Packet, addr: Addr) -> None:
        state = pkt.body[: STATE.size]
        STATE.unpack(state)
        peer = self.peers.get(pkt.sender)
        if peer is not None and peer.addr == addr:
            self._update(peer, state)
        elif time.monotonic() - self._pinged.get(addr, 0.0) > 1.0:
            self._ping(addr)  # a stranger: check that it really lives at this address
        others = [p for p in self.peers.values() if p.id != pkt.sender]
        sample = random.sample(others, min(PEX_PER_PONG, len(others)))
        pex = b"".join(PEX_ENTRY.pack(p.id, socket.inet_aton(p.addr[0]), p.addr[1]) for p in sample)
        self._send(addr, wire.PONG, self._state() + pkt.nonce + bytes([len(sample)]) + pex)

    def _on_pong(self, pkt: wire.Packet, addr: Addr) -> None:
        body = pkt.body
        state = body[: STATE.size]
        STATE.unpack(state)
        sent = self._pings.pop(body[STATE.size : STATE.size + 12], None)
        if sent is None or sent[1] != addr:
            return  # unsolicited, or from somewhere we did not ping
        peer = self.peers.get(pkt.sender)
        if peer is None:
            if len(self.peers) >= 4096:
                return
            peer = self.peers[pkt.sender] = Peer(pkt.sender, addr, time.monotonic())
        peer.addr = addr
        self._update(peer, state)
        rtt = max(0.01, (time.monotonic() - sent[0]) * 1000.0)
        peer.rtt_ms = rtt if peer.rtt_ms is None else 0.8 * peer.rtt_ms + 0.2 * rtt
        self.coord.update(peer.rtt_ms, peer.pos, peer.height, peer.error)
        count = min(body[STATE.size + 12], PEX_PER_PONG)
        offset = STATE.size + 13
        for i in range(count):
            pid, ip, port = PEX_ENTRY.unpack_from(body, offset + i * PEX_ENTRY.size)
            candidate = (socket.inet_ntoa(ip), port)
            if pid != self.id and pid not in self.peers and port and candidate not in self._pinged:
                self._ping(candidate)

    def _on_sketch_req(self, pkt: wire.Packet, addr: Addr) -> None:
        if self._known(pkt, addr, cost=2.0) is None:
            return
        raw = pack_levels(self.index.sketch)
        parts = [raw[i : i + SKETCH_PART] for i in range(0, len(raw), SKETCH_PART)]
        for i, part in enumerate(parts):
            self._send(addr, wire.SKETCH, SKETCH_HEAD.pack(self.index.version, len(self.index.vectors), i, len(parts)) + part)

    def _on_sketch(self, pkt: wire.Packet, addr: Addr) -> None:
        peer = self.peers.get(pkt.sender)
        if peer is None or peer.addr != addr:
            return
        version, passages, part, parts = SKETCH_HEAD.unpack_from(pkt.body)
        if parts * SKETCH_PART != SKETCH_BUCKETS // 2 or part >= parts:
            return
        if peer.parts_version != version:
            peer.parts, peer.parts_version = {}, version
        peer.parts[part] = pkt.body[SKETCH_HEAD.size :]
        if len(peer.parts) == parts:
            peer.sketch = unpack_levels(b"".join(peer.parts[i] for i in range(parts)))
            peer.have_version, peer.sketch_version, peer.passages = version, version, passages
            peer.parts = {}

    def _on_query(self, pkt: wire.Packet, addr: Addr) -> None:
        if self._known(pkt, addr) is None:
            return
        limit, dim = QUERY_HEAD.unpack_from(pkt.body)
        if dim != self.encoder.dim:
            return
        query = Trits.from_bytes(pkt.body[QUERY_HEAD.size :], dim)
        hits = self.index.search(query, min(limit, MAX_HITS_PER_QUERY))
        self.stats["queries_served"] += 1
        if not hits:
            self._send(addr, wire.HIT, HIT_HEAD.pack(pkt.nonce, 0, 0, 0.0))
        for i, (score, passage) in enumerate(hits):
            label = passage.label.encode()[:200]
            text = passage.text.encode()
            body = HIT_HEAD.pack(pkt.nonce, i, len(hits), score) + bytes([len(label)]) + label + struct.pack("!H", len(text)) + text
            self._send(addr, wire.HIT, body)  # always to the datagram's source, never to an address in a payload

    def _on_hit(self, pkt: wire.Packet, addr: Addr) -> None:
        qid, i, total, _score = HIT_HEAD.unpack_from(pkt.body)
        pending = self._queries.get(qid)
        if pending is None or pkt.sender not in pending.waiting:
            return
        if total:
            if pending.counts[pkt.sender] >= MAX_HITS_PER_QUERY:
                return
            off = HIT_HEAD.size
            llen = pkt.body[off]
            label = pkt.body[off + 1 : off + 1 + llen].decode()
            off += 1 + llen
            (tlen,) = struct.unpack_from("!H", pkt.body, off)
            text = pkt.body[off + 2 : off + 2 + tlen].decode()
            pending.hits.append((pkt.sender, addr, label, text))
            pending.counts[pkt.sender] += 1
        if pending.counts[pkt.sender] >= total:
            pending.waiting.discard(pkt.sender)
            if not pending.waiting:
                pending.done.set()

    # -- routing and search --------------------------------------------------
    def estimated_rtt(self, peer: Peer) -> float:
        return peer.rtt_ms if peer.rtt_ms is not None else self.coord.distance(peer.pos, peer.height)

    def route(self, text: str, fanout: int, policy: str = "sketch") -> List[Peer]:
        """Pick the peers most likely to hold passages matching the query."""
        holders = [p for p in self.peers.values() if p.passages > 0]
        if fanout <= 0 or not holders:
            return []
        if policy == "all":
            return holders
        if policy == "random":
            return random.sample(holders, min(fanout, len(holders)))
        sketched = [p for p in holders if p.sketch is not None]
        blind = [p for p in holders if p.sketch is None]
        words = {w for w in tokenize(text) if w not in STOPWORDS}
        buckets = [self.encoder.bucket(w, SKETCH_BUCKETS) for w in words]
        ranked: List[Peer] = []
        if sketched:
            # this node's own passages count toward the swarm-wide word statistics
            sketches = np.stack([p.sketch for p in sketched] + [self.index.sketch])
            sizes = np.array([p.passages for p in sketched] + [len(self.index.vectors)])
            scores = route_scores(buckets, sketches, sizes)[:-1]
            scores = scores - 0.001 * np.array([min(self.estimated_rtt(p), 500.0) for p in sketched])
            ranked = [sketched[i] for i in np.argsort(-scores)]
        return (ranked + blind)[:fanout]

    async def search(
        self,
        text: str,
        limit: int = 5,
        fanout: int = 3,
        timeout: Optional[float] = None,
        policy: str = "sketch",
    ) -> SearchReport:
        started = time.perf_counter()
        accum = self.encoder.accumulate(text)
        query = Trits.from_dense(ternarize(accum))
        per_peer = min(MAX_HITS_PER_QUERY, max(limit, 3))
        found: List[Tuple[int, Optional[Addr], str, str]] = [
            (0, None, p.label, p.text) for _, p in self.index.search(query, per_peer)
        ]
        targets = self.route(text, fanout, policy) if query.nnz[0] else []
        pending = _Pending({p.id for p in targets})
        body = QUERY_HEAD.pack(per_peer, self.encoder.dim) + query.to_bytes()
        nonces = []
        for peer in targets:
            nonce = self._send(peer.addr, wire.QUERY, body)
            self._queries[nonce] = pending
            nonces.append(nonce)
        if timeout is None:
            slowest = max((self.estimated_rtt(p) for p in targets), default=0.0)
            timeout = min(1.5, max(0.05, 3.0 * slowest / 1000.0 + 0.03))
        try:
            await asyncio.wait_for(pending.done.wait(), timeout)
        except asyncio.TimeoutError:
            self.stats["query_timeouts"] += 1
        finally:
            for nonce in nonces:
                self._queries.pop(nonce, None)
        found.extend(pending.hits)

        # Peers' own scores are ignored: every returned passage is re-scored
        # here against the full-precision query, so a peer cannot talk its
        # way to the top of the list.
        best: Dict[bytes, Result] = {}
        for pid, addr, label, passage in found:
            key = hashlib.blake2b(passage.encode(), digest_size=12).digest()
            if key in best:
                continue
            score = cosine_dense(accum, self.encoder.accumulate(passage))
            best[key] = Result(score, label, passage, pid, addr)
        results = sorted(best.values(), key=lambda r: r.score, reverse=True)[:limit]
        asked = [p.id for p in targets]
        return SearchReport(
            results=results,
            asked=asked,
            answered=[pid for pid in asked if pid not in pending.waiting],
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
            known_peers=len(self.peers),
        )

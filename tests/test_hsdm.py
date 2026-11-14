import asyncio
import struct
import time

import numpy as np
import pytest

from hsdm import wire
from hsdm.hdc import Encoder, Trits
from hsdm.index import SKETCH_BUCKETS, LocalIndex, pack_levels, route_scores, unpack_levels
from hsdm.node import HIT_HEAD, Node

KEY = b"test-swarm"

DOCS = {
    "cooking": "Simmer the tomato sauce with garlic and basil for twenty minutes.\n\n"
               "Knead the bread dough until smooth, then let it rise in a warm place.\n\n"
               "Roast the chicken with lemon and thyme until the skin is crisp.",
    "astronomy": "The telescope resolved the rings of Saturn and four of its moons.\n\n"
                 "A supernova briefly outshines the entire galaxy that hosts it.\n\n"
                 "Neutron stars are the collapsed cores of massive supergiant stars.",
    "networking": "The router forwards each packet toward its destination subnet.\n\n"
                  "TCP retransmits segments when acknowledgements fail to arrive.\n\n"
                  "A firewall drops inbound connections that match no allow rule.",
}


# ---------------------------------------------------------------- encoder
def test_trits_roundtrip_and_validation():
    enc = Encoder(KEY)
    t = enc.encode("the quick brown fox jumps over the lazy dog")
    back = Trits.from_bytes(t.to_bytes(), enc.dim)
    assert np.array_equal(back.to_dense(), t.to_dense())
    with pytest.raises(ValueError):
        Trits.from_bytes(b"\xff" * (enc.dim // 4), enc.dim)  # a trit both + and -
    with pytest.raises(ValueError):
        Trits.from_bytes(b"\x00" * 10, enc.dim)


def test_similar_text_scores_higher():
    enc = Encoder(KEY)
    a = enc.encode("neutron stars are collapsed cores of massive stars")
    assert a.cosine(enc.encode("the collapsed core of a massive star becomes a neutron star"))[0] > 0.2
    assert abs(a.cosine(enc.encode("knead the bread dough until smooth"))[0]) < 0.08


def test_vectors_are_keyed():
    text = "the router forwards each packet"
    same = Encoder(KEY).encode(text).cosine(Encoder(KEY).encode(text))[0]
    other = Encoder(KEY).encode(text).cosine(Encoder(b"another").encode(text))[0]
    assert same == pytest.approx(1.0) and abs(other) < 0.1


# ---------------------------------------------------------------- wire
@pytest.mark.parametrize("seal", [False, True])
def test_wire_authentication_and_replay(seal):
    if seal and not wire.HAVE_AEAD:
        pytest.skip("cryptography not installed")
    a, b = wire.Codec(KEY, 1, seal), wire.Codec(KEY, 2, seal)
    data, nonce = a.pack(wire.QUERY, b"hello")
    pkt = b.unpack(data)
    assert pkt and pkt.body == b"hello" and pkt.sender == 1 and pkt.nonce == nonce
    assert b.unpack(data) is None  # replay
    fresh, _ = a.pack(wire.QUERY, b"hello")
    assert wire.Codec(b"wrong key", 3, seal).unpack(fresh) is None
    tampered = bytearray(a.pack(wire.QUERY, b"hello")[0])
    tampered[-20] ^= 1
    assert b.unpack(bytes(tampered)) is None
    assert a.unpack(a.pack(wire.PING)[0]) is None  # our own packets are ignored
    if seal:
        assert b"hello" not in a.pack(wire.QUERY, b"hello")[0]


def test_wire_rejects_stale_packets(monkeypatch):
    a, b = wire.Codec(KEY, 1), wire.Codec(KEY, 2)
    real = time.time
    monkeypatch.setattr(time, "time", lambda: real() - 600)
    old, _ = a.pack(wire.PING)
    monkeypatch.setattr(time, "time", real)
    assert b.unpack(old) is None


# ---------------------------------------------------------------- index
def test_sketch_packing_and_routing_score():
    enc = Encoder(KEY)
    sketches, sizes = [], []
    for text in DOCS.values():
        ix = LocalIndex(enc)
        ix.add_text("d", text)
        ix.build()
        assert np.array_equal(unpack_levels(pack_levels(ix.sketch)), ix.sketch)
        sketches.append(ix.sketch)
        sizes.append(len(ix))
    buckets = [enc.bucket(w, SKETCH_BUCKETS) for w in ("supernova", "galaxy")]
    scores = route_scores(buckets, np.stack(sketches), np.array(sizes))
    assert int(np.argmax(scores)) == list(DOCS).index("astronomy")


# ---------------------------------------------------------------- swarm over UDP
async def _swarm(extra=0):
    nodes = []
    for label, text in DOCS.items():
        node = Node(KEY, ping_interval=0.3)
        node.index.add_text(label, text)
        node.index.build()
        await node.start(bootstrap=[nodes[0].addr] if nodes else [])
        nodes.append(node)
    client = Node(KEY, ping_interval=0.3)
    await client.start(bootstrap=[nodes[0].addr])  # knows only the first peer
    await asyncio.sleep(0.3)
    assert await client.wait_for_peers(len(nodes), timeout=5)
    return nodes, client


def _stop(*nodes):
    for n in nodes:
        n.stop()


def test_discovery_routing_and_retrieval():
    async def run():
        nodes, client = await _swarm()
        assert set(client.peers) == {n.id for n in nodes}  # found the others through peer exchange
        for query, label in [("which packets does the firewall drop", "networking"),
                             ("how long to simmer tomato sauce", "cooking"),
                             ("what is a neutron star", "astronomy")]:
            rep = await client.search(query, limit=1, fanout=1)
            assert len(rep.asked) == 1 and rep.answered == rep.asked
            assert rep.results and rep.results[0].label == label
        _stop(client, *nodes)

    asyncio.run(run())


def test_outsider_with_wrong_key_learns_nothing():
    async def run():
        nodes, client = await _swarm()
        outsider = Node(b"not the key", ping_interval=0.3)
        await outsider.start(bootstrap=[nodes[0].addr])
        await asyncio.sleep(0.5)
        assert not outsider.peers and nodes[0].stats["dropped"] > 0
        _stop(outsider, client, *nodes)

    asyncio.run(run())


def test_queries_from_unverified_senders_are_refused():
    """A key holder that never completed a ping exchange gets no passages back."""

    async def run():
        nodes, client = await _swarm()
        got = []

        class Raw(asyncio.DatagramProtocol):
            def datagram_received(self, data, addr):
                got.append(data)

        loop = asyncio.get_running_loop()
        transport, _ = await loop.create_datagram_endpoint(Raw, local_addr=("127.0.0.1", 0))
        codec = wire.Codec(KEY, 999)
        query = Encoder(KEY).encode("firewall drops inbound connections")
        body = struct.pack("!BH", 5, 2048) + query.to_bytes()
        transport.sendto(codec.pack(wire.QUERY, body)[0], nodes[2].addr)
        transport.sendto(codec.pack(wire.SKETCH_REQ)[0], nodes[2].addr)
        await asyncio.sleep(0.2)
        assert not got and nodes[2].stats["refused_unknown"] == 2
        transport.close()
        _stop(client, *nodes)

    asyncio.run(run())


def test_peer_cannot_inflate_its_own_score():
    async def run():
        nodes, client = await _swarm()
        liar = nodes[0]  # holds cooking; will answer every query with spam and a perfect score

        def lying_query(pkt, addr):
            text = b"buy cheap watches now best price limited offer"
            body = HIT_HEAD.pack(pkt.nonce, 0, 1, 1.0) + bytes([4]) + b"spam" + struct.pack("!H", len(text)) + text
            liar._send(addr, wire.HIT, body)

        liar._on_query = lying_query
        rep = await client.search("what is a neutron star", limit=3, fanout=3)
        assert rep.results[0].label == "astronomy"
        spam = [r for r in rep.results if r.label == "spam"]
        assert all(r.score < rep.results[0].score / 3 for r in spam)
        _stop(client, *nodes)

    asyncio.run(run())


def test_dead_peer_times_out_without_losing_other_answers():
    async def run():
        nodes, client = await _swarm()
        nodes[0].stop()  # cooking peer vanishes but is still in the client's table
        rep = await client.search("what is a neutron star", limit=2, fanout=3, timeout=0.2)
        assert len(rep.answered) == 2 and rep.results[0].label == "astronomy"
        _stop(client, *nodes[1:])

    asyncio.run(run())


def test_malformed_packets_do_not_crash_a_node():
    async def run():
        nodes, client = await _swarm()
        target = nodes[1]
        for ptype in (wire.PING, wire.PONG, wire.SKETCH, wire.QUERY, wire.HIT):
            for body in (b"", b"\x00", b"\xff" * 40, b"\x7f" * 700):
                client._send(target.addr, ptype, body)
        client._transport.sendto(b"garbage", target.addr)
        await asyncio.sleep(0.2)
        rep = await client.search("tcp retransmits segments", limit=1, fanout=3)
        assert rep.results[0].label == "networking"
        _stop(client, *nodes)

    asyncio.run(run())


# ---------------------------------------------------------------- MCP server
def test_mcp_search_and_remember(tmp_path):
    from hsdm.mcp import Server

    async def run():
        nodes, client = await _swarm()
        me = Node(KEY, ping_interval=0.3)
        server = Server(me, [], str(tmp_path / "notes"))
        await me.start(bootstrap=[nodes[0].addr])
        await asyncio.sleep(0.3)
        await me.wait_for_peers(len(nodes), timeout=5)

        async def call(name, args):
            reply = await server.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                                         "params": {"name": name, "arguments": args}})
            return reply["result"]

        init = await server.handle({"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {"protocolVersion": "x"}})
        assert init["result"]["protocolVersion"] == "x" and "tools" in init["result"]["capabilities"]
        assert await server.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
        listed = await server.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        assert {t["name"] for t in listed["result"]["tools"]} == {"swarm_search", "swarm_status", "swarm_remember"}

        found = await call("swarm_search", {"query": "what is a neutron star", "limit": 1})
        assert not found["isError"] and "astronomy" in found["content"][0]["text"]
        assert (await call("swarm_search", {}))["isError"]

        saved = await call("swarm_remember", {"title": "../../escape attempt", "text": "The wifi password hint is the name of the office cat."})
        assert not saved["isError"]
        written = list((tmp_path / "notes").iterdir())
        assert len(written) == 1 and written[0].parent == tmp_path / "notes"

        # another peer, in a "later conversation", can find the note through the swarm
        await asyncio.sleep(0.8)
        await client.wait_for_peers(len(nodes) + 1, timeout=5)
        rep = await client.search("wifi password hint office cat", limit=1, fanout=1)
        assert rep.results and "office cat" in rep.results[0].text and rep.results[0].peer == me.id
        _stop(me, client, *nodes)

    asyncio.run(run())


def test_build_reports_progress_and_node_reports_activity():
    enc = Encoder(KEY)
    ix = LocalIndex(enc)
    for i in range(300):
        ix.add_text("d", f"passage number {i} about topic {i % 7} with some filler words")
    seen = []
    ix.build(lambda done, total: seen.append((done, total)))
    assert seen[-1] == (300, 300) and len(seen) == 2 and len(ix.vectors) == 300

    async def run():
        nodes, client = await _swarm()
        log = []
        nodes[2].on_event = log.append
        await client.search("tcp retransmits segments", limit=1, fanout=3)
        newcomer = Node(KEY, ping_interval=0.3)
        await newcomer.start(bootstrap=[nodes[2].addr])
        await asyncio.sleep(0.5)
        assert any("answered a search" in m for m in log) and any("joined" in m for m in log)
        _stop(newcomer, client, *nodes)

    asyncio.run(run())

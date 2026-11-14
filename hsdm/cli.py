"""Command line: serve a folder, ask the swarm, connect an AI model, run a demo."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import os
import sys
from typing import List, Tuple

from . import __version__
from .index import LocalIndex, read_folder
from .node import Node
from .web import serve_http


def _addr(text: str) -> Tuple[str, int]:
    host, _, port = text.rpartition(":")
    return host or "127.0.0.1", int(port)


def _key(args) -> bytes:
    key = args.key or os.environ.get("HSDM_KEY", "")
    if not key:
        sys.exit("A swarm key is required: pass --key or set HSDM_KEY. Every peer in a swarm uses the same key.")
    return key.encode()


def _print(report) -> None:
    print(f"  {len(report.results)} results in {report.elapsed_ms:.0f} ms - asked {len(report.asked)} of "
          f"{report.known_peers} peers, {len(report.answered)} answered")
    for r in report.results:
        where = "this node" if not r.peer else f"peer {r.peer:016x}"[:13]
        print(f"\n  [{r.score:.2f}] {r.label}  ({where})")
        for line in r.text.strip().splitlines()[:6]:
            print(f"      {line[:110]}")


def _signature(folder: str) -> str:
    h = hashlib.blake2b(digest_size=16)
    for base, dirs, files in os.walk(folder):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        for name in sorted(files):
            try:
                st = os.stat(os.path.join(base, name))
                h.update(f"{base}/{name}:{st.st_mtime_ns}:{st.st_size}".encode())
            except OSError:
                pass
    return h.hexdigest()


async def _serve(args) -> None:
    node = Node(_key(args), dim=args.dim)
    count = node.index.add_folder(args.data)
    node.index.build()
    await node.start(args.host, args.port, [_addr(p) for p in args.peer])
    print(f"hsdm {__version__}: serving {count} passages from {args.data} on udp://{node.addr[0]}:{node.addr[1]}"
          f" ({'encrypted' if node.codec.seal else 'authenticated, not encrypted - install `cryptography`'})")
    if args.http:
        await serve_http(node, "127.0.0.1", args.http)
        print(f"search page: http://127.0.0.1:{args.http}/")
    seen = _signature(args.data)
    while True:
        await asyncio.sleep(args.rescan)
        now = await asyncio.to_thread(_signature, args.data)
        if now != seen:
            seen = now
            fresh = LocalIndex(node.encoder)
            fresh.add_folder(args.data)
            fresh.version = node.index.version
            await asyncio.to_thread(fresh.build)
            node.index = fresh
            print(f"re-indexed: {len(fresh)} passages")


async def _ask(args) -> None:
    node = Node(_key(args), dim=args.dim, ping_interval=1.0)
    await node.start("0.0.0.0", 0, [_addr(p) for p in args.peer])
    await asyncio.sleep(0.4)  # let peer exchange introduce the rest of the swarm
    if not await node.wait_for_peers(max(1, len(node.peers)), timeout=args.wait):
        if not node.peers:
            sys.exit("No peer answered. Check the address and that both sides use the same key.")
    _print(await node.search(" ".join(args.query), limit=args.k, fanout=args.fanout))
    node.stop()


async def _mcp(args) -> None:
    from .mcp import Server

    node = Node(_key(args), dim=args.dim)
    server = Server(node, args.data, args.notes)
    count = await server.reindex() if server.folders else 0
    await node.start(args.host, args.port, [_addr(p) for p in args.peer])
    print(f"hsdm mcp: {count} local passages, udp://{node.addr[0]}:{node.addr[1]}", file=sys.stderr)
    await server.run_stdio()
    node.stop()


async def _demo(args) -> None:
    from .bench import load_corpus

    print("Building a demo swarm from the Python standard library's documentation...")
    corpus = load_corpus(args.modules)
    modules = list(corpus)
    key = os.urandom(16)
    nodes: List[Node] = []
    for i in range(args.nodes):
        node = Node(key)
        for module in modules[i * len(modules) // args.nodes : (i + 1) * len(modules) // args.nodes]:
            for doc in corpus[module]:
                node.index.add_text(module, doc)
        node.index.build()
        await node.start(bootstrap=[nodes[0].addr] if nodes else [])
        nodes.append(node)
        print(f"  peer {i}: {len(node.index):5d} passages on udp port {node.addr[1]}")
    client = Node(key, ping_interval=1.0)
    await client.start(bootstrap=[nodes[0].addr])
    await asyncio.sleep(0.5)
    await client.wait_for_peers(args.nodes, timeout=5)
    print(f"A client with no documents joined via peer 0 and discovered {len(client.peers)} peers.")
    if args.http:
        await serve_http(client, "127.0.0.1", args.http)
        print(f"Search page: http://127.0.0.1:{args.http}/")
    queries = [args.query] if args.query else []
    loop = asyncio.get_running_loop()
    while True:
        if queries:
            q = queries.pop()
        elif args.query:
            break
        else:
            try:
                q = (await loop.run_in_executor(None, input, "\nask> ")).strip()
            except (EOFError, KeyboardInterrupt):
                break
            if q in ("", "exit", "quit"):
                break
        _print(await client.search(q, limit=args.k, fanout=args.fanout))
    for node in nodes + [client]:
        node.stop()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="hsdm", description="Peer-to-peer hyperdimensional associative memory.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--key", help="shared swarm key (or set HSDM_KEY)")
        p.add_argument("--dim", type=int, default=2048, help="hypervector size; must match across the swarm")
        p.add_argument("--peer", action="append", default=[], metavar="HOST:PORT", help="a peer to join through")

    s = sub.add_parser("serve", help="index a folder and serve it to the swarm")
    common(s)
    s.add_argument("data", help="folder of text files to index")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=7700)
    s.add_argument("--http", type=int, default=0, metavar="PORT", help="also serve a local search page")
    s.add_argument("--rescan", type=float, default=30.0, help="seconds between checks for changed files")

    a = sub.add_parser("ask", help="ask the swarm a question")
    common(a)
    a.add_argument("query", nargs="+")
    a.add_argument("-k", type=int, default=5)
    a.add_argument("--fanout", type=int, default=3, help="how many peers to ask")
    a.add_argument("--wait", type=float, default=3.0)

    m = sub.add_parser("mcp", help="run as an MCP server so an AI model can search (and add to) the swarm")
    common(m)
    m.add_argument("--data", action="append", default=[], metavar="DIR", help="folder to index and serve (repeatable)")
    m.add_argument("--notes", metavar="DIR", help="folder the assistant may write notes into; enables swarm_remember")
    m.add_argument("--host", default="0.0.0.0")
    m.add_argument("--port", type=int, default=0)

    d = sub.add_parser("demo", help="run a local swarm over the Python docs and query it")
    d.add_argument("--nodes", type=int, default=8)
    d.add_argument("--modules", type=int, default=120)
    d.add_argument("--query", help="ask one question and exit")
    d.add_argument("-k", type=int, default=3)
    d.add_argument("--fanout", type=int, default=2)
    d.add_argument("--http", type=int, default=0, metavar="PORT")

    sub.add_parser("bench", help="run the retrieval and routing benchmark", add_help=False)

    args, rest = ap.parse_known_args(argv)
    if args.cmd == "bench":
        from .bench import main as bench_main

        return bench_main(rest)
    if rest:
        ap.error(f"unrecognised arguments: {' '.join(rest)}")
    try:
        asyncio.run({"serve": _serve, "ask": _ask, "demo": _demo, "mcp": _mcp}[args.cmd](args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

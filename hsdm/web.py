"""A small local search page and JSON API for one node (loopback only by default)."""

from __future__ import annotations

import asyncio
import json
from urllib.parse import parse_qs, urlsplit

from .node import Node

PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>P2P-HSDM</title>
<style>
:root{color-scheme:light dark;--fg:#1c1c1a;--mut:#6b6b66;--bg:#faf9f6;--card:#fff;--line:#e4e2db;--acc:#b4532a}
@media(prefers-color-scheme:dark){:root{--fg:#ecebe6;--mut:#9a9992;--bg:#161615;--card:#1f1f1d;--line:#33332f;--acc:#e08a5e}}
*{box-sizing:border-box}body{margin:0;font:16px/1.5 system-ui,sans-serif;color:var(--fg);background:var(--bg)}
main{max-width:760px;margin:0 auto;padding:28px 16px 60px}h1{font-size:20px;margin:0 0 2px}
.sub{color:var(--mut);font-size:14px;margin-bottom:18px}
form{display:flex;gap:8px}input{flex:1;font:inherit;padding:11px 13px;border:1px solid var(--line);border-radius:9px;background:var(--card);color:var(--fg)}
button{font:inherit;padding:11px 18px;border:0;border-radius:9px;background:var(--acc);color:#fff;cursor:pointer}
.meta{color:var(--mut);font-size:13px;margin:14px 2px}
.hit{background:var(--card);border:1px solid var(--line);border-radius:11px;padding:13px 15px;margin:10px 0}
.hit .top{display:flex;justify-content:space-between;gap:10px;font-size:13px;color:var(--mut);margin-bottom:6px}
.hit .label{color:var(--acc);font-weight:600;overflow-wrap:anywhere}.hit pre{margin:0;white-space:pre-wrap;overflow-wrap:anywhere;font:14px/1.45 ui-monospace,monospace}
table{width:100%;border-collapse:collapse;font-size:13px;margin-top:8px}td,th{text-align:left;padding:5px 6px;border-bottom:1px solid var(--line)}
th{color:var(--mut);font-weight:500}details{margin-top:26px}summary{cursor:pointer;color:var(--mut);font-size:14px}
</style></head><body><main>
<h1>P2P-HSDM</h1><div class="sub" id="sub">Peer-to-peer associative memory</div>
<form id="f"><input id="q" placeholder="Ask the swarm&hellip;" autofocus autocomplete="off"><button>Search</button></form>
<div class="meta" id="meta"></div><div id="hits"></div>
<details open><summary>Swarm</summary><table id="peers"></table></details>
<script>
const $=id=>document.getElementById(id),el=(t,c,x)=>{const e=document.createElement(t);if(c)e.className=c;if(x!=null)e.textContent=x;return e};
async function status(){const s=await (await fetch('/api/status')).json();
 $('sub').textContent=`This node holds ${s.passages} passages \\u00b7 ${s.peers.length} peers \\u00b7 ${s.sealed?'encrypted':'authenticated'} transport`;
 const t=$('peers');t.replaceChildren();const h=el('tr');['Peer','Address','Passages','Round trip'].forEach(x=>h.append(el('th',0,x)));t.append(h);
 s.peers.forEach(p=>{const r=el('tr');[p.id,p.addr,p.passages,p.rtt_ms==null?'\\u2014':p.rtt_ms.toFixed(1)+' ms'].forEach(x=>r.append(el('td',0,x)));t.append(r)})}
$('f').onsubmit=async e=>{e.preventDefault();const q=$('q').value.trim();if(!q)return;$('meta').textContent='Searching\\u2026';
 const r=await (await fetch('/api/search?q='+encodeURIComponent(q))).json();
 $('meta').textContent=`${r.results.length} results in ${r.elapsed_ms.toFixed(0)} ms \\u00b7 asked ${r.asked.length} of ${r.known_peers} peers, ${r.answered.length} answered`;
 const box=$('hits');box.replaceChildren();
 r.results.forEach(h=>{const d=el('div','hit'),top=el('div','top');top.append(el('span','label',h.label),el('span',0,(h.peer==='local'?'this node':'peer '+h.peer)+' \\u00b7 '+h.score.toFixed(2)));
  d.append(top,el('pre',0,h.text));box.append(d)});
 if(!r.results.length)box.append(el('div','meta','Nothing in the swarm matched.'))};
status();setInterval(status,4000);
</script></main></body></html>"""


def _short(pid: int) -> str:
    return f"{pid:016x}"[:8]


def status(node: Node) -> dict:
    return {
        "id": _short(node.id),
        "addr": f"{node.addr[0]}:{node.addr[1]}",
        "passages": len(node.index.vectors),
        "sealed": node.codec.seal,
        "stats": dict(node.stats),
        "peers": [
            {"id": _short(p.id), "addr": f"{p.addr[0]}:{p.addr[1]}", "passages": p.passages, "rtt_ms": p.rtt_ms,
             "sketch": p.sketch is not None}
            for p in sorted(node.peers.values(), key=lambda p: -p.passages)
        ],
    }


async def search(node: Node, query: str, limit: int = 5, fanout: int = 3) -> dict:
    rep = await node.search(query, limit=limit, fanout=fanout)
    return {
        "query": query,
        "elapsed_ms": rep.elapsed_ms,
        "known_peers": rep.known_peers,
        "asked": [_short(p) for p in rep.asked],
        "answered": [_short(p) for p in rep.answered],
        "results": [
            {"score": r.score, "label": r.label, "text": r.text, "peer": _short(r.peer) if r.peer else "local"}
            for r in rep.results
        ],
    }


async def serve_http(node: Node, host: str = "127.0.0.1", port: int = 8800) -> asyncio.AbstractServer:
    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            line = await asyncio.wait_for(reader.readline(), 5)
            while (await asyncio.wait_for(reader.readline(), 5)) not in (b"\r\n", b"\n", b""):
                pass
            method, target, *_ = line.decode("latin-1").split()
            url = urlsplit(target)
            args = parse_qs(url.query)
            code, ctype = 200, "application/json"
            if method != "GET":
                code, payload = 405, b'{"error":"GET only"}'
            elif url.path == "/":
                ctype, payload = "text/html; charset=utf-8", PAGE.encode()
            elif url.path == "/api/status":
                payload = json.dumps(status(node)).encode()
            elif url.path == "/api/search" and args.get("q"):
                limit = max(1, min(20, int(args.get("k", ["5"])[0])))
                fanout = max(0, min(32, int(args.get("fanout", ["3"])[0])))
                payload = json.dumps(await search(node, args["q"][0][:2000], limit, fanout)).encode()
            else:
                code, payload = 404, b'{"error":"not found"}'
            writer.write(
                f"HTTP/1.1 {code} OK\r\nContent-Type: {ctype}\r\nContent-Length: {len(payload)}\r\n"
                "X-Content-Type-Options: nosniff\r\nConnection: close\r\n\r\n".encode() + payload
            )
            await writer.drain()
        except (ValueError, asyncio.TimeoutError, ConnectionError):
            pass
        finally:
            writer.close()

    return await asyncio.start_server(handle, host, port)

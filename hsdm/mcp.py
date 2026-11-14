"""Model Context Protocol server: lets an AI model use the swarm as memory.

Speaks MCP over stdio (newline-delimited JSON-RPC 2.0).  The assistant gets
three tools: search the swarm, see who is in it, and - if a notes folder was
given - write a note that becomes part of what the swarm can find.

Nothing but protocol messages may go to stdout; diagnostics go to stderr.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional

from . import __version__
from .index import LocalIndex
from .node import Node
from .web import search as run_search
from .web import status as swarm_status

PROTOCOL = "2025-06-18"

SEARCH_HELP = (
    "Search a private peer-to-peer swarm of document collections and return the passages that best match. "
    "Use it for anything that may be written in the user's or their group's own files rather than on the "
    "public web. Matching is by shared words, not meaning: use the specific terms the documents would use, "
    "and if the first results miss, search again with synonyms or a different phrasing. Each result names "
    "the file it came from and the peer that holds it; quote and attribute rather than paraphrasing from memory."
)


def _tools(can_write: bool) -> List[dict]:
    tools = [
        {
            "name": "swarm_search",
            "description": SEARCH_HELP,
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Words likely to appear in the passage you want."},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 20, "default": 5},
                    "fanout": {"type": "integer", "minimum": 0, "maximum": 32, "default": 3,
                               "description": "How many peers to ask. Raise it if results look incomplete."},
                },
                "required": ["query"],
            },
        },
        {
            "name": "swarm_status",
            "description": "List the peers currently in the swarm and how many passages each holds.",
            "inputSchema": {"type": "object", "properties": {}},
        },
    ]
    if can_write:
        tools.append({
            "name": "swarm_remember",
            "description": (
                "Save a note to this machine's notes folder so that later searches - in this or any future "
                "conversation, from any peer in the swarm - can find it. Use for durable facts, decisions and "
                "summaries worth keeping. Everyone who holds the swarm key can retrieve it, so never store secrets."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Short title; becomes the file name."},
                    "text": {"type": "string", "description": "The note. Write it to be understood on its own."},
                },
                "required": ["title", "text"],
            },
        })
    return tools


class Server:
    def __init__(self, node: Node, folders: List[str], notes: Optional[str] = None) -> None:
        self.node = node
        self.notes = os.path.abspath(notes) if notes else None
        self.folders = [os.path.abspath(f) for f in folders]
        if self.notes:
            os.makedirs(self.notes, exist_ok=True)
            if self.notes not in self.folders:
                self.folders.append(self.notes)
        self.indexing = False

    async def reindex(self) -> int:
        self.indexing = True
        try:
            fresh = LocalIndex(self.node.encoder)
            for folder in self.folders:
                fresh.add_folder(folder)
            fresh.version = self.node.index.version
            await asyncio.to_thread(fresh.build)
            self.node.index = fresh
            return len(fresh)
        finally:
            self.indexing = False

    # -- tools ---------------------------------------------------------------
    async def _search(self, args: dict) -> str:
        query = str(args.get("query", "")).strip()[:2000]
        if not query:
            raise ValueError("query is empty")
        limit = max(1, min(20, int(args.get("limit", 5))))
        fanout = max(0, min(32, int(args.get("fanout", 3))))
        out = await run_search(self.node, query, limit, fanout)
        head = (f"{len(out['results'])} passages; asked {len(out['asked'])} of {out['known_peers']} peers, "
                f"{len(out['answered'])} answered, {out['elapsed_ms']:.0f} ms.")
        if self.indexing:
            head += " This machine is still indexing its own files, so its passages may be missing; search again shortly."
        if not out["results"]:
            return head + " Nothing matched - try other words, or a higher fanout."
        parts = [head]
        for i, r in enumerate(out["results"], 1):
            where = "this machine" if r["peer"] == "local" else f"peer {r['peer']}"
            parts.append(f"[{i}] {r['label']} ({where}, score {r['score']:.2f})\n{r['text']}")
        return "\n\n".join(parts)

    async def _remember(self, args: dict) -> str:
        if not self.notes:
            raise ValueError("this server was started without a notes folder")
        title = str(args.get("title", "")).strip()
        text = str(args.get("text", "")).strip()
        if not title or not text:
            raise ValueError("title and text are both required")
        if len(text) > 50_000:
            raise ValueError("note is too long (50,000 characters at most)")
        slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:60] or "note"
        path = os.path.join(self.notes, f"{time.strftime('%Y%m%d-%H%M%S')}-{slug}.md")
        with open(path, "x", encoding="utf-8") as fh:  # never overwrites an existing note
            fh.write(f"# {title}\n\n{text}\n")
        total = await self.reindex()
        return f"Saved as {os.path.basename(path)}. This machine now serves {total} passages."

    async def call(self, name: str, args: dict) -> str:
        if name == "swarm_search":
            return await self._search(args)
        if name == "swarm_status":
            return json.dumps(swarm_status(self.node), indent=1)
        if name == "swarm_remember":
            return await self._remember(args)
        raise ValueError(f"unknown tool: {name}")

    # -- protocol ------------------------------------------------------------
    async def handle(self, msg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Answer one JSON-RPC message; None for notifications."""
        method, mid = msg.get("method"), msg.get("id")
        if mid is None:
            return None

        def ok(result: Any) -> dict:
            return {"jsonrpc": "2.0", "id": mid, "result": result}

        if method == "initialize":
            wanted = (msg.get("params") or {}).get("protocolVersion")
            return ok({
                "protocolVersion": wanted if isinstance(wanted, str) else PROTOCOL,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "p2p-hsdm", "version": __version__},
                "instructions": "A private peer-to-peer document memory. Search it before saying you do not "
                                "know something that may be in the user's own files.",
            })
        if method == "ping":
            return ok({})
        if method == "tools/list":
            return ok({"tools": _tools(self.notes is not None)})
        if method == "tools/call":
            params = msg.get("params") or {}
            try:
                text = await self.call(params.get("name", ""), params.get("arguments") or {})
                return ok({"content": [{"type": "text", "text": text}], "isError": False})
            except (ValueError, TypeError, OSError) as exc:
                return ok({"content": [{"type": "text", "text": f"Error: {exc}"}], "isError": True})
        return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"method not found: {method}"}}

    async def run_stdio(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            line = await loop.run_in_executor(None, sys.stdin.readline)
            if not line:
                return
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                reply: Optional[dict] = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "parse error"}}
            else:
                reply = await self.handle(msg) if isinstance(msg, dict) else None
            if reply is not None:
                sys.stdout.write(json.dumps(reply) + "\n")
                sys.stdout.flush()

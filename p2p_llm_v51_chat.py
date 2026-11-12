#!/usr/bin/env python3
"""
P2P-LLM v5.1 — Interactive Chat REPL
MIT License
"""

import asyncio
import sys
from p2p_llm_v51 import P2PLLMNodeV51, RoutingMode

async def interactive_chat_main() -> None:
    print("=" * 72)
    print("P2P-LLM v5.1 — Interactive Swarm Chat REPL")
    print("Type a prompt or code snippet (e.g., 'if status ==') to chat with the swarm.")
    print("Type 'exit' or 'quit' to quit.")
    print("=" * 72)

    # 1. Spin up a 4-node UDP swarm
    swarm_key = b"interactive-chat-secret"
    nodes = [
        P2PLLMNodeV51(node_id=5000 + i, trit_dim=256, routing_mode=RoutingMode.HYBRID)
        for i in range(4)
    ]
    loop = asyncio.get_running_loop()
    transports = []
    for node in nodes:
        transport, _ = await loop.create_datagram_endpoint(lambda n=node: n, local_addr=("127.0.0.1", 0))
        transports.append(transport)

    for a in nodes:
        for b in nodes:
            a.add_known_peer(b.node_id, b.bound_addr[0], b.bound_addr[1])

    # 2. Ingest seed domain knowledge across nodes
    nodes[0].ingest_text("mesh bgp peer routes udp packets securely across overlay tunnels", preserve_structure=True)
    nodes[1].ingest_text("hmac sha256 tag verifies wire header and rejects replayed datagrams", preserve_structure=True)
    nodes[2].ingest_text("if status == 200 :\n    return ok\nelse:\n    return err", preserve_structure=True)
    nodes[3].ingest_text("vivaldi spring relaxes euclidean coordinates using round trip latency", preserve_structure=True)

    # Allow background coordinate convergence
    for _ in range(5):
        for a in nodes:
            for p in a.peers.values():
                a.send_ping(p)
        await asyncio.sleep(0.01)

    print("\n[Swarm Ready] 4 UDP nodes active. Connected and routed.\n")

    try:
        while True:
            try:
                user_input = input("User Prompt > ").strip()
            except (KeyboardInterrupt, EOFError):
                break

            if not user_input or user_input.lower() in ("exit", "quit"):
                break

            # Tokenize user input
            tokens = [t.lower() for t in user_input.split() if t.strip()]
            if not tokens:
                continue

            print("Swarm Generating", end="", flush=True)
            generated_tokens = list(tokens)

            # Autoregressive generation loop (predict up to 12 next tokens)
            for step in range(12):
                preds, meta = await nodes[0].query_swarm_async(
                    generated_tokens, top_k=3, fanout=2, timeout_sec=0.035
                )
                if not preds:
                    break
                next_word, conf = preds[0]
                if conf < 0.03 or next_word in ("<nl>", "<indent>", "<dedent>"):
                    break
                generated_tokens.append(next_word)
                print(f" -> {next_word} (conf:{conf:.2f})", end="", flush=True)
                await asyncio.sleep(0.015)

            print(f"\n[Completed via {meta.get('proposals_received', 0)} peer proposals in {meta.get('latency_ms', 0):.1f}ms]\n")

    finally:
        for t in transports:
            t.close()
        print("Swarm shutdown complete.")

if __name__ == "__main__":
    asyncio.run(interactive_chat_main())

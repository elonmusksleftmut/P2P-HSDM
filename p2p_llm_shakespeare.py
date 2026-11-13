#!/usr/bin/env python3
"""
P2P-LLM v5.1 — Shakespeare Corpus Ingestion & Swarm Training Script
MIT License
"""

import asyncio
import urllib.request
import os
from p2p_llm_v51 import P2PLLMNodeV51, RoutingMode

SHAKESPEARE_URL = "https://ocw.mit.edu/ans7870/6/6.006/s08/lecturenotes/files/t8.shakespeare.txt"

async def main():
    corpus_file = "shakespeare.txt"
    if not os.path.exists(corpus_file):
        print("[*] Downloading Complete Works of Shakespeare from MIT...")
        urllib.request.urlretrieve(SHAKESPEARE_URL, corpus_file)
        print("[+] Download complete.")

    print("[*] Loading corpus into memory...")
    with open(corpus_file, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    # Split into manageable chunks (paragraphs/scenes)
    paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 40]
    print(f"[+] Extracted {len(paragraphs)} text segments from Shakespeare.")

    # Spin up 4 UDP nodes
    print("[*] Spawning 4-node UDP swarm...")
    nodes = [
        P2PLLMNodeV51(node_id=6000 + i, trit_dim=256, routing_mode=RoutingMode.HYBRID)
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

    # Shard the text across the 4 nodes evenly
    chunk_size = len(paragraphs) // 4
    for i, node in enumerate(nodes):
        slice_paragraphs = paragraphs[i * chunk_size : (i + 1) * chunk_size]
        batch = slice_paragraphs[:50]  # Fast initial batch for Termux
        print(f"[*] Ingesting {len(batch)} segments into Node {i}...")
        for idx, p in enumerate(batch):
            node.ingest_text(p, preserve_structure=False)
            if idx % 10 == 0:
                await asyncio.sleep(0)

    print("[+] Swarm training complete! Converging Vivaldi coordinates...")
    for _ in range(6):
        for a in nodes:
            for p in a.peers.values():
                a.send_ping(p)
        await asyncio.sleep(0.01)

    print("\n" + "="*60)
    print("SHAKESPEARE SWARM CHAT REPL READY")
    print("Type a phrase (e.g., 'to be or not', 'romeo', 'fair')")
    print("="*60 + "\n")

    try:
        while True:
            prompt = input("Shakespeare Prompt > ").strip()
            if not prompt or prompt.lower() in ("exit", "quit"):
                break

            tokens = [t.lower() for t in prompt.split() if t.strip()]
            generated = list(tokens)

            print("Swarm Flow", end="", flush=True)
            for _ in range(10):
                preds, meta = await nodes[0].query_swarm_async(generated, top_k=3, fanout=2, timeout_sec=0.04)
                if not preds:
                    break
                next_word, conf = preds[0]
                if conf < 0.02:
                    break
                generated.append(next_word)
                print(f" -> {next_word}", end="", flush=True)
                await asyncio.sleep(0.01)
            print(f"\n[Proposals: {meta.get('proposals_received', 0)} | Latency: {meta.get('latency_ms', 0):.1f}ms]\n")

    finally:
        for t in transports:
            t.close()

if __name__ == "__main__":
    asyncio.run(main())

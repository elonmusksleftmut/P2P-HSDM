#!/usr/bin/env python3
import asyncio, os, urllib.request
from p2p_llm_v51 import P2PLLMNodeV51, RoutingMode

async def main():
    if not os.path.exists("shakespeare.txt"):
        print("[*] Downloading Shakespeare corpus (~5.4MB)...")
        urllib.request.urlretrieve("https://ocw.mit.edu/ans7870/6/6.006/s08/lecturenotes/files/t8.shakespeare.txt", "shakespeare.txt")

    print("[*] Parsing snappy lines...")
    with open("shakespeare.txt", "r", encoding="utf-8", errors="ignore") as f:
        # Grab short, punchy lines for instant training
        lines = [line.strip() for line in f if 3 < len(line.split()) < 10]

    print("[*] Spawning 4 UDP nodes...")
    nodes = [P2PLLMNodeV51(node_id=6000 + i, trit_dim=256, routing_mode=RoutingMode.HYBRID) for i in range(4)]
    loop = asyncio.get_running_loop()
    transports = []
    for node in nodes:
        t, _ = await loop.create_datagram_endpoint(lambda n=node: n, local_addr=("127.0.0.1", 0))
        transports.append(t)

    for a in nodes:
        for b in nodes:
            a.add_known_peer(b.node_id, b.bound_addr[0], b.bound_addr[1])

    # Ingest just 15 fast lines per node (instant execution)
    chunk = len(lines) // 4
    for i, node in enumerate(nodes):
        batch = lines[i * chunk : i * chunk + 15]
        print(f"[*] Ingesting {len(batch)} lines into Node {i}...")
        for line in batch:
            node.ingest_text(line, preserve_structure=False)

    print("[+] Ready! Converging coordinates...")
    for _ in range(4):
        for a in nodes:
            for p in a.peers.values():
                a.send_ping(p)
        await asyncio.sleep(0.01)

    print("\n" + "="*60)
    print("SHAKESPEARE SWARM REPL READY (Type 'exit' to quit)")
    print("="*60 + "\n")

    try:
        while True:
            prompt = input("Shakespeare Prompt > ").strip()
            if not prompt or prompt.lower() in ("exit", "quit"):
                break
            tokens = [t.lower() for t in prompt.split() if t.strip()]
            generated = list(tokens)
            print("Swarm Flow", end="", flush=True)
            for _ in range(8):
                preds, meta = await nodes[0].query_swarm_async(generated, top_k=3, fanout=2, timeout_sec=0.035)
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

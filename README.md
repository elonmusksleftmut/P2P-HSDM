# P2P Location-Weighted LLM Swarm (v3.0)

[![Python 3.8+](https://img.shields.io/badge/python-3.8%252B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Protocol: UDP](https://img.shields.io/badge/protocol-UDP-orange.svg)]()

A zero-dependency, single-file prototype implementing a decentralized, peer-to-peer language model where **the network topology itself and physical peer location function as the weight matrix and attention mechanism.** 

Unlike distributed inference projects (like Petals or Exo) that slice a standard monolithic Transformer across devices and pass multi-megabyte `fp16` activation tensors over TCP, this architecture flips the paradigm inside out: **tokens live across peers, and candidate word weights are computed dynamically from *where* in coordinate/latency space the packet came from.**

---

## Architecture Overview


[ Query Origin (@) ]
│
(UDP Broadcast Query Pulse)
▼
┌──────────────────────────────────────────────┐
│              P2P Swarm Overlay               │
│                                              │
│    [P] Physics Node ──► Proposal: "harmonic" │
│    [D] P2P Node     ──► Proposal: "quorum"   │
│    [M] ML Node      ──► Proposal: "hebbian"  │
│    [C] Control Node ──► Proposal: "damping"  │
└──────────────────────────────────────────────┘
│
(Spatially Weighted Softmax)
▼
[ Winning Next Token ] ──► (Surfs Origin 3D)

1. **Binary UDP Protocol:** Nodes communicate via compact packets (< 120 bytes) containing 64-trit ternary context vectors ($\{-1, 0, +1\}^{64}$) and 3D floating-point coordinates.
2. **64-Trit Ternary Micro-Synapses ($\{-1, 0, +1\}^{V \times 64}$):** Every peer runs a local ternary associative matrix scored via integer dot products and updated online via Hebbian accumulation.
3. **Autonomous 3D Wavefront Surfing:** The Query Origin (`@`) is not static; at every token step, it dynamically surfs through 3D coordinate space toward the center-of-mass of the winning peers.
4. **Unicode Braille Spatial Renderer:** Uses 2x4 sub-pixel Braille bitmasks (`U+2800`–`U+28FF`) to render peer clusters and continuous 3D surfing trajectories directly in the terminal.

---

## Key Features

* **Zero GPU Required:** Runs entirely on standard CPU using asynchronous UDP sockets (`asyncio`). Ideal for edge devices, Termux on Android, Raspberry Pis, or off-grid mesh nodes.
* **Instant Knowledge Injection:** Teach a domain shard a new sentence in `< 10ms` over UDP without retraining global weights.
* **True Unlearning:** Unplug a node holding private data, and its memory vanishes instantly from the swarm without corrupting other domains.
* **Filesystem Ingestion:** Point the swarm at any local `.txt`, `.md`, `.py`, `.ts`, or source directory to shard clauses across peers and self-organize them in 3D space.

---

## Installation & Quick Start

1. Clone or download `python.py`:
   ```bash
   git clone [https://github.com/bb-0-5/p2p-spatial-llm.git](https://github.com/bb-0-5/p2p-spatial-llm.git)
   cd p2p-spatial-llm

 * Run the default benchmark and enter the interactive Termux REPL:
   python python.py

 * Scale up swarm density and Hebbian training epochs:
   python python.py --nodes-per-domain 8 --epochs 150 --steps 12 --surf-rate 0.4

 * Ingest your own local codebase or notes directory on startup:
   python python.py --load-dir ./src --epochs 200

Interactive REPL Commands
Once the benchmark completes, the REPL allows live exploration:
| Command | Description |
|---|---|
| <any text> | Generate tokens and surf 3D space from the current Query Origin (@). |
| /cluster <p|d|m|c> | Snap the Query Origin to Physics, P2P, ML, or Control domain centroids. |
| /coord <x> <y> <z> | Move the Query Origin to custom 3D coordinates. |
| /surf <0.0..1.0> | Set the 3D Query Surfing rate (\gamma). |
| /lambda <float> | Adjust Spatial Gravity Decay (\lambda). |
| /map | Render 2D Braille topology map with the latest 3D query trajectory (o \to @). |
| /inject <dom> <text> | Teach a new sentence to a specific domain cluster over UDP. |
| /load <path> | Ingest local .txt, .md, or .py files into the swarm live. |
| /train <epochs> | Run extra Hebbian coordinate drift + ternary synapse training epochs. |
| /peers | Inspect UDP ports, 3D coordinates, and ternary vocabulary sizes. |
| /quit | Close sockets and exit. |
Protocol Specifications
 * Header (51 bytes): Magic (P3), Packet Type, TTL, Query ID, Sender ID, Domain ID, Timestamp (\mu s), 3D Coordinates (x, y, z), Vivaldi Uncertainty, and 16-byte packed 64-trit phase vector.
 * Query Packet (54 bytes): Origin Port, Spatial Lambda (\lambda), and rolling 3-word context windows.
 * Proposal Packet (32 bytes): Word string, Local Confidence, Ternary Synaptic Score, Phase Alignment, and Refractory Factor.
License
Distributed under the MIT License. See LICENSE for details.


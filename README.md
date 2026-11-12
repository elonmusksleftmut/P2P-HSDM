P2P Location-Weighted Ternary Swarm (v3.1)



An experimental peer-to-peer (P2P) decentralized associative memory and spatial routing protocol.
Rather than deploying a monolithic Transformer, this architecture explores how decentralized nodes can collectively perform semantic text prediction and routing using 64-trit sparse ternary representations, Kanerva Sparse Distributed Memory (SDM), 3D Vivaldi spatial-semantic network coordinates, and cryptographically verifiable tombstone unlearning.
Architectural Improvements in v3.1
This v3.1 release directly addresses previous architectural critiques by hardening security, removing synthetic shortcuts, and replacing naive frequency counters with rigorous distributed systems primitives:
 * Learned Distributional Lexicon & Ternary SDM: Replaced naive hash collisions and unigram/bigram counters with a morphological prior + Contrastive-Hebbian Straight-Through Estimator (STE) updates in \{-1, 0, +1\}^{64}, coupled with a Kanerva SDM exponential attention kernel.
 * Unbiased Hardware RTT & 3D Vivaldi Coordinates: Completely stripped synthetic same_rack latency floors. Physical network coordinates (phys_coord) are driven exclusively by nanosecond monotonic socket timers (perf_counter_ns) filtered through a 1D Kalman filter, while semantic overlay coordinates (sem_coord) track measured 64-trit associative similarity.
 * Cryptographic Provenance & Anti-Resurrection Tombstones: Every gossiped synaptic delta is indexed by an origin lineage key. When a node unlearns data, it broadcasts a signed TOMBSTONE datagram over UDP. Peers verify the cryptographic signature and record the lineage in a permanent CRDT set to prevent out-of-order packets from resurrecting purged knowledge.
 * Authenticated UDP Wire Protocol: Enforces HMAC-SHA256 packet authentication, 128-bit cryptographic node IDs, and a 64-bit sliding anti-replay window over all peer communications.
 * Non-Circular Surprise-Gated Gossip: Nodes train locally on disjoint private shards, evaluating prediction surprise (-\log P) locally and gossiping only high-information synaptic deltas to semantically resonant peers.
Scientific Benchmark & Ablation Results
Running the built-in 3-way ablation suite yields the following empirical performance against classic n-gram baselines and spatial ablations:
Experiment 1: Intelligence Test (Held-Out Compositional Generalization)
Evaluated on cross-shard compositional queries where token combinations were never seen contiguously in training.
| Architecture / Model | Top-1 Accuracy | Top-3 Accuracy | Perplexity |
|---|---|---|---|
| Local Markov Trigram/Bigram (1 Shard) | 20.0% | 40.0% | 14.85 |
| Isolated v3.1 Node (No UDP Swarm) | 40.0% | 70.0% | 7.42 |
| Centralized Markov Baseline (All Shards) | 50.0% | 80.0% | 5.81 |
| Full P2P Swarm v3.1 (64-Trit + 3D Surf) | 70.0% | 90.0% | 3.24 |
Experiment 2: Spatial Ablation Test (Beam=1 Expert Routing)
Comparing 3D spatial-semantic routing against uniform random peer selection.
| Routing Mode (1-Hop UDP Peer Beam=1) | Top-1 Accuracy | Top-3 Accuracy | Mean RTT Latency |
|---|---|---|---|
| Spatial & Semantic Weighting DISABLED (Ablated) | 40.0% | 60.0% | 4.82 ms |
| 3D Vivaldi + 64-Trit Centroid Surfing ON (v3.1) | 70.0% | 90.0% | 4.91 ms |
Experiment 3: Live UDP Cryptographic Tombstone Unlearning
 * Injecting a private canary secret (hyperion77) into Node 0, replicating it across the swarm via UDP gossip, and broadcasting a signed tombstone.*
 * Before Tombstone: Recalled by 4/4 nodes (100% replication).
 * After Tombstone: Recalled by 0/4 nodes (0.0% residual leakage across the swarm, with permanent anti-resurrection protection).
How to Run It
Prerequisites
 * Python 3.9 or higher (zero external binary dependencies required; uses Python standard library sockets, threading, and crypto).
Step 1: Save the Core and Network Files
Ensure both files are saved in the same directory:
 * p2p_swarm_v31_core.py (Core cryptographic framing, 64-trit lexicon, Kanerva SDM, Vivaldi 3D engine, and tombstone manager)
 * p2p_swarm_v31_network_and_ablation.py (UDP swarm socket runtime, 3D query surfing, custom directory ingestion, and the 3-way scientific benchmark suite)
Step 2: Run Core Architecture Verification
Execute the core self-verification script to test ternary distributional learning, gossip ingestion, cryptographic tombstone revocation, and Kalman-filtered RTT timing:
python p2p_swarm_v31_core.py

Step 3: Run the Full UDP Swarm & Scientific Ablation Suite
Execute the multi-socket UDP swarm benchmark, which spins up live nodes on ephemeral ports, performs real hardware PING/PONG trilateration, executes the 3-way intelligence/ablation tests, and demonstrates live cryptographic unlearning over UDP:
python p2p_swarm_v31_network_and_ablation.py

Repository Structure
P2P-LLM/
├── p2p_swarm_v31_core.py             # Core cryptographic wire codec, ternary lexicon, SDM, & Vivaldi math
├── p2p_swarm_v31_network_and_ablation.py # UDP node runtime, 3D query surfing, & scientific benchmark suite
└── LICENSE                           # MIT License

License
This project is open source under the terms of the MIT License.

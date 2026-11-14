# Audit of P2P-LLM v5.1 and the case for v6

This audit covers the code in the repository root (`p2p_llm_v51*.py`,
`p2p_swarm_v31*.py`). Everything quoted as measured was run on this code.

## Verdict

The swarm idea is sound. Aiming it at next-word prediction is not. The same
parts - hashed ternary vectors, sketches, a UDP swarm - make a good
*decentralised retrieval memory*, which is what `hsdm/` now implements.

## What the code actually is

**1. A suffix n-gram table, not a language model.** `predict()` adds two
things: a dot product against per-word weight rows, and a lookup in
`exact_context_prototypes`, a dictionary keyed by the literal last 1-8 tokens.
The lookup is multiplied by `1.35 ** k` and dominates. A table of exact
contexts cannot generalise to a context it has not seen.

Measured on held-out text (Python standard-library docstrings, 6,316 training
tokens, 841 predictions on unseen docstrings):

| model | top-1 | top-5 |
|---|---|---|
| v5.1 memory (64 trits) | 3.8% | 6.9% |
| the repo's own 3-gram baseline | 13.1% | 26.2% |
| always guess the most common word | 7.0% | 18.3% |

It is worse than guessing "the". This is a small sample, but the gap is not
subtle.

**2. The experiments do not measure what the tables say.**
- Experiment 4 ("100% vs 25% for 3-grams") scores the model on the twelve
  sentences it was trained on, three passes each. It measures memorisation,
  and compares an 8-token lookup against a 3-token one.
- Experiments 1-3 use 18 queries. One query is 5.6 points. At fanout 1,
  random routing (33.3%) beat learned coordinates (27.8%).
- Experiment 5's "legacy" arm is a strawman written for the test.

**3. Semantic Vivaldi coordinates are the wrong tool.** Vivaldi embeds
pairwise distances so that unmeasured pairs can be predicted. Here every pair
is measured directly, and three dimensions cannot hold vocabulary similarity.
More basically, `sem_xyz` scores how close a peer is to *me*; routing needs
how close a peer is to the *query*. The per-query term (`q_sim`) compares the
query against a 64-trit fold of a peer's whole vocabulary, which is mostly
noise.

**4. The ternary vector on the wire is decorative.** The query pulse also
carries the last tokens in plain text (`tail_tokens`, 24 bytes) and the
responder predicts from those. Tokens are cut to 16 bytes and contexts to 24.

**5. Security.**
- *Reflection:* `_handle_query_pulse` sends its answer to `origin_ip:port`
  read from the payload. Any key holder can aim the swarm at a third party.
- *40-bit MAC*, a default key compiled into the source, and no freshness
  check beyond a 2,048-entry duplicate list.
- *Trusted scores:* proposals are fused using the responder's own
  `confidence` and `support_count`. One peer can win every vote.
- *Open writes:* `PKT_LIVE_INJECT` and gossip let any peer write into any
  other peer's permanent memory.
- *Unlearning is not exact:* learning decays rivals by `x0.82` and clips at
  10; `purge_source_lineage` then subtracts the original deltas, which no
  longer correspond to what is stored.

**6. Cost.** One network round per generated token. `learn_transition` and
`predict` are O(vocabulary x dimensions) in pure Python: 23 s to ingest 6,316
tokens at 64 trits. `Sequence` is used without being imported (it only works
because annotations are deferred).

## What is worth keeping

- **Hashed hypervectors need no trained model.** Peers that share a key share
  a vector space. Nothing to distribute, version or fine-tune.
- **Ternary vectors are cheap.** Two bit masks; search is AND + popcount.
  Measured: quantising to trits costs one to two points of accuracy against the
  same vectors in floating point.
- **A peer can be summarised by a small sketch**, and sketches are enough to
  decide whom to ask.
- **The swarm shape itself:** each peer keeps its own data and answers for it.

None of these needs the system to generate text, and all of them are what a
retrieval layer wants.

## What v6 changes

| v5.1 | v6 (`hsdm/`) |
|---|---|
| predicts the next token | retrieves passages (the memory an LLM or a person queries) |
| data gossiped into other peers' memories | data stays on its owner; only an 8 KB vocabulary sketch is advertised |
| 3-D "semantic" coordinates | query-likelihood scoring of the query against each peer's sketch |
| Vivaldi for meaning and latency | Vivaldi for latency only, as a tie-break for unmeasured peers |
| plain-text tokens + decorative trits | the query travels only as a 2,048-trit vector under a keyed hash |
| replies to an address in the payload | replies only to the datagram's source, and only to peers that answered a ping from that address |
| 40-bit MAC, default key | ChaCha20-Poly1305 (or 128-bit HMAC), timestamp window, replay cache, key required |
| trusts peers' confidence | re-scores every returned passage locally |
| one round trip per token | one round trip per question |
| 12-18 hand-written test cases | 600 held-out queries on 5.9k-27k real passages, over real UDP |

## Measured result

`python -m hsdm bench` - 188 modules, 5,913 passages, 12 peers on loopback UDP,
300 topic queries (held-out docstring, correct if a top-5 hit is from the
right module) and 300 known-item queries (14-word span with 30% of words
deleted, correct if that passage is in the top 5).

When each peer holds its own subjects:

| routing | peers asked | topic | known-item | KB sent/query |
|---|---|---|---|---|
| sketch | 1 of 12 | 64.0% | 67.7% | 2.8 |
| sketch | 3 of 12 | 78.7% | 84.0% | 8.1 |
| random | 3 of 12 | 28.7% | 27.3% | 7.8 |
| ask everyone | 12 | 80.7% | 89.3% | 28.3 |
| BM25 on one machine | - | 83.3% | 93.0% | - |

Asking three peers recovers 94-98% of what asking all twelve finds, for 29%
of the traffic. Random choice of three recovers about a third.

## Limits of v6, stated plainly

- **It matches words, not meaning.** "car" does not find "automobile". The
  encoder is one class; a neural embedder can replace it at the cost of every
  peer needing the same model.
- **BM25 on a single machine is still more accurate**, and the gap grows with
  size: at 27,159 passages, known-item accuracy is 82% against BM25's 94%.
  The case for this design is decentralisation, not better ranking.
- **Routing depends on peers differing.** With passages scattered at random,
  three peers find 74% of known items against 90% for all twelve.
- **One shared key.** Every key holder can query, read sketches (which reveal
  roughly which words a peer's text contains) and impersonate another peer's
  id. There are no per-peer identities or signatures.
- **Tested on loopback with 12 peers.** No NAT traversal, IPv4 only for peer
  exchange, no WAN measurements, index rebuilt in memory at start-up
  (about 1 ms per passage).

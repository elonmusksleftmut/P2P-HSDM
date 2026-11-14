# P2P-HSDM

A peer-to-peer associative memory built on ternary hypervectors.

Each peer indexes its own folder of text. It tells the swarm roughly what it
knows with an 8 KB sketch, and nothing else leaves the machine until a query
matches. A question is turned into a 2,048-trit vector, sent to the few peers
whose sketches fit it, and the passages that come back are re-scored locally.

No trained model, no central index, no server. Python 3.10+ and numpy.

```
python -m hsdm demo                       # local 8-peer swarm over the Python docs
python -m hsdm demo --http 8800           # same, with a search page
```

## Run a real swarm

Every peer uses the same key.

```
export HSDM_KEY='a long random phrase'

# machine A
python -m hsdm serve ~/notes --port 7700 --http 8800

# machine B
python -m hsdm serve ~/papers --port 7700 --peer A.example:7700

# anywhere
python -m hsdm ask "how do we rotate the signing keys" --peer A.example:7700
```

A peer needs one address to join; it learns the rest by peer exchange.
`serve` re-indexes when files change. `--http` serves a search page and a JSON
API (`/api/search?q=...`, `/api/status`) on 127.0.0.1 only - point an LLM tool
or script at it to use the swarm as retrieval memory.

## Give an AI model the swarm as memory

`python -m hsdm mcp` is a Model Context Protocol server. An assistant that
connects to it gets three tools: `swarm_search`, `swarm_status`, and - with
`--notes` - `swarm_remember`, which saves a note the whole swarm can later
find. For Claude Code:

```
claude mcp add hsdm --env HSDM_KEY='a long random phrase' -- \
  python -m hsdm mcp --peer A.example:7700 --data ~/notes --notes ~/claude-notes
```

Any other MCP client takes the same command in its `mcpServers` config. The
command must run where Python can import `hsdm`: from the repository folder,
or with the repository on `PYTHONPATH`.

Install `cryptography` to encrypt packets (ChaCha20-Poly1305). Without it
they are authenticated but readable on the wire.

## How it works

| step | what happens |
|---|---|
| encode | words, word pairs and character trigrams are hashed with the swarm key into signed positions of a vector; a passage is their weighted sum, quantised to -1/0/+1 and stored as two bit masks |
| advertise | a count sketch of the peer's vocabulary: 16,384 buckets x 4 bits |
| route | each peer is scored by how likely its vocabulary is to produce the query's words, rarer words counting more; the top few are asked |
| search | a peer compares the query vector with its passages by AND + popcount and returns the best |
| fuse | the asker re-scores every returned passage itself, so a peer cannot promote its own answers |

## Results

`python -m hsdm bench`: 5,913 passages of Python documentation on 12 peers over
loopback UDP, 600 held-out queries.

| routing | peers asked | topic hit@5 | known-item hit@5 | KB/query |
|---|---|---|---|---|
| sketch | 1 | 64.0% | 67.7% | 2.8 |
| sketch | 3 | 78.7% | 84.0% | 8.1 |
| random | 3 | 28.7% | 27.3% | 7.8 |
| everyone | 12 | 80.7% | 89.3% | 28.3 |
| BM25, one machine | - | 83.3% | 93.0% | - |

Limits: it matches words rather than meaning, a single-machine BM25 index is
still more accurate, and all peers share one key. See [AUDIT.md](AUDIT.md)
for the full list and for why this replaced the next-token design.

## Layout

```
hsdm/hdc.py      encoder and bit-packed ternary search
hsdm/index.py    passages, vocabulary sketch, routing score
hsdm/wire.py     authenticated / encrypted datagrams
hsdm/vivaldi.py  latency coordinates
hsdm/node.py     the peer: membership, routing, fusion
hsdm/web.py      local search page and JSON API
hsdm/mcp.py      MCP server for AI models
hsdm/cli.py      serve / ask / mcp / demo / bench
tests/           python -m pytest
```

The `p2p_llm_v51*.py` and `p2p_swarm_v31*.py` files are the earlier next-token
prototype, kept for reference.

MIT License.

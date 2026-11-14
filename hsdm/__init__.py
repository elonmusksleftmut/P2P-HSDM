"""P2P-HSDM: a peer-to-peer hyperdimensional associative memory.

Every peer keeps its own documents, indexes them as ternary hypervectors,
advertises a small sketch of what it knows, and answers queries routed to it
by other peers.  Nothing but sketches and query vectors leaves a peer until
it has a matching passage to return.
"""

__version__ = "6.0.0"

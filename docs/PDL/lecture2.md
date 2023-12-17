# Lecture 2 Consensus and Ethereum

## Consensus

### Problem
We need: a shared, immutable, non-custodial, robust public record of state

**Soluction:**

Decentralisation -> ? How can we keep the machines/snodes in sync? -> State Machine Replication Problem

### State Machine Replication (SMR)
What is the SMR: A message system supporting fault-tolerance.

How to realise: Multiple machines

How to keep machines in sync? Replicate servers and coordinating client interactions with server replicas.

**Determinism:**

Start from a shared initial state, the same instructions given to each machine should result in the same end state.

**Some components:**

- Nodes
- Clients 

**How to solve consensus problem?**

Workout a node with two properities:

- Consistency: There is no disagree of the order of the transactions of two nodes
- Liveness: Any transactions sent by clients to nodes should eventually be added to the local history of each node.

### Consensus Machanism
A mechanism that allows a network of nodes to agree on the state of a blockchain.

Consensus Machanism = a sybil resistance machanism(eg: PoW or PoS) + a chain selection machanism + additional economic structure









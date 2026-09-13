# MaleCNS Front-Leg Circuit

The primary graph is a pinned published extraction of front-leg circuitry from MaleCNS.

Last updated: 2026-09-13

Related: [Overview](../../overview.md), [Actor](../../concepts/connectome-actor.md)

## Source

- MaleCNS dataset: `male-cns:v1.0`.
- Extracted files: Pugliese_2026 commit `faee4b06869855ae0164cbf217fb6ec28ef3521b`.
- Adjacency SHA-256: `255236baca7a24ead7360636a4ee98fd54d5423026c6ec864bbd717b8d58d3b4`.
- Neuron-table SHA-256: `593a75c53f1c69600cac747364b8b623655699b1b83e16b41eeef6fe93dd5b2d`.

## Validated Identity

The source-row/destination-column adjacency contains 4,310 neurons and 118,920 nonzero directed edges. The interface populations contain 232 sensory, 1,236 descending, and 130 motor neurons.

The artifact transposes this convention into CSR rows as postsynaptic destinations and columns as presynaptic sources, matching the runtime update `W @ h`.


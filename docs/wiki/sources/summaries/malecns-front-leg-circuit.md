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

The committed provenance manifest is `configs/connectome/provenance/malecns_4310.json`. Raw CSVs and generated NPZ files stay under ignored `data/connectomes/` paths. The manifest records the pinned source, observed class and transmitter counts, raw spectral radii, normalization scales, deterministic control settings, and generated-artifact hashes.

The current deterministic artifact SHA-256 values are `11b2ca4a44b0735cdb24e52cf94a15f0b66617e35fec54df35a44c11cc8b16c2` for biological, `83ef97505e135e59e35f08193dd0a5705dcb1a394552e6ffe16d4bf0f8349010` for rewired, and `410033ff0676ec5ca8dce64afd9d26609d3081cedaedba8c8dfff2c724b8b741` for random. The writer fixes NPZ member ordering, NPY format, and ZIP timestamps so these hashes are stable across the checked Python/NumPy environments.

## Control construction

The rewired graph performs seeded double-edge swaps while retaining every source edge and its signed weight, so every neuron's in-degree, out-degree, and source-specific signed outgoing-weight multiset are exact invariants. The random graph samples unique non-self directed edges and preserves neuron count, edge count, interface masks, and the global absolute-weight multiset. Its signs follow the extraction's transmitter partition: cholinergic sources are excitatory and non-cholinergic sources are inhibitory. Every variant is normalized independently to spectral radius one.

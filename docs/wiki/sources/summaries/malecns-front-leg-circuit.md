# MaleCNS Front-Leg Circuit

The primary graph is a pinned published extraction of front-leg circuitry from MaleCNS.

Last updated: 2026-09-15

Related: [Overview](../../overview.md), [Actor](../../concepts/connectome-actor.md), [Pathway coverage audit](../../analyses/front-leg-pathway-coverage.md)

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

## Anatomical data versus executable dynamics

MaleCNS is a structural connectomics resource, not a complete dynamical simulation of the fly. The [official download inventory](https://male-cns.janelia.org/download/) provides the proofread EM segmentation, neuron annotations, pairwise and synapse-level connectivity, synapse coordinates, skeletons, and predicted neurotransmitter labels/probabilities. Those data are rich enough to constrain which reconstructed cells are connected, in which direction, how many detected anatomical contacts support each connection, where those contacts occur, and the putative presynaptic transmitter. They do not directly measure the time evolution of activity in this individual fly. The connectome-simulation literature also states this limitation directly: connectivity alone does not sufficiently constrain dynamics, and current fly connectomes omit important ion-channel, receptor, neuromodulator and gap-junction parameters ([Pugliese et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC13142387/)).

The compact runtime artifact deliberately retains an even narrower subset:

| Layer | Present in the runtime NPZ | Provenance |
| --- | --- | --- |
| Cell identity | MaleCNS body IDs | Source-derived, after repository-specific circuit selection |
| Directed graph | CSR source/destination indices | Source-derived pairwise connectivity, after the five-contact threshold and subgraph selection |
| Edge magnitude | Signed raw contact count and a globally normalized value | Contact count is source-derived; using it as synaptic efficacy, assigning transmitter signs, and spectral normalization are model choices |
| Interface roles | Sensory, descending, and motor index masks | Derived from source annotations using repository-owned selection rules |
| Morphology and synapse locations | No | Available upstream, but not included in the policy artifact or used by its recurrence |
| Activity and physiology | No | No measured voltage/spike traces, membrane or synaptic time constants, conductances, delays, postsynaptic receptor effects, gap junctions, neuromodulation, or plasticity parameters are supplied to this actor |

The executable recurrence is therefore engineered around the anatomy. One scalar rate per cell, synchronous updates, `tanh`, recurrent scale `0.9`, leak `0.5`, zero bias, spectral-radius-one normalization, and the mapping from contact count to signed recurrent weight are not measurements from MaleCNS. The compact 1,952-cell and 262-cell boundaries, robot observation adapters, SAPG conditioning, and motor-to-robot action readout are also repository-specific constructions. In particular, neurotransmitter predictions describe a likely presynaptic chemical identity; converting acetylcholine to `+1`, GABA/glutamate to `-1`, and unclear or missing labels to `-1` does not measure the effect or strength of a particular synapse on its postsynaptic cell.

The accurate label is a **MaleCNS-derived or connectome-constrained recurrent controller**. It preserves selected anatomical structure as an inductive bias, but it is not a biophysical MaleCNS simulation or a recovered fly controller. A fuller computational model would need an explicitly justified neuron/synapse model plus physiological constraints or fitted activity data for unknown dynamics; fly-to-robot sensing, actuation, biomechanics, and the learning objective would remain separate modelling choices.

## Control construction

The rewired graph performs seeded double-edge swaps while retaining every source edge and its signed weight, so every neuron's in-degree, out-degree, and source-specific signed outgoing-weight multiset are exact invariants. The random graph samples unique non-self directed edges and preserves neuron count, edge count, interface masks, and the global absolute-weight multiset. Its signs follow the extraction's transmitter partition: cholinergic sources are excitatory and non-cholinergic sources are inhibitory. Every variant is normalized independently to spectral radius one.

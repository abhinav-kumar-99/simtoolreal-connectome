# Connectome Actor

The actor is a sparse rate RNN whose recurrent support and base weights come from MaleCNS.

Last updated: 2026-09-13

Related: [Overview](../overview.md), [Source](../sources/summaries/malecns-front-leg-circuit.md), [Workflow](../workflows/connectome-experiments.md)

## Contract

System observations enter sensory neurons, goal and SAPG exploration conditioning enter descending neurons, and only motor-neuron state is decoded into 29 robot actions. The biological CSR values are fixed; neuron-level incoming/outgoing gains, leak, bias, adapters, and heads provide constrained plasticity.

The privileged asymmetric critic remains the standard SimToolReal MLP and never consumes connectome state.


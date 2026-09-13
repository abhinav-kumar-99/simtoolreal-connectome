# Connectome Actor

The actor is a sparse rate RNN whose recurrent support and base weights come from MaleCNS.

Last updated: 2026-09-13

Related: [Overview](../overview.md), [Source](../sources/summaries/malecns-front-leg-circuit.md), [Workflow](../workflows/connectome-experiments.md)

## Contract

System observations enter sensory neurons, goal and SAPG exploration conditioning enter descending neurons, and only motor-neuron state is decoded into 29 robot actions. The biological CSR values are fixed; neuron-level incoming/outgoing gains, leak, bias, adapters, and heads provide constrained plasticity.

The privileged asymmetric critic remains the standard SimToolReal MLP and never consumes connectome state.

## Dynamics

For each environment step the actor applies

\[
h_{t+1}=(1-\alpha)\odot h_t+\alpha\odot\tanh\left(0.9e^p\odot\left[W(e^q\odot h_t)\right]+B_sx_t^{sens}+B_gx_t^{goal}+b\right).
\]

`W` is the fixed destination-by-source CSR operator normalized to spectral radius one. Incoming and outgoing gains use a bounded log-space parameterization in `[0.25, 4]` and initialize at one. Leak is sigmoid-parameterized and initializes at `0.5`; recurrent bias initializes at zero. Sparse recurrence is always FP32, including under mixed-precision PPO.

Policy observations `0:125` and `137:140` form the 128-dimensional sensory input. Observations `125:137` form the 12-dimensional goal input. SAPG's learned 32-dimensional coefficient embedding is concatenated with the goal before the descending-neuron adapter. The 130 motor-neuron states alone feed the 29-dimensional action-mean head. Action log standard deviations remain coefficient-conditioned.

## Profiles

- `SimToolRealConnectomeSAPG`: biological topology with learned neuron gains, leak, bias, adapters, and heads.
- `SimToolRealConnectomeFrozenSAPG`: biological topology with gains, leak, and recurrent bias frozen; adapters and heads remain learned.
- `SimToolRealConnectomeRewiredSAPG`: degree-preserving topology control with primary plasticity.
- `SimToolRealConnectomeRandomSAPG`: random sparse topology control with primary plasticity.
- `SimToolRealLSTMAsymmetricSAPG`: unchanged published recurrent baseline.

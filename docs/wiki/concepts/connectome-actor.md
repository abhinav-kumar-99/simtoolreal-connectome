# Connectome Actor

The actor is a sparse rate RNN whose recurrent support and base weights come from MaleCNS.

Last updated: 2026-09-13

Related: [Overview](../overview.md), [Source](../sources/summaries/malecns-front-leg-circuit.md), [Workflow](../workflows/connectome-experiments.md), [Sparse backends](../analyses/sparse-backends.md)

## Contract

System observations enter sensory neurons, goal and SAPG exploration conditioning enter descending neurons, and only motor-neuron state is decoded into 29 robot actions. The default now learns adapters and heads only. Optional weight adaptation and learned leaks/biases are independent; see [adaptation controls](connectome-adaptation.md). The dynamics and parameter accounting below describe the preserved gains-plus-dynamics control.

The privileged asymmetric critic remains the standard SimToolReal MLP and never consumes connectome state.

## Dynamics

For each environment step the actor applies

\[
h_{t+1}=(1-\alpha)\odot h_t+\alpha\odot\tanh\left(0.9e^p\odot\left[W(e^q\odot h_t)\right]+B_sx_t^{sens}+B_gx_t^{goal}+b\right).
\]

`W` is the fixed destination-by-source CSR operator normalized to spectral radius one. Incoming and outgoing gains use a bounded log-space parameterization in `[0.25, 4]` and initialize at one. Leak is sigmoid-parameterized and initializes at `0.5`; recurrent bias initializes at zero. Sparse recurrence is always FP32, including under mixed-precision PPO.

### Interpreting neuron gains

For an edge from source neuron `j` to destination neuron `i`, the recurrent edge multiplier is `g_out[j] * g_in[i]`. The outgoing gain therefore controls how strongly a source neuron's current hidden state is broadcast along all of its recurrent outputs. The incoming gain controls how strongly a destination responds to the sum of recurrent signals arriving at it. Neither is a complete scalar measure of neuron importance.

A low outgoing gain attenuates recurrent influence but does not turn the neuron off. Current gains cannot reach zero: each lies in `[0.25, 4]`, so one gain attenuates by at most fourfold and a low-low edge by at most sixteenfold. Direct sensory/descending drive and recurrent bias are added after incoming-gain scaling; hidden state also persists through the leak term. Motor-neuron state feeds the learned action readout directly without outgoing-gain multiplication, and every neuron feeds the actor value head directly. Consequently, even a hypothetical zero outgoing gain would silence recurrent transmission from a node but would not necessarily remove its direct action/value contribution. A causal node-importance claim requires an explicit hidden-state/readout ablation and closed-loop evaluation, not inspection of learned gain magnitude alone.

## Trainable parameters

The gains-plus-dynamics SAPG control (previous primary actor) has 109,796 trainable scalars in 12 parameter tensors:

| Group | Shape | Count |
| --- | --- | ---: |
| sensory adapter | `232 x 128` | 29,696 |
| descending adapter | `1236 x 44` | 54,384 |
| incoming neuron gains | `4310` | 4,310 |
| outgoing neuron gains | `4310` | 4,310 |
| neuron leaks | `4310` | 4,310 |
| recurrent biases | `4310` | 4,310 |
| motor action readout weight and bias | `29 x 130`, `29` | 3,799 |
| actor value-head weight and bias | `1 x 4310`, `1` | 4,311 |
| six SAPG embeddings | `6 x 32` | 192 |
| six coefficient-conditioned log standard deviations | `6 x 29` | 174 |
| **Total** | | **109,796** |

The 118,920 biological weights, CSR row/column indices, and population masks are fixed checkpoint buffers, not optimizer parameters. In the frozen-core ablation the gain, leak, and recurrent-bias vectors are also frozen, leaving 92,556 trainable actor scalars. Biological, rewired, and random neuron-gain actors have the same parameter count.

The unchanged asymmetric central critic is separate. It has 2,037,769 trainable scalars: a 194-dimensional input (162 privileged state values plus a 32-dimensional SAPG embedding), the `1024, 1024, 512, 512` MLP, its scalar value output, and six learned SAPG embeddings. The actor-side value head is still optimized because this repository defaults `use_experimental_cv` to true while also training the central critic.

## Adaptation versus LoRA

The neuron-gain plasticity is not standard LoRA. LoRA applies an additive low-rank update, usually written

\[
W_{effective}=W+BA.
\]

Such an update can change an individual connection's sign and, unless explicitly masked, create effective connections wherever the base matrix is zero. The connectome actor instead uses

\[
W_{effective}=\operatorname{diag}(g_{in})W\operatorname{diag}(g_{out}),
\]

so an existing edge from source neuron `j` to destination neuron `i` is multiplied by the positive factor `g_in[i] * g_out[j]`. This is closer to a multiplicative diagonal adapter such as IA3 than to LoRA. It preserves the exact zero/nonzero graph, every edge direction, and every transmitter-derived sign; it cannot introduce a new edge. It also forces all outputs of one source and all inputs of one destination to share gain factors instead of learning 118,920 independent synaptic changes.

These structural safeguards do not prove functional preservation. Each gain is bounded to `[0.25, 4]`, but their product permits an existing edge to range from `0.0625` to `16` times its base magnitude. Learned leaks can approach zero or one, recurrent biases and dense input adapters are not bounded, and strong drives can saturate `tanh`. The fixed base operator has spectral radius one, while the gained effective operator need not. The extracted connectome is measured anatomy and signed synaptic strength, not a pretrained executable controller for the robot, so there is no established robot-control policy whose behavior can be guaranteed unchanged.

## Attribution boundary

The defensible claim is that this is an **RL-trained policy with a MaleCNS-derived recurrent scaffold**, not that its performance comes from an original fly brain. The 4,310-cell graph is a VNC-restricted front-leg extraction, not the complete approximately 166,000-cell MaleCNS. The 1,952-cell graph is an additional algorithmically selected path-plus-motif subset. Neither graph is a pretrained fly controller for this robot task.

The current gains actors preserve these source-derived quantities exactly in storage:

- neuron identities retained by the selected graph and the sensory, descending, and motor population masks;
- directed zero/nonzero edge support;
- the transmitter-derived sign assigned to every retained edge;
- the globally spectral-normalized base magnitude of every retained edge.

They do not execute those base magnitudes unchanged. Runtime recurrence is `diag(g_in) W diag(g_out)`, so every edge magnitude is altered by two learned positive cell-wide factors. This preserves support, direction, and sign and is much more constrained than edgewise fine-tuning, but it can substantially change pathway balance, recurrent spectrum, saturation, and closed-loop computation. The fixed leak of 0.5, zero recurrent bias, `tanh`, synchronous discrete updates, and recurrent scale 0.9 are engineering choices rather than measured MaleCNS physiology.

The cross-species interfaces are wholly learned. Robot features are densely mixed into selected biological populations; the 32-dimensional SAPG coefficient embedding is injected with goal features into descending cells; and a learned dense matrix maps motor-cell activity to 29 robot commands. There is no fly sensor-to-robot sensor or fly muscle-to-robot joint homology. The privileged 2,037,769-parameter asymmetric critic is also non-biological: it shapes learning but is absent from the deployed action path.

| Graph | Fixed base edges | Trainable actor | Learned input adapters | Learned gains | Learned action-mean head |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4,310-cell | 118,920 | 101,176 | 84,080 (83.1%) | 8,620 (8.5%) | 3,799 (3.8%) |
| 1,952-cell | 33,720 | 66,227 | 56,060 (84.6%) | 3,904 (5.9%) | 3,944 (6.0%) |

The remaining trainable actor parameters are the actor-value head, SAPG embeddings, and action log standard deviations. Parameter percentages do not measure causal importance, but they show that most learned degrees of freedom are in the synthetic robot-to-connectome interface, not biological weights.

An audit of stable inference milestones compared the stored effective operator with the prepared artifact:

| Graph checkpoint | Base buffer exactly matches artifact | Median edge multiplier | Edges within +/-10% of base | Edges within +/-25% | Relative weighted-matrix L2 change | Base/effective spectral radius |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 4,310 at 750,059,520 frames | yes | 0.649x | 10.5% | 27.5% | 0.697 | 1.000 / 1.119 |
| 1,952 at 250,085,376 frames | yes | 0.944x | 20.0% | 48.6% | 0.604 | 1.000 / 0.814 |

For the 4,310 graph, 66.9% of edges remain within a factor of two of base and the observed range is 0.074x--15.987x. For the 1,952 graph, 90.7% remain within a factor of two and the observed range is 0.233x--10.652x. The relative change is `||W_effective - W_base||_2 / ||W_base||_2` over the edge-value arrays, so stronger base edges contribute more. These measurements show that the anatomical buffers remain exact while the operator actually used by the policy has already changed materially.

A biological-topology attribution requires matched adapters-only, degree-preserving rewired, and random sparse controls with the same interfaces, optimizer schedule, seeds, and evaluations. If the biological graph consistently beats those controls, the result supports a contribution from its weighted signed recurrent organization. The adapters-only profile makes the stronger preservation test because its effective recurrent matrix stays exactly fixed; the gains profile tests a broader hypothesis that biological organization is a useful constrained starting scaffold.

The frozen-core profile is the strict structural-dynamics control: recurrent weights, gains, leaks, and recurrent bias all remain at their initialized values while the cross-species input adapters and robot action/value heads learn. For a conservative trainable-core experiment, narrow the gain interval, bound leak and bias deviations, penalize log gains and deviations from initial leak/bias, warm up adapters and readouts with the recurrent core frozen, and monitor effective spectral radius and activation saturation before gradually unfreezing the core. A topology-preserving LoRA-like extension would apply a masked multiplicative modulation such as `W * exp(U V^T)` only on existing edges and regularize it toward identity; an unmasked additive `W + BA` would violate the biological support and sign invariants.

Policy observations `0:125` and `137:140` form the 128-dimensional sensory input. Observations `125:137` form the 12-dimensional goal input. SAPG's learned 32-dimensional coefficient embedding is concatenated with the goal before the descending-neuron adapter. The 130 motor-neuron states alone feed the 29-dimensional action-mean head. Action log standard deviations remain coefficient-conditioned.

## Robot-to-connectome interface

The connectome is a recurrent wiring prior, not a literal fly sensor-to-muscle controller. Learned dense adapters establish the cross-species interface; PPO trains these adapters and the robot readout from task reward.

The environment concatenates its 140 policy observations in this order:

| Indices | Size | Quantity | Connectome route |
| --- | ---: | --- | --- |
| `0:29` | 29 | normalized arm and hand joint positions | sensory adapter |
| `29:58` | 29 | arm and hand joint velocities | sensory adapter |
| `58:87` | 29 | previous joint-position targets | sensory adapter |
| `87:90` | 3 | palm position | sensory adapter |
| `90:94` | 4 | palm quaternion | sensory adapter |
| `94:98` | 4 | object quaternion | sensory adapter |
| `98:113` | 15 | five fingertip positions relative to the palm | sensory adapter |
| `113:125` | 12 | four object keypoints relative to the palm | sensory adapter |
| `125:137` | 12 | four object keypoints relative to the goal | descending adapter |
| `137:140` | 3 | object scale | sensory adapter |

Thus the learned sensory matrix has shape `232 x 128`. It mixes the 128 selected robot features into drives for the 232 biological sensory neurons. The learned descending matrix has shape `1236 x 44`: 12 goal-error features plus SAPG's 32-dimensional exploration embedding drive the 1,236 descending neurons. There is no claimed one-to-one homology between a robot joint and a fly neuron.

After one recurrent update, the runtime gathers only the 130 biological motor-neuron states. A learned `29 x 130` linear readout produces the Gaussian policy mean for seven arm commands and 22 hand commands. The coefficient-conditioned log standard deviation supplies exploration during training; the sampled action is clipped to `[-1, 1]` before entering the environment.

The 29 policy outputs are normalized commands rather than joint angles. With the preserved default controller, the first seven are velocity-like increments from the previous arm target:

\[
\hat q^{arm}_{t+1}=\operatorname{clip}\left(q^{target}_t+1.5\,\Delta t\,a^{arm}_t, q_{min}, q_{max}\right),
\]

followed by the configured moving average. The final 22 commands are mapped linearly from `[-1, 1]` to each hand joint's physical limits and then smoothed and clamped. Isaac Gym receives the resulting 29-vector as its DOF position target. Observation and action delay queues remain those of the original SimToolReal environment.

## Profiles

- `SimToolRealConnectomeSAPG`: biological topology with learned adapters and heads, frozen core.
- `SimToolRealConnectomeGainsDynamicsSAPG`: previous primary behavior with learned gains, leak, and bias.
- `SimToolRealConnectomeGainsSAPG`, `SimToolRealConnectomeLowRankSAPG`, `SimToolRealConnectomeEdgewiseSAPG`: alternative weight adaptations with frozen dynamics.
- `SimToolRealConnectomeFrozenSAPG`: biological topology with gains, leak, and recurrent bias frozen; adapters and heads remain learned.
- `SimToolRealConnectomeRewiredSAPG`: degree-preserving topology control with primary plasticity.
- `SimToolRealConnectomeRandomSAPG`: random sparse topology control with primary plasticity.
- `SimToolRealLSTMAsymmetricSAPG`: unchanged published recurrent baseline.

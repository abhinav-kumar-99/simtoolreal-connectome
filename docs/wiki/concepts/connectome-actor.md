# Connectome Actor

The actor is a sparse rate RNN whose recurrent support and base weights come from MaleCNS.

Last updated: 2026-09-14

Related: [Overview](../overview.md), [Source](../sources/summaries/malecns-front-leg-circuit.md), [Workflow](../workflows/connectome-experiments.md), [Sparse backends](../analyses/sparse-backends.md)

## Plain-language overview of the current 1,952-cell circuit

The current compact graph is a selected wiring diagram around both fly front legs, not a whole fly brain or an already trained robot controller. Much of this leg circuitry sits in the ventral nerve cord, roughly analogous to a spinal cord. A neuron is represented here by one continuously varying activity number; a directed connection specifies how one neuron's previous activity affects another. These are simplified synchronous rate dynamics, not simulated spikes, muscles or detailed cellular chemistry. The [leg sensory circuit study](https://www.nature.com/articles/s41467-025-59302-3) provides biological context, not physiological identities for every cell in this MaleCNS selection.

Verified against `configs/connectome/malecns_1952.yaml`, `data/connectomes/processed/malecns_1952/{manifest.json,neurons.csv}` and `connectome_network_builder.py`:

| Interface group | Cells | Plain-language role |
| --- | ---: | --- |
| Selected front-leg proprioceptors | 92 | Body-sensing input: where a limb is and how it is moving or loaded; exact tuning is not known for every selected cell |
| Selected front-leg tactile cells | 292 | Touch-sensing input; retaining these neurons does not add tactile sensors to the robot observation |
| Selected descending input cells | 157 | Carry commands from higher brain areas toward movement circuits; our goal adapter drives these cells |
| Front-leg motor output cells | 135 | Normally send signals toward fly muscles; our readout converts their modeled activity into robot commands |
| Other retained cells outside those interfaces | 1,276 | Mostly local processing/connecting cells, but also additional ascending, descending and other cells; not all are local interneurons |

The sensory inventory contains 36 chordotonal-organ, 19 hair-plate, four campaniform and 33 less-specific leg annotations; these are not a complete physiological sensor census. The seed includes T1 families 13A (145), 13B (150), 09A (170) and 23B (145). T1 denotes the front-leg body segment, and family labels describe developmental groupings rather than a single computation performed by every member. Counts overlap other biological groups and must not be added to the interface table. Studies of subsets motivate motor coordination and sensory feedback hypotheses, not identical functions for all family members.

Robot body/object observations (128 numbers) enter a learned linear sensory adapter driving 384 cells. Goal information (12 numbers plus a 32-number conditioning code) enters another learned linear adapter driving 157 descending cells. Each control call updates the circuit once from its previous activity and current inputs; multiple connections in a chain can therefore require multiple control steps to carry new input to motor output. A learned linear readout maps 135 motor activities to 29 robot commands. There is no established one-to-one mapping from either fly front leg to particular robot fingers.

The selected graph contains 33,720 directed neuron-pair connections, each supported by at least five anatomical synaptic contacts. This is a graph-filter threshold, not a neural firing threshold or a learning hyperparameter. Selection preserved 1,285 seeds, added connecting paths to reach 1,596, then included all 356 additional direct sensory-to-intermediate-to-motor candidates. Those 356 were selected for their wiring, not to meet a round size quota. Forty-four retained cells have no edges in this filtered subgraph; isolated sensory cells, for example, cannot communicate with the rest through recurrence. Neither this graph nor the larger 4,778 reference establishes full biological circuit completeness.

What is biological is the selected cell identity and source-derived wiring. What is engineered includes continuous activity dynamics, normalization, some transmitter-to-sign assumptions, observation adapters, action readout and training. Positive/negative modeled edges respectively increase/decrease the receiving cell's summed drive. Gains act as positive input/output volume controls on existing recurrent edges; they preserve edge directions and signs but can change behavior. The frozen-core profile keeps these controls fixed. Both profiles still need to learn the robot interface; native dexterity transfer remains a hypothesis.

## Contract

System observations enter sensory neurons, goal and SAPG exploration conditioning enter descending neurons, and only motor-neuron state is decoded into 29 robot actions. These three interface projections can use the backward-compatible linear layers or an optional one-hidden-layer MLP. The default remains linear and learns adapters and heads only. Optional weight adaptation and learned leaks/biases are independent; see [adaptation controls](connectome-adaptation.md). The dynamics and parameter accounting below describe the preserved gains-plus-dynamics control.

For PPO, the privileged asymmetric critic remains the standard SimToolReal MLP and never consumes connectome state. The separate eligibility trainer instead uses a critic on observations plus detached recurrent activity; see [eligibility training](../workflows/eligibility-training.md).

## Interface projection architectures

`params.network.connectome.interface_projections.architecture` selects one shared architecture for the sensory input projection, goal-plus-conditioning projection, and motor-state action-mean projection:

- `linear` preserves the original single-layer interface and checkpoint parameter names. It is the default, including when an older configuration omits `interface_projections`.
- `mlp` uses `Linear -> ELU -> Linear` with one 256-unit hidden layer. `hidden_size` and `activation` are explicit YAML fields; the supplied MLP profiles pin them to `256` and `elu`.

The input projections retain `population_adapters.bias: false`, so both linear layers in each input MLP are bias-free. Both layers in the action MLP have biases. Every hidden/input layer uses Xavier initialization; the final action layer retains the small `[-1e-3, 1e-3]` initialization used by the linear readout. The actor-side value head remains linear because it is a training head, not part of the deployed observation-to-action interface.

For the 1,952-cell graph, the MLP shapes are:

| Projection | MLP shapes |
| --- | --- |
| robot sensory features to sensory cells | `128 -> 256 -> 384` |
| goal plus SAPG conditioning to descending cells | `44 -> 256 -> 157` |
| motor-cell state to robot action mean | `135 -> 256 -> 29` |

This changes the adapters-only actor from 62,323 to 227,116 trainable scalars, and the neuron-gains actor from 66,227 to 231,020. The increase is 164,793 trainable scalars in either case. It makes the cross-species interface substantially more expressive, so attribution to the fixed fly-derived recurrent computation becomes weaker unless linear and MLP interfaces are evaluated as matched controls.

### Attribution with MLP interfaces and no gain adaptation

The compact `SimToolRealConnectome1952AdaptersMLPSAPG` profile inherits `weight_mode: adapters_only` and `learn_dynamics: false`: recurrent weights, gains, leak and bias remain fixed. Its three interface MLPs contain 224,797 learned parameters (131,072 sensory, 51,456 descending, 42,269 action readout); the actor total additionally includes its value head, conditioning embeddings and exploration parameters. This count measures learned capacity, not percentage of causal computation.

Source inspection of `_step` and `forward` in `connectome_network_builder.py` confirms no direct robot-observation shortcut to the action-mean MLP: that MLP receives only the 135 motor-cell activities. The input MLPs can nevertheless compute sophisticated features or control-relevant codes, and the output MLP can decode them nonlinearly. A fixed recurrent network can act as a useful transformation/memory channel without its particular biological wiring being uniquely useful. Furthermore, learned inputs can change which dynamical regimes the fixed circuit visits. Therefore necessary routing through the circuit does not establish that native fly motor computations supply the task strategy.

Defensible without further evaluation: the policy uses a fixed fly-derived recurrent circuit with learned nonlinear input/output interfaces. Not established: a percentage of intelligence supplied by the circuit, native dexterity transfer, or an advantage of biological wiring. Test biological specificity by independently retraining matched MLP interfaces around degree-preserving rewired and random frozen graphs, matching populations, edge/sign/scale statistics, optimization and evaluation. Compare against a directly trained MLP controller with a stated capacity and observation/history budget as well. Perturbing circuit state or wiring only at inference tests dependence of an already trained policy, but its distribution shift does not show that a matched replacement could not learn. These are proposed controls, not completed attribution results.

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
- `SimToolRealConnectomeMLPSAPG`: the same 4,310-cell frozen-core profile with 256-unit MLP input/output projections.
- `SimToolRealConnectome1952AdaptersMLPSAPG`: compact 1,952-cell frozen-core profile with 256-unit MLP projections.
- `SimToolRealConnectome1952GainsMLPSAPG`: compact 1,952-cell neuron-gains profile with 256-unit MLP projections.
- `SimToolRealConnectomeGainsDynamicsSAPG`: previous primary behavior with learned gains, leak, and bias.
- `SimToolRealConnectomeGainsSAPG`, `SimToolRealConnectomeLowRankSAPG`, `SimToolRealConnectomeEdgewiseSAPG`: alternative weight adaptations with frozen dynamics.
- `SimToolRealConnectomeFrozenSAPG`: biological topology with gains, leak, and recurrent bias frozen; adapters and heads remain learned.
- `SimToolRealConnectomeRewiredSAPG`: degree-preserving topology control with primary plasticity.
- `SimToolRealConnectomeRandomSAPG`: random sparse topology control with primary plasticity.
- `SimToolRealLSTMAsymmetricSAPG`: unchanged published recurrent baseline.

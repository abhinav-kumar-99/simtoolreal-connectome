# Fixed-input MaleCNS reservoir controller

The fixed-reservoir policy runs the fly-derived circuit during rollout but never differentiates through it. PPO trains only a robot-action readout and the SAPG conditioning table.

Last updated: 2026-09-16

Related: [Connectome actor](connectome-actor.md), [Experiment workflow](../workflows/connectome-experiments.md), [MaleCNS source boundary](../sources/summaries/malecns-front-leg-circuit.md)

## Execution boundary

One network instance owns the fixed MaleCNS parameters and processes the full environment batch. It does not create one parameter copy per environment. Each environment still needs its own recurrent activity because its observation history differs. At 12,288 environments, the live FP32 state is therefore `12288 x 1952`, about 91.5 MiB.

During the 16-step rollout, every control decision applies four recurrent updates under `torch.no_grad()`. The policy stores only the 135 motor-cell activities for that decision. PPO minibatches replay those cached features through the readout; they do not rerun the MaleCNS circuit and do not retain its autograd graph. The feature buffer is `16 x 12288 x 135`, about 101.25 MiB, versus about 1.43 GiB for a same-sized full-state rollout before accounting for four-update autograd intermediates. The reservoir must still run online because its state depends on each trajectory, so it cannot be precomputed once for the whole dataset.

The active restricted-Beta readout concatenates 135 cached motor activities with the selected 32-value SAPG embedding, then applies independent `167 -> 128 -> 29` ELU heads for alpha and beta. This keeps task/exploration conditioning trainable without feeding it into the fly. The nine parameter tensors that receive PPO gradients contain 50,682 scalars: the `6 x 32` SAPG table and both Beta heads. A 168-scalar actor value head exists for model compatibility but receives no loss because `use_experimental_cv: false`; the standard privileged central critic remains authoritative and is trained separately.

This is reservoir computing with source-derived recurrent topology, not evidence that the fixed graph already implements robot dexterity. The fixed input assignments and scales below are engineering choices. A larger subcircuit or whole-CNS artifact can now be explored without PPO activation memory scaling with neuron count, but rollout compute and the single live state per environment still scale with the selected circuit.

## Fixed 144-value input map

The task uses the Rotation-6D policy observation. Actor running normalization is disabled so the configured physical scales have stable meaning. Every scalar passes through `tanh(value / scale)`; `opponent_tanh` writes the positive and negative halves to separate cells. Unlisted target cells receive exactly zero direct drive.

| Robot source | Observation range | Encoding and scale | MaleCNS target |
| --- | --- | --- | --- |
| 29 joint positions | `[0,29)` | signed, 1.0 | proprioceptor slots 0-28 |
| 29 joint velocities | `[29,58)` | opponent, 10.0 | proprioceptor slots 29-86 |
| 29 previous joint targets | `[58,87)` | signed, 1.0 | descending slots 0-28 |
| palm position | `[87,90)` | signed, 0.5 | descending slots 29-31 |
| palm and object Rotation-6D | `[90,102)` | signed, 1.0 | descending slots 32-43 |
| five fingertips relative to palm | `[102,117)` | signed, 0.25 | descending slots 44-58 |
| object keypoints relative to palm | `[117,129)` | signed, 0.25 | descending slots 59-70 |
| object scale | `[141,144)` | signed, 2.0 | descending slots 71-73 |
| object keypoints relative to goal | `[129,141)` | opponent, 0.25 | descending slots 74-97 |

This uses 87 of the 92 selected proprioceptors and 98 of the 157 descending cells. The remaining five proprioceptors, 59 descending cells and all 292 selected tactile cells receive zero direct input. Tactile cells remain in the recurrent graph; the current robot observation simply supplies no actual touch measurement. Source ranges, encoding, scale and target offsets are all YAML-owned and validated for bounds and overlap during model construction.

## Verified execution

The two-epoch GPU-1 smoke suite completed 393,216 frames and reloaded its final checkpoint to produce a finite 29-value deployment action. Its warm second epoch used 1.082 seconds for rollout and 0.267 seconds for the actor update, for 1.349 seconds total and 145,726 frames/s. The stopped structured/BPTT run's nearby steady-state epochs used about 1.17-1.26 seconds for rollout and 0.915-0.925 seconds for the update, for about 2.09-2.19 seconds total. The key intended change is therefore visible in the roughly 3.4-fold actor-update reduction; total throughput also improves, although the two runs use different action distributions and the smoke sample is short.

The reservoir trainer used about 6.2 GiB on physical GPU 1 during initialization/rollout. The comparison is an observed process-memory snapshot, not a controlled peak-memory benchmark. Focused network, configuration and Beta tests total 134 passing tests. The smoke suite's checkpoint verification reports epoch 2, frame 393,216, nine optimizer-state entries and a finite deployment action.

## Boundaries and next experiments

- The first executable version deliberately keeps the validated 1,952-cell MaleCNS graph. Larger subcircuits should be separate artifacts and matched YAML profiles, not silent graph substitutions.
- Four held-input recurrent updates remain the current timing choice. With backpropagation removed, K can be swept higher, but rollout time still grows with K.
- The current continuous tanh rate model is retained for the first controlled comparison. A spiking implementation should be a separate recurrent backend with an explicit timestep, reset, state and output-rate contract.
- Camera-to-fly vision is not implemented here. A later visual profile should use a fixed retinotopic mapping and an appropriate visual/CNS artifact, while keeping goal and nonvisual feedback channels explicit.
- Compare learning curves and task videos against learned structured adapters before attributing any speedup or control quality to the biological circuit.

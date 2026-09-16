# Fixed-input MaleCNS reservoir controller

The fixed-reservoir policy runs the fly-derived circuit during rollout but never differentiates through it. PPO trains only a robot-action readout and the SAPG conditioning table.

Last updated: 2026-09-16

Related: [Connectome actor](connectome-actor.md), [Experiment workflow](../workflows/connectome-experiments.md), [MaleCNS source boundary](../sources/summaries/malecns-front-leg-circuit.md)

## Execution boundary

One network instance owns the fixed MaleCNS parameters and processes the full environment batch. It does not create one parameter copy per environment. Each environment still needs its own recurrent activity because its observation history differs. At 12,288 environments, the live FP32 state is therefore `12288 x 1952`, about 91.5 MiB.

During the 16-step rollout, every control decision applies four recurrent updates under `torch.no_grad()`. The policy stores only the 135 motor-cell activities for that decision. PPO minibatches replay those cached features through the readout; they do not rerun the MaleCNS circuit and do not retain its autograd graph. The feature buffer is `16 x 12288 x 135`, about 101.25 MiB, versus about 1.43 GiB for a same-sized full-state rollout before accounting for four-update autograd intermediates. The reservoir must still run online because its state depends on each trajectory, so it cannot be precomputed once for the whole dataset.

The active clipped-Gaussian readout concatenates 135 cached motor activities with the selected 32-value SAPG embedding, then applies a `167 -> 128 -> 29` ELU action-mean head. Its coefficient-conditioned `6 x 29` log-standard-deviation table and differentiable sigma-three ceiling exactly match the dense-input Gaussian control. The auxiliary actor value head is a separate `167 -> 1` linear layer. Its `c_loss` updates that value head and the shared SAPG embedding, but cannot update the detached motor features or fixed MaleCNS. The privileged central critic remains authoritative for rollout values, GAE and returns and is trained separately under `cval_loss`.

SAPG divides 12,288 environments into six 2,048-environment blocks with entropy-loss coefficients `[.0025,.002,.0015,.001,.0005,0]`. The rollout appends scalar identifiers `[50,40,30,20,10,0]`; the actor uses the nearest identifier to select one row of a learned `6 x 32` table. These numbers are labels, not physical inputs or fixed embedding coordinates. In the fixed-reservoir policy the selected 32-vector enters only the output readout. LF reuse retains cached motor features while relabeling the selected experience as identifier zero, so PPO evaluates the same fixed reservoir output under the leader embedding.

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

The restricted-Beta and matched-Gaussian two-epoch GPU-1 smoke suites each completed 393,216 frames and reloaded their final checkpoints to produce finite 29-value deployment actions. The Gaussian's warm second epoch used 1.182 seconds for rollout and 0.263 seconds for the actor update, for 1.445 seconds total and 136,069 frames/s. Its actor `c_loss` was nonzero (`1.3846` then `.6165`) and the separate `cval_loss` was also nonzero (`.9863` then `.2063`). The stopped structured/BPTT run's nearby steady-state epochs used about 1.17-1.26 seconds for rollout and 0.915-0.925 seconds for the update, for about 2.09-2.19 seconds total. The key intended change is therefore visible in the roughly 3.5-fold actor-update reduction.

The reservoir trainer used about 6.2 GiB on physical GPU 1 during initialization/rollout and grew after optimizer warmup. These are observed process snapshots, not controlled peak-memory benchmarks. Focused network, configuration and distribution tests total 136 passing tests. The Gaussian smoke checkpoint verification reports epoch 2, frame 393,216, eight optimizer-state entries and a finite deployment action.

## Boundaries and next experiments

- The first executable version deliberately keeps the validated 1,952-cell MaleCNS graph. Larger subcircuits should be separate artifacts and matched YAML profiles, not silent graph substitutions.
- Four held-input recurrent updates remain the current timing choice. With backpropagation removed, K can be swept higher, but rollout time still grows with K.
- The current continuous tanh rate model is retained for the first controlled comparison. A spiking implementation should be a separate recurrent backend with an explicit timestep, reset, state and output-rate contract.
- Camera-to-fly vision is not implemented here. A later visual profile should use a fixed retinotopic mapping and an appropriate visual/CNS artifact, while keeping goal and nonvisual feedback channels explicit.
- Compare learning curves and task videos against learned structured adapters before attributing any speedup or control quality to the biological circuit.

## Is K=4 enough?

Four updates are structurally sufficient for fresh values from every currently mapped input class to reach every motor cell that is reachable from that class within the same robot-control decision. They are not yet empirically established as sufficient for useful nonlinear computation.

The recurrent matrix convention is `W[target, source]`, and current input is injected into its target population after the recurrent multiply has read the old hidden state. A directed input-to-motor path with `L` recurrent edges therefore first affects the motor state on update `L+1`. An exact breadth-first search over `biological.npz` gives:

| Fixed source population | Distance 1 | Distance 2 | Distance 3 | Unreachable motor cells | Minimum K for all reachable motors |
| --- | ---: | ---: | ---: | ---: | ---: |
| 87 used proprioceptor cells | 53 | 68 | 9 | 5 | 4 |
| 98 used descending cells | 68 | 61 | 1 | 5 | 4 |
| 24 goal-driven descending cells | 17 | 88 | 25 | 5 | 4 |
| Union of all 185 used input cells | 80 | 50 | 0 | 5 | 3 |

Thus K=4 removes the graph-distance reason to increase K for the current interface, including the goal path. K=8 cannot make the five structurally unreachable motor cells input-dependent; with zero initial state and zero recurrent bias, those five features remain zero. A higher K can still change temporal filtering, recurrent mixing, transient amplification, cancellation and tanh saturation. The retained-leak conversion preserves passive decay over one control interval but does not make K=4 and K=8 equivalent nonlinear systems.

The next timing study should therefore be a matched `K in {1, 2, 3, 4, 8}` ablation, not an assumption that more updates are better. Compare task return/success and throughput, plus current-observation and current-goal sensitivity of the motor feature vector, feature variance/effective rank, step-to-step change, and saturation. K=3 versus K=4 isolates the 25 goal-to-motor distance-three paths especially well; K=4 versus K=8 tests additional recurrent computation after structural coverage is already complete.

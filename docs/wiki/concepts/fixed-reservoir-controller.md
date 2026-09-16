# Fixed-input MaleCNS reservoir controller

The fixed-reservoir policy runs the fly-derived circuit during rollout but never differentiates through it. PPO trains only a robot-action readout and the SAPG conditioning table.

Last updated: 2026-09-16

Related: [Connectome actor](connectome-actor.md), [Experiment workflow](../workflows/connectome-experiments.md), [MaleCNS source boundary](../sources/summaries/malecns-front-leg-circuit.md)

## Execution boundary

One network instance owns the fixed MaleCNS parameters and processes the full environment batch. It does not create one parameter copy per environment. Each environment still needs its own recurrent activity because its observation history differs. For tanh dynamics, the live FP32 state at 12,288 environments is therefore `12288 x 1952`, about 91.5 MiB. The optional LIF dynamics retain membrane, refractory and previous-spike tensors, or three times that recurrent-state storage, while still avoiding rollout autograd activations.

During the 16-step rollout, every control decision applies four recurrent updates under `torch.no_grad()`. The policy stores only the 135 motor-cell activities for that decision. PPO minibatches replay those cached features through the readout; they do not rerun the MaleCNS circuit and do not retain its autograd graph. The feature buffer is `16 x 12288 x 135`, about 101.25 MiB, versus about 1.43 GiB for a same-sized full-state rollout before accounting for four-update autograd intermediates. The reservoir must still run online because its state depends on each trajectory, so it cannot be precomputed once for the whole dataset.

The clipped-Gaussian readout concatenates 135 cached motor activities with the selected 32-value SAPG embedding, then applies a `167 -> 128 -> 29` ELU action-mean head. Its coefficient-conditioned `6 x 29` log-standard-deviation table and differentiable sigma-three ceiling exactly match the dense-input Gaussian control. The auxiliary actor value head is a separate `167 -> 1` linear layer. When `use_experimental_cv: true`, its `c_loss` updates that value head and the shared SAPG embedding, but cannot update the detached motor features or fixed MaleCNS. With `use_experimental_cv: false`, that auxiliary objective is omitted; the action, entropy, PPO, SAPG and central-critic paths remain. In both cases the privileged central critic remains authoritative for rollout values, GAE and returns and is trained separately under `cval_loss`.

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

## Optional hard-spiking LIF dynamics

`dynamics.activation` now accepts `tanh` or `lif`. The LIF option is intentionally restricted to the fixed cached-reservoir path: hard spikes run under `torch.no_grad()`, and PPO differentiates only through the cached motor-rate readout. This prevents a configuration from silently selecting a hard threshold while expecting gradients through learned input adapters or recurrent parameters.

The implementation takes the standard useful pieces from Quantum-Coded Fruitfly's `brain_wordle.py`: persistent membrane voltage, binary spikes, hard reset and refractory state. It does not copy the Wordle calibration. That code multiplies raw connectome weights by an engineering `0.045`, adds external and synaptic voltage increments once per 0.2 ms substep, and exposes only final-substep descending spikes even though it computes window counts. Our artifact instead contains a signed operator normalized to spectral radius one, so its drive scale cannot inherit Wordle's millivolt interpretation.

For each recurrent update, the implemented dimensionless LIF system computes

\[
I = 0.9\,g_{in}\odot W(g_{out}\odot s) + 1.5\,r(x) + b,
\qquad
v' = e^{-\Delta t/\tau_m}v + (1-e^{-\Delta t/\tau_m})I.
\]

A non-refractory cell emits `s'=1` when `v' >= 0.25`, resets its membrane to zero and sets a 2 ms refractory timer; otherwise it emits zero. Four updates divide one 60 Hz robot interval, giving `dt=4.1667 ms`, and `tau_m=10 ms`. Refractory time is decremented in physical milliseconds before threshold evaluation. The fused Triton implementation matches the dense reference on CUDA. A full 1,952-cell inference probe caught that threshold one produced finite internal spikes but silent motor features; the configured `.25` threshold produced nonzero motor rates without changing the tanh job. This is an operating-point smoke check, not a task-performance calibration.

The fixed encoder supplies rates in `[0,1]`. Existing opponent codes are already nonnegative and are unchanged. A signed code `z=tanh(x/scale)` becomes `(z+1)/2`, so zero is a half-rate baseline and negative values suppress rather than require non-biological negative spikes. Unmapped population cells, including tactile cells without measurements, remain exactly zero instead of receiving that baseline. The action readout receives each motor cell's mean binary spike count across all four updates, not merely its last substep.

The YAML-selectable profile is `SimToolRealConnectome1952FixedReservoirRotation6DGaussianSigma3SAPGLIF`. Its important parameters are `activation`, `control_frequency_hz`, `membrane_time_constant_ms`, `spike_threshold`, `refractory_period_ms`, `input_current_scale` and the inherited `neural_updates`. These are engineering operating-point choices, not measured MaleCNS physiology. The LIF full run subsequently passed its smoke gate and ran on GPU 1; it was then stopped at the user's direction for the separate [visual tanh reservoir](visual-reservoir.md).

### Fixed-scale calibration diagnostic

The lack of actor running normalization is intentional, but the current scalar scales are engineering defaults rather than validated calibration. A 12,288-environment raw-observation snapshot from the fixed-reservoir epoch-800 recovery checkpoint showed strongly unequal encoded magnitudes. With the configured maps, joint velocity produced mean absolute encoded activity `.046` and no `|tanh| > .95` samples, while object scale produced mean absolute activity `.496` with `12.1%` above `.95`. Goal-error cells had mean absolute activity `.580` with `2.7%` above `.95`; joint-position and previous-target saturation fractions were `4.8%` and `7.4%`. This is one saved simulator state, not a complete trajectory distribution, but it shows that one scalar scale per feature family can suppress some channels while nearly saturating others.

The recommended no-BPTT refinement is a frozen per-channel or physically grouped calibration followed by the existing bounded population code. Derive it from declared robot/task ranges or from a pinned pre-training observation corpus, store its center/scale and source hash in configuration, and never update it online. Preserve semantic zero for velocities, relative positions and goal error; use fixed reference centers for absolute workspace position and object dimensions. Rotation-6D should retain a common fixed scale rather than data-dependent coordinate whitening if preserving its geometric representation is the goal. Compare the calibrated map against the current map as a fresh matched run; do not alter or resume the current policy under a different transform.

## Verified execution

The restricted-Beta and matched-Gaussian two-epoch GPU-1 smoke suites each completed 393,216 frames and reloaded their final checkpoints to produce finite 29-value deployment actions. The Gaussian's warm second epoch used 1.182 seconds for rollout and 0.263 seconds for the actor update, for 1.445 seconds total and 136,069 frames/s. Its actor `c_loss` was nonzero (`1.3846` then `.6165`) and the separate `cval_loss` was also nonzero (`.9863` then `.2063`). The stopped structured/BPTT run's nearby steady-state epochs used about 1.17-1.26 seconds for rollout and 0.915-0.925 seconds for the update, for about 2.09-2.19 seconds total. The key intended change is therefore visible in the roughly 3.5-fold actor-update reduction.

The reservoir trainer used about 6.2 GiB on physical GPU 1 during initialization/rollout and grew after optimizer warmup. These are observed process snapshots, not controlled peak-memory benchmarks. Focused network, configuration and distribution tests total 136 passing tests. The Gaussian smoke checkpoint verification reports epoch 2, frame 393,216, eight optimizer-state entries and a finite deployment action.

## Boundaries and next experiments

- The first executable version deliberately keeps the validated 1,952-cell MaleCNS graph. Larger subcircuits should be separate artifacts and matched YAML profiles, not silent graph substitutions.
- Four held-input recurrent updates remain the current timing choice. With backpropagation removed, K can be swept higher, but rollout time still grows with K.
- The continuous tanh rate model and hard-spiking LIF model are separate YAML-selected dynamics. The LIF operating point still needs spike-rate, motor-coverage, throughput and task-learning validation before it supports a biological or performance claim.
- The separate [visual reservoir profile](visual-reservoir.md) now implements real L1/L2 column inputs and measured optic-to-motor paths, raw evaluation-angle cameras, retained proprioception and explicit goal channels. The small 1,952-cell profiles themselves have no visual cells.
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

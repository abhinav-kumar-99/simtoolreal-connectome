# Minimal circuits for dexterous manipulation

A read-only structural screen identifies 86-, 262-, and 408-neuron front-leg candidates, while a cross-system literature comparison separates useful small modules from unproven full-task controllers.

Last updated: 2026-09-15

Related: [Existing selection audit](front-leg-pathway-coverage.md), [Current 1,952-cell graph](compact-1952-training.md), [Actor and attribution](../concepts/connectome-actor.md)

## 2026-09-15 training and functional-audit update

The earlier recommendation to try 262 as the first full-task candidate is **weakened** by the [current end-to-end audit](system-audit-2026-09-15.md). The fresh 262 tanh run reached 4.593B reward frames with trailing return 378.30, versus 386.27 for 1,952 tanh at 2.378B, and was slower at matched early frame coordinates. Both remain mostly lifting solutions with almost flat deterministic Task Progress. This does not isolate graph size from every training confound, but provides no observed sample-efficiency advantage for 262.

The two descending ports are both left-sided DNge061 cells. Goal and SAPG features are compressed into two scalar drives, and the checkpoint probe found 99.09% of their preactivations beyond absolute value three. The selected graph retains 15.45% of its cells' observed incoming contacts and 25.38% of motor incoming contacts within the audited VNC scope. These are contact fractions, not fractions of preserved computation. Treat 262 primarily as a local feedback-module experiment until its task-relevant responses and command interface are validated. The structural screen and initial-launch evidence below are historical, not a current completion claim.

## Conclusion and evidence boundary

The smallest concrete candidate justified for testing by this screen is an 86-cell left front-leg tibia sensory-premotor-motor subgraph, not a demonstrated biological or computational minimum. A 262-cell distal-leg version is the preferred first full manipulation experiment because it adds foot/grip-related outputs. These recommendations are hypotheses, not performance results. Smaller reflex/feedback motifs could conceivably help a subtask, but no source or experiment here establishes the minimum sufficient circuit for the full arm-and-hand task.

All three candidates are strict subsets of the existing 1,952 IDs. The structural screen itself performed no training. The 262/3,194 candidate was subsequently materialized and launched as described below; it still has no task-performance result. The study uses the existing 4,778-cell reference universe for exact connectivity calculations, not an exhaustive search over the entire MaleCNS graph. Full public annotations were inventoried for alternative body systems; their complete premotor pathways were not extracted.

## Comparison across major fly systems

These assessments concern fit to the robot task, not rankings established by robot experiments:

- **Local leg feedback:** Best directly relevant candidate for position/movement feedback and coordinated opposing muscle outputs. [Central processing of leg proprioception](https://pmc.ncbi.nlm.nih.gov/articles/PMC7752136/) experimentally links particular 13B-alpha/9A-alpha populations to postural responses; [divergent sensory pathways](https://www.nature.com/articles/s41467-025-59302-3) separates local leg-control routes from vibration routes. Neither identifies every MaleCNS cell in this screen physiologically.
- **Proboscis reaching:** Strong alternative for compact directed reaching and contact response. [McKellar et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC7316511/) maps primary motor neurons and their movement contributions. A small set of motor outputs is not a census of the complete sensory/premotor controller, and reaching is not independent multi-finger manipulation. Worth a separate full-connectivity extraction, not an assumed smaller replacement.
- **Wing steering and haltere feedback:** Fine control with compact motor outputs, but flight-specific mechanics. [Leg/wing premotor architecture](https://www.nature.com/articles/s41586-024-07600-z) finds modular organization and distinct recruitment patterns. This motivates an alternative control module, not assuming flight precision transfers to the hand or is captured by current rate dynamics.
- **Antenna/head positioning:** Potential small orienting or stabilization modules, less directly matched to multi-contact manipulation. [Active antennal movements](https://pmc.ncbi.nlm.nih.gov/articles/PMC9992063/) provides an example of active positioning and sensory integration.
- **Central-complex navigation:** A useful heading-to-goal steering computation, but not inherently a hand controller. [Goal-oriented steering](https://www.nature.com/articles/s41586-024-07039-2) motivates an orientation-error module if that is the chosen subtask.
- **Mushroom-body learning:** A candidate associative-learning system, not a direct replacement for detailed motor coordination; [dopamine/memory interactions](https://www.nature.com/articles/s41586-024-07819-w) studies this learning role. Visual, olfactory, feeding-state, wingbeat-power and abdominal systems are not automatic inclusions when robot observations and the task are already supplied externally.

The local `annotations.feather` inventory includes front/middle/hind-leg motor subclass counts 135/116/130, wing 67, haltere 16, and central-brain proboscis subclass `pm` 67. These raw motor-label counts do not rank complete controller sizes or coverage. They are not used as neuron quotas.

## Exact read-only selection procedure

Inputs are the pinned `profiles/connectome/front_leg_coverage/candidate_neurons.csv` and `reference_incoming_edges.csv.gz`, with hashes recorded in `configs/connectome/malecns_1952.yaml`. Keep endpoints in the reference neuron inventory and directed pairs with `synapses >= 5`. Use `rootSide == L` for sensory inputs and `somaSide == L` for motor outputs. These are annotation-based screens, not proof of projection-side physiology; intermediate cells are not artificially forced to the same side.

For input set S and motor set M, choose every intermediary X satisfying an observed `S -> X -> M` path, excluding S/M from the intermediary count. Start with `K = S union M union X`; retain all induced directed edges, not just the two-edge motifs. Restrict K to the intersection of forward reachability from S and backward reachability from M. In these three cases, all removed nodes are isolates. No shortest-path tie-break, degree top-k or target neuron count is used.

- **Tibia:** S is the front-proprioceptor mask restricted to left-root chordotonal-organ annotations. M has stripped type names `Ti flexor MN`, `Acc. ti flexor MN`, or `Ti extensor MN` in left front motors.
- **Distal leg:** S is all left-root front-proprioceptor annotations. M additionally includes `Ta depressor MN`, `Ta levator MN`, `ltm MN`, `ltm1-tibia MN`, and `ltm2-femur MN`.
- **Whole-leg screen:** Same sensory set as distal; M is every left-soma front motor. This name describes the motor selection, not complete circuit reconstruction.

| Candidate | Before route pruning | Removed | Retained sensory | Retained motors | Other/intermediary cells | Total | Directed edges |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Tibia | 95 | 9 | 14 | 17 | 55 | 86 | 616 |
| Distal leg | 272 | 10 | 48 | 31 | 183 | 262 | 3,194 |
| Whole-leg screen | 417 | 9 | 50 | 66 | 292 | 408 | 6,939 |

The 55 intermediary cells in the 86 set are 52 VNC-intrinsic, two ascending and one descending neuron. Its largest strongly connected component contains 52 neurons, with 56 total in components larger than one. Corresponding largest/total recurrent-component counts are 190/194 for 262, and 296/300 for 408. Thus these are recurrent subgraphs, not only chains, but recurrence is not demonstrated useful memory or native motor functionality.

Sorted little-endian int64 body-ID SHA256 identifiers:

- 86: `524d752d449d229db9dbde39d7585c2bf55afdd42d96346d50a25a4d7255b185`
- 262: `e29e16914c7ba14c4839621c4c667cf9bd7eeaf4bba9cef899f84690ca183097`
- 408: `aca97454af47ac0046da0a5773dd7a5071623722cb56f866f67699435b6da7a2`

## What trimming excludes and what still needs design

This removes the obligation to keep both legs, all 135 motor outputs, whole 13A/13B/9A/23B families, all 292 tactile seeds, or all 157 descending input seeds. Membership follows the chosen proprioceptive-to-motor routes; cells of any other class remain eligible if they meet that rule. This can discard longer pathways, tactile integration and modulatory feedback. Missing tactile channels in current robot observations motivate testing a proprioception-focused circuit, but do not prove tactile-labelled neurons are computationally useless.

Goal routing is not settled by node counts. The 86 set retains no current selected descending-adapter port, despite containing one other descending neuron. The 262 set retains two current descending ports (DNge061 bodies 220203 and 909558); the 408 set retains those plus DNge068 body 16079. Goal information must be explicitly mapped to retained supported ports, additional justified command cells, or a deliberately changed premotor/sensory interface. The 86 set is therefore not a drop-in training config.

The five-contact filter remains a sensitivity variable, not biological ground truth. Restoring all >=1-contact pairs *among the same retained IDs* yields 1,425 / 9,483 / 20,273 edges without increasing the 86 / 262 / 408 neuron counts. In contrast, redoing the entire intermediary selection at threshold one expands the initial sets to 322 / 759 / 965 before route pruning. Do not conflate those operations. Also, the right-side tibia screen produces only 36 nodes before pruning but 15 isolates, versus left 95/9; that asymmetry must not be interpreted as proof the right circuit needs fewer neurons. Reconstruction, annotation and filtering differences require review.

Other unresolved choices include keeping the parent graph's weight normalization versus rescaling the smaller operator, preserving required delayed/inhibitory feedback, and matching input/output MLP capacity. With a nonlinear readout, 17 motor features can map to 29 action numbers, but this does not ensure sufficient independently variable control. A tiny core plus a large MLP also weakens biological attribution. No mathematical neuron-count lower bound follows from the number of robot actuators alone.

## Five-contact threshold and resource comparison

The cutoff counts anatomical contacts per directed neuron pair: a pair with four contacts is omitted, while one with five is retained with its count-derived weight. It is inherited from the reference extraction, not fitted to robot performance. The [reference CPG study](https://pmc.ncbi.nlm.nih.gov/articles/PMC13142387/) describes using a minimum count of five for mCNS, but not for its BANC/FANC matrices. [Distributed control circuits across a brain-and-cord connectome](https://www.nature.com/articles/s41586-026-10735-w) also describes filtering connections below five to reduce reconstruction or biological noise. This supports a conservative filtering convention, not a biological boundary or proof that four-contact connections are unimportant.

For the fixed 262 IDs, threshold five retains 3,194 pairs representing 69,216 contacts; threshold one retains 9,483 pairs representing 80,324 contacts. Thus dropping 66.3% of pairs removes only 13.8% of recorded contact count. Contact count is not a measured fraction of functional importance. Threshold one still obeys the reference audit's confidence and VNC-region filters; it is not every synapse in the full CNS. Rerunning neuron selection at threshold one would be a different experiment.

Noise interpretation: these are uncertain anatomical edges, not freshly sampled numerical noise on each forward pass. The CPG paper explicitly motivates its floor by removing very weak connections and neurons connected only weakly; this is not an edge-by-edge error calibration. The separate BANC influence analysis explicitly cites reconstruction or biological noise. [Descending-network analysis](https://www.nature.com/articles/s41586-024-07523-9) uses the same five-contact convention as the FlyWire explorer, but does not establish that every retained edge is correct or every omitted edge false. Multiple contacts provide stronger support than an isolated detection, but correlated reconstruction errors and genuine weak connections limit that inference. A frozen erroneous edge would remain a fixed coupling; its downstream effect could be amplified or suppressed by recurrent dynamics.

Given the user's explicit priority to avoid uncertain wiring, the recommended initial experiment is now the 262/3,194 variant, superseding the earlier conversational preference to restore all 9,483 pairs for completeness. Keep the existing confidence filters and consider specific weak-edge exceptions only after anatomical review, corroboration across corresponding circuits, or controlled functional tests. This is a conservative experiment choice, not a demonstrated clean graph, optimal threshold, or authorized runtime change.

Read-only snapshot on 2026-09-14: the active clipped-Gaussian KL-0.004 and tanh-Gaussian KL-0.016 jobs both use the 1,952/33,720 graph with frozen core, 256-hidden-unit MLP adapters, `triton_fused`, 12,288 environments, horizon/sequence length 16, physical training microbatch 49,152, FP32 core and AMP PPO. Main trainer GPU allocations reported by NVIDIA were approximately 13,023 and 12,550 MiB respectively; these are live allocations, not controlled peak measurements.

The following are shape-derived component sizes, not measurements of a launched 262 policy. Both candidates retain 48 sensory, two existing descending and 31 motor ports, with the same robot observation split and MLP hidden width:

| Component | Current 1,952 | 262 / 9,483 edges | 262 / 3,194 edges |
| --- | ---: | ---: | ---: |
| Directed edges | 33,720 | 9,483 | 3,194 |
| One live FP32 recurrent state, MiB | 91.50 | 12.28 | 12.28 |
| Full 16-step rollout state buffer, MiB | 1,464.0 | 196.5 | 196.5 |
| Three core saved tensors over 49,152 training samples, MiB | 1,098.0 | 147.375 | 147.375 |
| Learned input/output MLP parameters | 224,797 | 72,477 | 72,477 |
| Forward CSR row chunks of 32 edges | 2,161 | 433 | 252 |
| Transpose/backward CSR row chunks of 32 edges | 2,144 | 429 | 251 |

Memory formulas follow `rl_games/rl_games/common/a2c_common.py::play_steps` and `rl_games/rl_games/algos_torch/connectome_triton.py::_Step`: live state is `B*N*4` bytes; rollout storage adds horizon 16; three saved core tensors are hidden input, recurrence and preactivation. These rows are separate components with different lifetimes and must not be blindly added into a total GPU peak. The rollout extras are replaced before PPO training, although the CUDA allocator can retain released storage. Additional activations, gradients, batch copies and workspaces are excluded.

Both alternatives reduce neuron-sized buffers by 86.6%. Frozen sparse graph storage is small compared with these batched tensors, and no recurrent-edge parameter gradient is needed. Restoring weak pairs therefore does not triple state memory. Edge visits fall to 28.1% or 9.5% of current, but this is not a wall-clock forecast: the implementation uses per-neuron programs, padded 32-edge chunks, dense adapters and additional backward work. Between the two small variants, actual edges differ by 2.97x but forward/backward chunk counts by about 1.71x, neither a measured runtime ratio. Physics, the privileged critic and general PPO overhead remain. Existing environment-step timing lacks GPU synchronization and cannot establish a clean simulator fraction or speedup bound. The launched 262 policies provide early throughput and allocation observations, but time to a fixed score remains unmeasured.

## Implemented 262 experiment

On 2026-09-14, `configs/connectome/malecns_262_distal_leg.yaml` materialized the exact 262/3,194 graph. Its artifact SHA256 is `6f07c89d70482fa6a2c93c9b61a20755b215393913493837f7cd210af882d08b`; it exposes 48 sensory, two descending-command and 31 motor ports. A matched full-geometry smoke completed two epochs and 393,216 frames for both policies with finite deployment reload actions.

Two fresh-from-frame-zero, 100B-cap adapters-only MLP runs are active: ordinary clipped Gaussian with actor KL target 0.004 on physical GPU 0, and tanh-squashed Gaussian with actor KL target 0.016 on physical GPU 1. Both retain 12,288 environments, horizon/sequence 16, 49,152 actor and critic minibatches, old update timing, seed 42 and 250M-frame inference checkpoints/videos. TensorBoard inspection after launch found 171 populated scalar tags in each fresh event stream, through steps 10,027,008 and 9,830,400 respectively. Both run directories are linked under the tree watched by the existing port-6008 TensorBoard, and its `/data/runs` endpoint resolves both names. An earlier pair reached about 15M frames before being stopped during a mistaken attempt to restore unrelated 1,952 jobs; those artifacts are preserved but are not the active comparison. The active pair intentionally restarted from zero rather than resuming them.

## Acceptance criteria

First test tracking in both directions and recovery from perturbations, then contact retention, coordinated finger motion and the actual object-manipulation objective on held-out conditions. Compare equally trained biological and rewired/random frozen graphs with matched interfaces, graph statistics and training budgets. Report performance versus frames and wall time. Do not accept a count because it is small, a graph because it is connected, or a controller because it only generates movement. Further pruning should follow measured loss of useful behavior and interpretable pathway coverage, not a size quota.

# Minimal circuits for dexterous manipulation

A read-only structural screen identifies 86-, 262-, and 408-neuron front-leg candidates, while a cross-system literature comparison separates useful small modules from unproven full-task controllers.

Last updated: 2026-09-14

Related: [Existing selection audit](front-leg-pathway-coverage.md), [Current 1,952-cell graph](compact-1952-training.md), [Actor and attribution](../concepts/connectome-actor.md)

## Conclusion and evidence boundary

The smallest concrete candidate justified for testing by this screen is an 86-cell left front-leg tibia sensory-premotor-motor subgraph, not a demonstrated biological or computational minimum. A 262-cell distal-leg version is the preferred first full manipulation experiment because it adds foot/grip-related outputs. These recommendations are hypotheses, not performance results. Smaller reflex/feedback motifs could conceivably help a subtask, but no source or experiment here establishes the minimum sufficient circuit for the full arm-and-hand task.

All three candidates are strict subsets of the existing 1,952 IDs. No new runtime graph, YAML training profile, checkpoint conversion, pruning of the active artifact, or training launch was performed. The study uses the existing 4,778-cell reference universe for exact connectivity calculations, not an exhaustive search over the entire MaleCNS graph. Full public annotations were inventoried for alternative body systems; their complete premotor pathways were not extracted.

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

## Acceptance criteria

First test tracking in both directions and recovery from perturbations, then contact retention, coordinated finger motion and the actual object-manipulation objective on held-out conditions. Compare equally trained biological and rewired/random frozen graphs with matched interfaces, graph statistics and training budgets. Report performance versus frames and wall time. Do not accept a count because it is small, a graph because it is connected, or a controller because it only generates movement. Further pruning should follow measured loss of useful behavior and interpretable pathway coverage, not a size quota.

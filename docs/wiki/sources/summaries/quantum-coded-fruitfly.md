# Quantum-Coded Fruitfly: sensory routing and dopamine audit

The Wordle project offers a useful fixed population-encoding idea, but its dopamine pulse does not drive learning and a matched zero-neural-input check retains its reported win rate.

Last updated: 2026-09-16

Related: [Actor](../../concepts/connectome-actor.md), [Dopamine learning](../../analyses/dopamine-inspired-learning.md), [Eligibility trainer](../../workflows/eligibility-training.md), [Front-leg coverage](../../analyses/front-leg-pathway-coverage.md)

## Source and scope

Inspected [Quantum-Coded/fruitfly](https://github.com/Quantum-Coded/fruitfly/tree/e1a2089a1dcc312df446379ca9cd30193575f4db) at master commit `e1a2089a1dcc312df446379ca9cd30193575f4db` on 2026-09-16. Read the encoder, extraction, simulator, readout, trainer, candidate filter, benchmark and relevant server paths; inspected the shipped numeric artifact/checkpoint and ran a CPU counterfactual evaluation. No robot training settings or processes changed.

All source paths below are under `wordle/backend/` at that pinned revision. Artifact SHA-256: `4cc5861e70c0519a4fc10cf2f19376ec8d5e79903e4313219594ed48e94363e7`; readout checkpoint SHA-256: `4538f0987e4511d6bc9875339f49c505b85fe2e45b05c1779625e3a151c3f6df`.

## Actual sensory mapping

The README describes a learned 31-to-13 projection. The executed [encoder](https://github.com/Quantum-Coded/fruitfly/blob/e1a2089a1dcc312df446379ca9cd30193575f4db/wordle/backend/encoding.py) instead uses fixed masks/scales and has no trainable parameters. Only readout parameters enter Adam.

- `extract_circuit_wordle.py` sorts cell types containing `ORN`, takes the first 26, and assigns them to A-Z. This is a designed symbol-to-population mapping, not measured letter/odor correspondence.
- Letter populations with fewer than 35 cells are supplemented by seeded random selections from the ORN pool. Extras can overlap the original population; observed final masks contain 36--68 cells.
- Visual cells are identified by superclass or a cell-type string containing `PN`, then split into five X-coordinate quantiles. These bins are not verified functional retinotopy.
- Each letter occurrence adds 160 Hz to its mask; feedback adds 220/140/35 Hz to the corresponding spatial sector. Summed rates are scaled by `0.012`. The letter-position loop variable is unused. Verified that `ARISE` and `RAISE` with identical feedback vectors produce identical drives; the training representation loses letter-position binding.

The useful principle is sparse modality-specific population coding, optionally with small learned calibration. Alphabetical assignments, coordinate bins, random augmentation and drive amplitudes are engineering choices.

## Dopamine is not in the learning path

In [train.py](https://github.com/Quantum-Coded/fruitfly/blob/e1a2089a1dcc312df446379ca9cd30193575f4db/wordle/backend/train.py), a completed game yields reward `1 + 0.25*(6-guesses)` for a win or `-1` for failure. A positive-only drive stimulates the dopamine mask and a random mushroom-body subset for 25 substeps. Then Adam updates only the readout using `loss = -reward * sum(log_probs)`.

Dopamine spikes/telemetry do not enter the loss, an eligibility trace, any synaptic update or a later decision in the episode. The next episode resets brain state. There is no persistent plasticity state. Removing the pulse preserves the already collected trajectory's loss and gradients; its random mask consumes global RNG, so exact full-run reproducibility would also require matching RNG consumption. The browser's dopamine phase also has no optimizer update. This implements external readout REINFORCE plus reward visualization. The README acknowledges biological plasticity as future work despite stronger docstring wording.

Our existing `connectome_eligibility.py::observe/update` does apply TD error times parameter-credit traces to adapters/readout/optional gains. That is a causal reward-modulated update, although it uses approximate spatial feedback and has no biological dopamine machinery.

## Provenance and architecture discrepancies

The shipped NPZ has 3,127 cells, 179,516 directed weighted entries, 748 selected ORNs, 406 visual-mask cells, 149 MB-mask cells, 13 dopamine-mask cells and 13 descending/motor-mask cells. These differ from the README's approximate 1,800 KCs, 60 DANs and five outputs.

The [readout](https://github.com/Quantum-Coded/fruitfly/blob/e1a2089a1dcc312df446379ca9cd30193575f4db/wordle/backend/readout.py) is `45 -> 64 -> 32 -> 1`: 13 neural features plus 32 direct candidate-word features. `brain_wordle.py::get_descending_rates` returns final-substep binary spikes, not integrated firing rates. Training/benchmark use 20 substeps at 0.2 ms, or 4 ms per decision.

The extractor labels an existing `../../backend/data/circuit.npz` that is absent from this repository; it does not implement the README's whole-connectome BFS. The README mixes MaleCNS and FlyWire names. The available script does not establish the full claimed upstream provenance; that alone does not show the shipped graph is non-biological.

`candidate_filter.py` maintains all feedback history, enforces positional/count constraints and ranks candidates by handcrafted letter-frequency/diversity scores. The readout also receives candidate features outside the connectome. The reported random baseline lacks that same constraint filter. Benchmark words come from the same answer list as training without an explicit excluded training split.

## Matched CPU check

Used the shipped checkpoint, Torch 2.4.0, one CPU thread, `random.seed(42)` and `random.sample(answers,100)`, matching `benchmark.py`. Saved Python RNG state immediately after cohort sampling and restored it before each condition to match opener shuffles. All conditions used the same 100 words, six-guess cap and `top_k=50`. No retraining occurred.

Numeric NPZ members were read with `allow_pickle=False`; unused cell-type/superclass metadata were replaced with placeholder strings for simulator construction. Numerical data and upstream computation were unchanged. Checkpoint loading used `weights_only=True`; no upstream checkpoint/result was overwritten.

| Condition | Wins / 100 | Mean guesses, failures counted as six |
| --- | ---: | ---: |
| Neural simulation and trained readout | 99 | 4.15 |
| All 13 neural features zeroed; same readout/filter | 99 | 4.13 |
| First filtered candidate; no brain/readout | 98 | 3.87 |

The committed result reports 99 wins/4.12 guesses; our CPU reproduction differs slightly in mean guesses. Neural features were nonzero on 45/415 normal decisions. On those same candidate sets, zeroing features changed greedy choice five times and changed probabilities by at most `0.006392777`. Normal and zero-feature closed-loop conditions produced identical full guess sequences on 95/100 targets. No empty-candidate secret-word fallback was used.

This is evidence of weak dependence on neural input for this checkpoint/cohort, not proof about all targets or training runs. The 99% result does not establish dopamine learning or a biological-topology benefit. The filter-first control has a win-rate/guess-count tradeoff; this small test does not establish a universal winner.

## What to test in our method

Our current dense projection mixes 128 robot features across 384 sensory cells; 12 goal features plus the SAPG embedding separately drive descending cells. The compact sensory group contains 92 proprioceptor candidates and 292 tactile candidates, but exact tuning is unresolved for many body IDs and our actor has no explicit tactile/contact observations.

The implemented first test uses annotation-based population masks with learned group-local linear projections rather than fixed tuning codes. It keeps robot joint/group correspondence explicit, routes context separately, leaves tactile-labelled cells without fabricated direct touch input, and converts palm/object orientations to 6D rotation-matrix columns. Fixed position/opponent-velocity codes remain a later ablation. Position, directional motion and vibration-sensitive FeCO populations have physiological support ([Mamiya et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC6481666/)), but broad sensory annotations do not identify exact claw/hook tuning. Adding contact channels is a separate observation/deployment change.

Calibrate drive amplitude and baseline activity using sensory saturation, motor sensitivity and propagation-delay diagnostics. Compare dense adapters, grouped learned adapters and fixed codes plus calibration under matched graph, PPO/SAPG, observations, critic and auxiliary-value settings. Include shuffled population assignments with the same sparsity/parameter count and independently test degree-preserving graph rewiring. Fixed masks and positive currents alone do not guarantee useful dynamics.

The all-cell auxiliary value versus motor-only action discrepancy remains an independent factor. A motor-feature auxiliary head tests supervision of the action representation, but may worsen value prediction; do not combine that change with input coding and attribute the result to one mechanism.

A genuine dopamine-inspired extension needs persistent plasticity, eligible parameters, a teaching signal and measurable later behavioral change. Schematically `Delta w_ij = eta * m_c(t) * e_ij(t)`, with compartment-specific modulation `m_c` and explicit eligibility/action credit `e`. Generic coactivity does not supply action credit automatically. [Aso and Rubin](https://elifesciences.org/articles/16135) demonstrate compartment- and timing-dependent learning effects; a global positive pulse is not a sufficient biological rule.

The current compact circuit has no dopamine consensus labels (53 unclear and four missing labels remain), and no implemented mushroom-body learning module. Anatomical DAN/KC/MBON extensions need circuit extraction, compartment targets and justified dynamics/plasticity beyond the current graph. A smaller engineering experiment could combine slow PPO training with bounded fast adapter/readout plasticity, but must specify reward availability, episode resets, evaluation adaptation and likelihood replay under PPO. A modulatory network trained by PPO is also possible; it becomes a plasticity mechanism only if it changes persistent learning state. Adding labelled neurons does not resolve our existing eligibility trainer's approximate action-credit problem.

Prioritize structured sensory coding and causal readout checks before another large plasticity experiment. These proposals have not been implemented or validated for robot performance.

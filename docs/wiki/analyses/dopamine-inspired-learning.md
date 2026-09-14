# Dopamine-inspired online learning

Reward-modulated eligibility traces are a plausible alternative learning rule, but the current connectome actor has no native dopamine plasticity and lower update cost does not guarantee faster task learning.

Last updated: 2026-09-13

Related: [Adaptation mechanisms](../concepts/connectome-adaptation.md), [Approved compact run](compact-1952-training.md), [Actor](../concepts/connectome-actor.md)

## Biological evidence and boundary

In fly mushroom-body learning, dopaminergic signals modulate plasticity at specific circuit sites; dopamine is not a uniform command to increase every active connection. Different pathways mediate appetitive and aversive reinforcement, and timing matters. [Felsenberg et al., Nature 2017](https://www.nature.com/articles/nature21716) demonstrate reward-memory re-evaluation involving distinct reinforcing dopamine populations and describe reward-associated depression of Kenyon-cell output to avoidance pathways. [Ueno et al., eLife 2017](https://pmc.ncbi.nlm.nih.gov/articles/PMC5262376/) examine coincident activity, dopamine release and mushroom-body plasticity. These are biological associative-learning results, not demonstrations of cheap dexterous robot learning.

[Bellec et al., Nature Communications 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC7367848/) derive eligibility propagation (e-prop): forward-computed local traces combined with learning signals approximate recurrent-network learning without backpropagation through time. Reward-based versions exist. Approximate neuron-specific credit signals are important; simply multiplying arbitrary Hebbian coactivity by positive reward is not equivalent to a policy gradient. The cited work concerns spiking networks (with additional derivations for other recurrent models), not this exact sparse tanh/gain/SAPG actor, and does not establish better sample efficiency here.

## Current implementation

`rl_games/rl_games/algos_torch/connectome_network_builder.py::_step` computes adapter drives, fixed signed recurrence with bounded learned gains, leak and tanh. It has no dopamine concentration, receptor dynamics, eligibility state, or online synaptic update. Parameter changes come from the PPO optimizer between rollouts. The compact manifest has no dopamine-labelled cells, although 53 labels are unclear and four missing; absence of a label alone is not proof about every cell's physiology. More importantly, chemical learning machinery and a mushroom-body learning circuit were not implemented. Adding a neuron labelled dopamine or stimulating current cells would not by itself enable learning.

The actor is anatomy-informed wiring with simplified dynamics and robot adapters, not a pretrained biological dexterity program with an intact biochemical learning system. A synthetic reward-modulated update does not require adding dopamine neurons or expanding the graph, but must be described as an engineered learning rule.

## Proposed first experiment, not implemented

Warm-start from an explicitly selected checkpoint; freeze recurrence, gains, input adapters, embeddings and normalization statistics for the first comparison. Train only the motor-to-action readout (135 x 29 weights plus 29 biases, 3,944 parameters) and a lightweight value predictor. Frozen input adapters are important: merely freezing the base recurrent weights does not eliminate recurrent backpropagation while input adapters still learn.

For a stochastic policy, let the teaching signal be the engineered TD error `delta_t = r_t + gamma * V(h_next) - V(h_t)`, with correct terminal handling. Maintain a fading eligibility trace based on the sampled action's log-probability gradient with respect to readout parameters, not unqualified neuron coactivity. Conceptually, `e_t = gamma * lambda * e_(t-1) + grad_theta log pi_theta(a_t | h_t)` and `theta += eta * delta_t * e_t`. The reward-prediction-error construction is an algorithmic dopamine analogy, not a claim that all fly dopamine neurons compute this equation. Signed errors mean better/worse than expected, not just larger reward equals more dopamine forever.

Because the recurrent feature generator is fixed, readout gradients require no backpropagation through recurrent state. Use on-policy exploration and small bounded updates; reset per-environment traces at episode boundaries and aggregate reward-times-trace products across environments (not a product of separately averaged rewards and traces). Existing mixed-exploration SAPG data cannot be reused indiscriminately without its policy/off-policy semantics. Value learning, normalization, log-standard-deviation adaptation and termination masks must be made explicit in a future YAML contract.

A later, riskier variant could update gains or adapters through derived rate-network eligibility traces and approximate feedback signals. Keep support/sign/gain constraints and benchmark stability. This is substantially more implementation work than changing a reward coefficient. Direct edge plasticity is not required for the first experiment. At 12,288 environments, even one FP32 trace per 33,720 edges occupies about 1.54 GiB before adapter traces and other state; local learning is not automatically cheap on a GPU.

## Measured cost constraint

Read-only inspection of the running compact job's `train.log`, with 1,607 timing records available at inspection, found these means over its last 100 phases:

| Timed region | Seconds per phase |
| --- | ---: |
| Rollout collection, including policy inference | 2.40101 |
| PPO update region | 0.55143 |
| Total timed epoch | 2.95247 |

The regions are defined in `rl_games/rl_games/common/a2c_common.py` around `update_time_start`, `update_time_end` and the timing printout. This is a time-specific snapshot, including reset-dependent variation, not an isolated benchmark. Even eliminating the entire measured update region would save only about 18.7% of this timed epoch at unchanged sample count and rollout cost (approximately 1.23x throughput). A practical replacement still costs something and total wall time also includes work outside these regions. Much larger speedups require fewer interactions or faster rollout collection, not just removing BPTT.

Evaluate any future experiment against PPO using both equal environment-frame budgets and equal wall time, with matched initial checkpoint, seeds and task distribution, and frozen-policy mean-action evaluation. Compare fine tracking, perturbation recovery and contact/multijoint coordination, not merely reward or seconds per update. Readout-only learning restricts capacity and may learn more slowly or plateau. Neither dopamine-inspired rules nor e-prop currently justify a claim of reduced time-to-dexterity on this task. This discussion does not authorize replacing or stopping the ongoing runs.

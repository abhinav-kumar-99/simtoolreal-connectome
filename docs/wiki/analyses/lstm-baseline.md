# Parameter-matched LSTM baseline

The compact fly control is a standard one-update LSTM matched to the current all-neuron actor's instantiated coefficient count.

Last updated: 2026-09-18

Related: [Connectome actor](../concepts/connectome-actor.md), [Experiment workflow](../workflows/connectome-experiments.md), [Compact training](compact-1952-training.md)

## Fairness contract

The 1,952-cell graph has 33,720 active recurrent edge weights. Its
`1,952^2 = 3,810,304` possible dense pairs are not a parameter count: an absent
edge is neither stored nor multiplied. Matching an LSTM to the dense possibility
count would therefore reward zeros that encode graph structure but consume no
model capacity.

The primary comparison instead counts every instantiated scalar coefficient
used by the actor, while excluding structural integer indices:

| Actor component | Fly | LSTM |
| --- | ---: | ---: |
| Trainable interface, policy and auxiliary-value parameters | 692,268 | 733,790 |
| Frozen gain, leak and recurrent-bias vectors | 7,808 | 0 |
| Fixed recurrent edge values | 33,720 | 0 |
| **Instantiated actor coefficients** | **733,796** | **733,790** |

The LSTM has six fewer total coefficients, but every LSTM coefficient is
trainable. It is therefore a deliberately strong learned alternative rather
than an equal trainable-degrees-of-freedom control. The unchanged 2,037,769
parameter privileged central critic is excluded from both sides of the match.

## Architecture

`SimToolRealLSTM323MatchedGaussianSigma3SAPG` consumes the same 140 policy
observations and replaces the scalar SAPG ID with the same learned 32-value
embedding, giving a 172-value LSTM input. It uses one 323-unit LSTM layer, one
recurrent update per environment step, LayerNorm, and a shared 256-unit ELU
trunk feeding the 29-action mean and auxiliary scalar value heads. Sequence
length remains 16, so the model performs ordinary 16-step recurrent BPTT.

The LSTM does not repeat an observation to imitate the fly's four sparse graph
propagation updates. A dense LSTM transition already mixes every hidden unit in
one update; repeating it would create a nonstandard deliberation architecture.
Runtime, memory and throughput are reported separately instead.

The actor retains the fly run's coefficient-conditioned six-row Gaussian scale,
smooth sigma-three ceiling, small action-head initialization, input
normalization, SAPG/LF objective and privileged central critic. The selected
replacement sets `use_experimental_cv: false`, so the actor-side value head is
not optimized; the privileged central critic still supplies rollout values,
GAE and its own value loss. The generic actor's optional
`space.continuous.max_sigma` key is backward compatible when omitted.

The actor's post-LSTM MLP has one 256-unit hidden layer, matching the fly
actor's learned interface width. The privileged critic retains its inherited
`[1024, 1024, 512, 512]` MLP so that critic capacity remains identical across
the fly-versus-LSTM actor comparison.

## YAML-driven workflow

Audit parameter counts and compute without starting Isaac Gym training:

```bash
.venv/bin/python scripts/profile_connectome_actors.py \
  --config configs/connectome/profiling/compact_fly_lstm_matched.yaml
```

The profiling YAML owns device, AMP, seed, rollout/training tensor shapes and
the expected 733,796/733,790 counts. The helper reports trainable and frozen
parameters, fixed edge values, excluded index buffers, hypothetical dense size,
latency, throughput and memory.

The smoke gate remains available but was not launched for the live handoff. The
production command below is the contract used by the active trainer:

```bash
# Two-epoch integration gate.
.venv/bin/python scripts/run_connectome_suite.py \
  --config configs/connectome/suites/ppo_lstm323_matched_gaussian_lf_entropy1x_sigma3_smoke.yaml

# Seed-42 production contract used by the live GPU-1 replacement.
.venv/bin/python scripts/run_connectome_suite.py \
  --config configs/connectome/suites/ppo_lstm323_matched_gaussian_lf_entropy1x_sigma3_100b.yaml
```

The suite YAMLs own GPU, seed, environment/batch geometry, frame cap, checkpoint
cadence, PPO/SAPG/LF settings, auxiliary value-loss choice, reward and
perturbations. The train profile owns only the LSTM architecture, initialization
and sigma ceiling. The suite helper owns process launch, resolved configuration,
checkpoint validation and artifact paths.

## Live replacement

On 2026-09-18, the physical-GPU-1 all-neuron fly job with
`use_experimental_cv: true` and its milestone watcher were stopped with all
artifacts preserved. Its final printed frame count was 9,642,442,752. The
physical-GPU-0 all-neuron fly job with `use_experimental_cv: false` was not
changed.

The 323-unit matched LSTM production suite then replaced the stopped GPU-1 job
from a fresh seed-42 initialization. Its resolved contract has actor MLP units
`[256]`, one LSTM transition per environment step, sequence length 16,
`use_experimental_cv: false`, and the unchanged privileged critic. Initial
telemetry through frame 5,505,024 was finite: aggregate entropy 41.1863, KL
`.004638`, actor loss `-.002376`, actor-side value loss exactly zero, privileged
critic loss `.083253`, and zero invalid-KL flags in both mini-epochs. This is an
integration and liveness check, not learning evidence.

The live tmux sessions are `connectome-lstm323-matched-noaux` and
`connectome-lstm323-matched-noaux-eval`. Artifacts are rooted under
`train_dir/connectome/lstm_baseline/ppo_lstm323_matched_gaussian_lf_entropy1x_sigma3_100b`;
the production and watcher logs are under
`profiles/connectome/lstm323_matched_noaux/`.

The live sentinel milestone watcher uses:

```bash
.venv/bin/python scripts/run_connectome_milestone_evaluation.py \
  --config configs/connectome/evaluation/ppo_lstm323_matched_gaussian_lf_entropy1x_sigma3_100b_milestones.yaml
```

It evaluates marker, eraser and spatula at exact 250M-frame targets. The
predeclared all-24-task comparison resolves both policies' exact 5B target
checkpoints from their suite identities, avoiding hard-coded actual-frame
filenames:

```bash
.venv/bin/python scripts/run_connectome_evaluation.py \
  --config configs/connectome/evaluation/ppo_1952_fly_lstm_matched_5b_all_tasks.yaml
```

Finally, analyze only the global intersection of completed sentinel milestones:

```bash
.venv/bin/python scripts/analyze_recurrent_baseline.py \
  --config configs/connectome/comparisons/ppo_1952_fly_lstm_matched.yaml
```

The analysis helper writes `summary.json`, `common_milestones.csv` and a task
progress plot. It rejects an empty common cohort and never substitutes nearby
checkpoints.

## Evidence boundary

The configured comparison uses only training seed 42. Task-wise differences
across the 24 deterministic trajectories are useful coverage but are not
independent training-seed replicates. Without multiple training seeds or a
topology/statistics-matched randomized reservoir, fly superiority is evidence
consistent with a helpful frozen prior, not causal attribution to biological
wiring.

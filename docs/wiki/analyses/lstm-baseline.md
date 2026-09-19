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

### What the coefficient match does not isolate

The 323-unit result is a coefficient-budget control, not a unit-count-matched
test of recurrent sparsity. A standard LSTM also carries both a hidden state
and a cell state. Direct construction through the repository actor builder gives:

| LSTM width | Recurrent state scalars per environment | Actor parameters | Interpretation |
| ---: | ---: | ---: | --- |
| 323 | 646 | 733,790 | instantiated-coefficient match |
| 976 | 1,952 | 4,749,740 | scalar recurrent-state match (`h` plus `c`) |
| 1,952 | 3,904 | 17,111,756 | hidden-output/unit-count match |

Therefore a claim about the benefit of representing dynamics with 1,952 sparse
units needs the 1,952-unit LSTM as an additional control. It holds the recurrent
output width fixed and lets parameter count, memory and latency expose the cost
of dense gated recurrence. The 323-unit control should still be retained: it
asks whether the fly actor outperforms a much narrower learned recurrent model
under approximately the same stored-coefficient budget. These are different
questions and neither should be relabeled as the other.

Even the 1,952-unit LSTM is not a pure topology control because its four gates,
cell state and update equation differ from the fly rate network. Biological
wiring attribution additionally requires 1,952-unit sparse controls with the
same edge count and dynamics but randomized or degree-preserving rewired
support. Report both equal-environment-frame and equal-wall-time results;
runtime and memory advantages at equal width are direct sparsity benefits,
while task-performance differences also reflect the recurrent equations.

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

For the active fly profile, `max_sigma: 3.0` transforms each raw learned
log-scale `r` into `log_sigma = log(3) - softplus(log(2) - r)`, equivalently
`sigma = 3 / (1 + 2 * exp(-r))`. This preserves sigma one at initialization
(`r=0`) and approaches three smoothly; the raw parameter is not clamped.
The derivative of emitted log-sigma decreases toward zero near the ceiling.
Each of six SAPG blocks has 29 independently learned raw scales. Bounding
their emitted scales bounds the 29-dimensional diagonal Gaussian entropy
above by `29 * (0.5 * log(2*pi*e) + log(3)) = 73.009` nats. This is not a
direct clamp on the entropy scalar and does not measure diversity after
hard clipping sampled actions to `[-1,1]`. The official LSTM profile omits
this custom cap, so the active official-versus-fly comparison also differs
in exploration parameterization.

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

The smoke gate remains available but was not launched for the historical
parameter-matched handoff. The production command below is its preserved
contract:

```bash
# Two-epoch integration gate.
.venv/bin/python scripts/run_connectome_suite.py \
  --config configs/connectome/suites/ppo_lstm323_matched_gaussian_lf_entropy1x_sigma3_smoke.yaml

# Historical seed-42 parameter-matched production contract.
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

This parameter-matched run was superseded on 2026-09-18 at printed frame
261,685,248. Its trainer and watcher were stopped and its artifacts were
preserved. It was replaced on physical GPU 1 by the repository's official
legacy LSTM/SAPG method described below.

## Official repository-method replacement

The active `SimToolRealLSTMAsymmetricSAPG` run reproduces the repository's
legacy `launch_training.py` contract with seed 42 and 12,288 physical
environments as the two intentional deviations from its defaults. The resolved
actor is a 1,024-unit, one-update LSTM followed by
`[1024, 1024, 512, 512]`; the critic has the same MLP widths. The active run
retains 98,304-sample actor and critic minibatches, two mini-epochs, KL `.016`,
entropy scale `.002`, force scale 20, torque scale 2 and explicit
`use_experimental_cv: true`. Its unbounded coefficient-conditioned Gaussian is
also the official profile rather than the sigma-three matched-baseline variant.

The exact 24,576-environment geometry fit physical GPU 1, but it was stopped at
printed frame 18,087,936 when the user selected the fly job's physical
environment count. Its artifacts remain preserved. The active fresh run uses
12,288 environments and six 2,048-environment SAPG blocks, while retaining the
official 98,304-sample minibatches and every other setting above. It does not
use rollout accumulation, so its update phase consumes 196,608 fresh frames,
matching the fly job's simultaneous environment count and rollout length but
not its smaller 49,152-sample minibatches.

Initial reduced-environment telemetry through frame 1,769,472 was finite:
reward 42.1621, actor loss `.00917`, actor-side value loss `.24833`, privileged
critic loss `.17640`, entropy 41.1773, KL `.01073`, and zero invalid-KL flags in
both mini-epochs. This establishes launch fidelity and liveness, not
reproduction of published learning results.

The live tmux sessions are `connectome-official-lstm-sapg-seed42-env12288` and
`connectome-official-lstm-sapg-seed42-env12288-eval`. Artifacts are rooted under
`train_dir/connectome/official_lstm/ppo_official_repo_lstm_sapg_seed42_env12288`;
logs are under `profiles/connectome/official_lstm_sapg_seed42_env12288/`.
TensorBoard 6008 exposes the run as
`official_repo_lstm_sapg_seed42_env12288`.

The actual production and watcher entrypoints are:

```bash
.venv/bin/python scripts/run_connectome_suite.py \
  --config configs/connectome/suites/ppo_official_repo_lstm_sapg_seed42_env12288.yaml

.venv/bin/python scripts/run_connectome_milestone_evaluation.py \
  --config configs/connectome/evaluation/ppo_official_repo_lstm_sapg_seed42_env12288_milestones.yaml
```

The suite YAML owns seed, physical GPU, environment/block geometry, profile,
optimizer sizes, epoch budget, task randomization and explicit auxiliary-value
setting. `run_connectome_suite.py` composes the Hydra configuration, launches
the child trainer, streams logs and verifies its final checkpoint. The watcher
polls for exact 250M-frame snapshots and evaluates three deterministic sentinel
tasks. The original 24,576-environment suite and `_memory_preserving` suite are
preserved historical/alternative contracts and must not be launched
concurrently with the active 12,288-environment run.

The historical matched-LSTM watcher command was:

```bash
.venv/bin/python scripts/run_connectome_milestone_evaluation.py \
  --config configs/connectome/evaluation/ppo_lstm323_matched_gaussian_lf_entropy1x_sigma3_100b_milestones.yaml
```

It evaluates marker, eraser and spatula at exact 250M-frame targets. The
predeclared all-24-task comparison resolves both policies' exact 5B target
checkpoints from their suite identities, avoiding hard-coded actual-frame
filenames:

## Fresh asymmetric matched-pair contract

The current paired contract returns the 323-unit coefficient-matched LSTM to
physical GPU 0 and runs the 1,952-neuron all-neuron fly actor on physical GPU
1. Both use `SimToolRealLSTMAsymmetric`, so the privileged critic is the same
`[1024, 1024, 512, 512]` MLP; `use_experimental_cv: false` disables the actor's
auxiliary value objective in both policies. The actor comparison remains
733,790 instantiated LSTM coefficients versus 733,796 fly coefficients.

```bash
.venv/bin/python scripts/run_connectome_suite.py \
  --config configs/connectome/suites/ppo_lstm323_fly1952_asymmetric_noaux_mlp128x32x32_100b.yaml

.venv/bin/python scripts/run_connectome_milestone_evaluation.py \
  --config configs/connectome/evaluation/ppo_lstm323_fly1952_asymmetric_noaux_mlp128x32x32_100b_milestones.yaml
```

The suite assigns the first profile to CUDA 0 and the second to CUDA 1 through
`gpu_assignments: [0, 1]` and `max_parallel: 2`. Each independent single-GPU
trainer uses 12,288 environments, six 2,048-environment SAPG blocks, 49,152
actor/critic logical and physical minibatches, two PPO mini-epochs, LF/1.0,
seed 42 and the same perturbation and 100B-frame budget. The evaluator produces
both paper-tolerance and exact checkpoint-training-tolerance videos for the
three deterministic sentinel tasks at every 250M-frame checkpoint.

The pair launched fresh on 2026-09-18 in tmux
`connectome-asymmetric-matched-pair-100b`: the LSTM trainer is PID 671749 on
physical CUDA 0 and the fly trainer is PID 671748 on physical CUDA 1. The
coordinator is PID 671476. Both emitted finite frame progress on the expected
196,608-frame epoch grid and report four logical minibatches per PPO mini-epoch.
The dual-tolerance watcher is PID 671483 in
`connectome-asymmetric-matched-pair-100b-eval`. A symlink exposes both summary
streams to the existing TensorBoard server on port 6008. These observations
establish placement and launch health, not comparative learning performance.

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

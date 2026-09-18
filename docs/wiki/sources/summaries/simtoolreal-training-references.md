# SimToolReal Training References

SimToolReal provides a published five-seed reward curve, a later upstream seed-0 numerical export, and a terminal pretrained checkpoint, but no raw five-seed paper history. The paper directly supports the privileged asymmetric critic, while the released code's additional actor-side value loss is not described or ablated in the paper.

Last updated: 2026-09-18

Related: [Billion-step adaptation run](../../analyses/adaptation-1b-run.md), [Experiment workflow](../../workflows/connectome-experiments.md)

## Sources

- Paper HTML and Figure 8: `https://arxiv.org/html/2602.16863#S4.F8`
- Main repository README and `download_pretrained_policy.py`
- Upstream branch `2026-08-12_SimToolReal_Transformer_Study`, commit `b34781b4ba36f05b7089a041530b62f186726d02`, especially `study/README.md`, `study/plot_reward_curves.py`, and `study/results/reward_curves_eval_policy_seed0_20260812_checkin.csv`
- Released archive: `https://download.cs.stanford.edu/juno/simtoolreal/pretrained_policy.zip`, last modified 2026-02-17 at inspection

## Paper critic contract versus released implementation

The main paper describes a conventional asymmetric actor-critic division: the recurrent actor receives the restricted, deployment-available observation, including noisy and delayed signals, while a separate critic receives exact, instantaneous, noise-free simulator state plus privileged velocities, reward/progress features, and object pose. Appendix Table I lists an `LSTM[1024] + MLP[1024,1024,512,512]` actor and an `MLP[1024,1024,512,512]` critic.

Figure 8 isolates the value-input asymmetry over five random seeds. Its `SAPG + Symmetric Critic` ablation forces the critic to use the same partial observations as the actor and performs substantially worse than `SAPG + Asymmetric Critic`. This supports retaining `central_value_config` and its clean privileged input. It does not test whether the actor network should simultaneously fit another value function.

The paper contains no description of `use_experimental_cv`, an auxiliary actor-side value head, two concurrent value losses, or a target for such a second value function. Absence from the paper is not proof that this path was inactive: the released code inherits `use_experimental_cv: true` from `rl_games`, so exact code reproduction retains it. The evidence therefore separates two notions of fidelity:

- Released-code fidelity uses `use_experimental_cv: true` and trains both the privileged critic and the actor model's value head.
- The paper's stated algorithm only establishes the separate privileged critic. Setting `use_experimental_cv: false` while retaining `central_value_config` preserves that paper-backed mechanism and removes an undocumented auxiliary loss.

No published ablation determines whether the inherited auxiliary actor-value objective helps or hurts SimToolReal. A matched true-versus-false run with identical privileged critic inputs is required to answer that question.

### Live official-method reproduction

On 2026-09-18, the tracked legacy launch was materialized as a YAML-owned live
run on physical GPU 1. Seed 42 is the only intentional training-setting change
from `launch_training.py`'s default seed 0. The run retains 24,576 environments,
4,096 environments per SAPG block, 98,304-sample minibatches, the official
1,024-unit LSTM and MLPs, entropy scale `.002`, KL `.016`, force/torque scales
20/2 and the released-code default `use_experimental_cv: true`. The exact
physical geometry completed optimizer updates on this GPU, so the prepared
12,288-environment rollout/microbatch fallback was not used. Early finite
telemetry is an integration result only and does not reproduce the paper curve.

## Available comparison data

Paper Figure 8 plots episode reward against 0--9 billion environment steps for SAPG plus asymmetric critic, PPO plus asymmetric critic, and SAPG plus symmetric critic. Curves and uncertainty bands are averages across five seeds. The paper and main branch do not provide the underlying per-seed values, TensorBoard event files, or a public W&B run identifier, so numerical values from this figure would require plot digitization and should be labeled approximate.

### Figure 8 pixel calibration

The arXiv v2 source contains a 1,915 x 1,041 PNG for Figure 8. Its plotting area spans approximately pixels 255--1,883 over 0--9 billion environment steps and pixels 820.5--59 over rewards 0--8,000. One horizontal pixel is therefore about 5.53 million frames and one vertical pixel about 10.5 reward. The teal mean stroke is approximately 11--12 pixels thick, or roughly 115--125 reward from lower to upper edge; extracting the stroke midpoint makes estimates to tens, rather than hundreds, of reward reasonable, but does not recover the source data.

At the 2026-09-13 21:01 EDT live snapshots, the 1,952-neuron run was at 249,888,768 frames and the 4,310-neuron run at 681,443,328 frames. These map to horizontal pixels 300.3 and 378.4. The teal stroke midpoints map to paper mean rewards of approximately 92 and 292, respectively; report these as about 95 and 295 with approximately +/-20 digitization uncertainty. The full visible stroke envelope is a more conservative approximately +/-60 reward.

| Actor | Matched frame | Current trailing-25M reward | Digitized Figure 8 mean | Current / paper estimate |
| --- | ---: | ---: | ---: | ---: |
| 1,952-neuron gains, old timing | 249,888,768 | 87.497 | about 95 | about 0.92x |
| 4,310-neuron gains, old timing | 681,443,328 | 228.383 | about 295 | about 0.77x |

The compact policy is indistinguishable from the paper mean at this precision. The 4,310-neuron policy is likely below the paper mean at the matched frame, although the plotted stroke width prevents a precise gap estimate. Unlike the easier-cuboid study CSV comparison below, Figure 8 is the paper's five-seed full-method curve and is the more relevant visual reference. It remains an approximate visual comparison: the image does not reveal its numerical smoothing procedure, per-seed values, or exact plotted samples.

The upstream transformer-study branch contains three exact numerical CSV exports. The newest and most useful comparison file is `reward_curves_eval_policy_seed0_20260812_checkin.csv`: 60,277 data rows covering seven seed-0 series. Its `rlgames-lstm-sapg-seed0` `rewards/step` series has 12,769 points through frame 1,255,145,472, ending at raw reward 371.404 and trailing-25-million-frame mean 374.783. The earlier non-check-in export has 41,615 data rows and five series; its LSTM curve ends at frame 874,217,472. `reward_curves_seed0_20260812.csv` is the all-policy aggregate rather than the evaluation-member comparison.

`study/plot_reward_curves.py` documents how these values were made. It merges restarted TensorBoard event files, keeps the newest event at duplicate steps, and applies a trailing 25,000,000-frame mean. With evaluation-block mode, simple-RL runs use `block_rewards/block_5`, while the RL-Games LSTM uses `rewards/step`, which its logger already filters to block 5. These are therefore curves for the study's zero-entropy exploitation member, not the ID-50/block-0 deployment member used for deterministic videos.

The repository also tracks PNG/PDF ten-run reward plots on branch `2026-08-12_Eigengrasp_Action_Space`, but it does not track a corresponding ten-run CSV. A repository-wide object audit found no committed TensorBoard event files, W&B histories, or raw numerical values for the paper's five-seed Figure 8. The exact CSVs above are the only committed numerical learning histories found.

The transformer study used seed 0, 6,144 environments, entropy scale 0.005, nominal minibatches of 98,304, and an easier 654-shape small-cuboid-only distribution. Its README explicitly says the old heavy-tool curve is context rather than a direct reproduction target. Reward-axis alignment therefore supports a useful diagnostic comparison, but reward magnitude is not a controlled actor-only comparison against the current seed-42 full tool-distribution runs.

The released 366,876,642-byte zip contains only `config.yaml` and `model.pth`, not event history. CPU inspection of the checkpoint recorded epoch 218,836, frame 86,049,816,576, `last_mean_rewards` 13,603.997, and 20 optimizer entries whose Adam step counters were all 1,750,000. The configuration identifies seed 0 and a prior checkpoint path, so this is a resumed/fine-tuned endpoint rather than evidence for the trajectory in Figure 8.

## Comparability to the connectome run

### Verified release settings and memory-preserving comparison

Reinspection of the archived release YAML confirms exploration scale 0.005, force scale 2, force probability range [0.001, 0.1], decay 0.99, decay interval 0.08, and forces only when lifted. Torque and velocity-impulse settings are absent; the environment loader defaults their scales to zero. `configs/connectome/suites/adaptation_1b_release_settings.yaml` explicitly applies these settings and is the active two-policy run. Reproduce it with `.venv/bin/python scripts/run_connectome_suite.py --config configs/connectome/suites/adaptation_1b_release_settings.yaml`.

Correction to earlier units: `forceScale` and `torqueScale` multiply Gaussian noise and object mass in `env.py`; they are not fixed forces in newtons or torques in newton-metres. The prior descriptions of 20 N/2 Nm overstated what those configuration fields mean.

The active run reproduces the released effective 393,216-transition update phase using two consecutive 12,288-environment, horizon-16 rollouts collected before any weight update. This avoids an oversized PhysX scene: at 24,576 simultaneous environments, full and microbatched training either OOMed before an update or repeatedly faulted in GPU contact preparation near epoch 24. The two rollout halves preserve horizon-16 GAE and use the same sampled off-policy SAPG block before concatenation and one global sequence shuffle. Simulator and recurrent state continue across the collection boundary, so this matches update scheduling and GAE horizon but has half as many simultaneous environment identities as the release.

Actor and critic retain 98,304-sample logical minibatches. Physical 24,576-sample chunks use sample-weighted accumulation, including the enlarged final SAPG minibatch, followed by one gradient clip and Adam step. The KL scheduler retains its once-per-mini-epoch cadence. A completed smoke gate records exactly eight actor Adam steps per 393,216-equivalent update phase, and epoch-10 checkpoints record 80 steps. Across the completed billion-step run, each actor applied 20,335 of 20,344 scheduled calls; PyTorch `GradScaler` suppressed nine non-finite mixed-precision updates. Focused numerical regressions confirm accumulated central-critic parameters and the optimizer count match a single finite logical update.

If rollout storage itself does not fit, offloading buffers or distributing a single policy across both GPUs can preserve the original logical batch. Simply collecting longer rollouts in half as many environments changes GAE and rollout semantics. For actor attribution, rerun the original LSTM and connectome variants with the same chosen memory strategy, seeds, data budget, optimizer schedule, and held-out evaluation protocol. A matched reduced-scale LSTM control is scientifically useful even when reproducing the original full-scale schedule is infeasible.

The live connectome jobs emit `rewards/step` on the same environment-frame unit. Applying the author's exact trailing-25-million-frame smoothing to a 2026-09-13 snapshot gave the following diagnostic comparison against the newest study CSV at the same frame:

| Actor | Live frame | Connectome smoothed reward | Study LSTM smoothed reward | Ratio |
| --- | ---: | ---: | ---: | ---: |
| 4,310-neuron gains, old timing | 646,447,104 | 236.422 | 87.893 | 2.690x |
| 1,952-neuron gains, old timing | 206,635,008 | 82.937 | 53.994 | 1.536x |

This does not establish superiority to the paper policy because the seed, object population, simultaneous environment identities, and training endpoint differ. It is also only the block-5 dense training-reward comparison. The 4,310-neuron policy's deterministic ID-50/block-0 evaluation was still at 0.7407% mean Task Progress at both the 250,085,376- and 500,170,752-frame milestones, while its latest training success ratio was approximately 0.0023%. The higher dense reward has therefore not yet produced a corresponding held-out deployment-policy success gain.

The stopped exploratory connectome contract was not optimizer-step matched to the release: it used 12,288 environments, exploration scale 0.002, force scale 20 and torque scale 2. The replacement contract restores the release perturbations and eight actor/critic steps per 393,216 fresh frames through rollout accumulation. Paper Table I and the released checkpoint still differ in their physical perturbation configuration, so comparisons must identify which source is being matched and report environment steps, optimizer steps, seed count, environment distribution, and resolved settings together.

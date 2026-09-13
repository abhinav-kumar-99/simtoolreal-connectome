# SimToolReal Training References

SimToolReal provides a published five-seed reward curve, a later upstream seed-0 numerical export, and a terminal pretrained checkpoint, but no raw five-seed paper history.

Last updated: 2026-09-13

Related: [Billion-step adaptation run](../../analyses/adaptation-1b-run.md), [Experiment workflow](../../workflows/connectome-experiments.md)

## Sources

- Paper HTML and Figure 8: `https://arxiv.org/html/2602.16863#S4.F8`
- Main repository README and `download_pretrained_policy.py`
- Upstream branch `2026-08-12_SimToolReal_Transformer_Study`, commit `8e17054dc8e2b7cd208ef55ef29a75114c82e29e`, especially `study/README.md` and `study/results/reward_curves_eval_policy_seed0_20260812.csv`
- Released archive: `https://download.cs.stanford.edu/juno/simtoolreal/pretrained_policy.zip`, last modified 2026-02-17 at inspection

## Available comparison data

Paper Figure 8 plots episode reward against 0--9 billion environment steps for SAPG plus asymmetric critic, PPO plus asymmetric critic, and SAPG plus symmetric critic. Curves and uncertainty bands are averages across five seeds. The paper and main branch do not provide the underlying per-seed values, TensorBoard event files, or a public W&B run identifier, so numerical values from this figure would require plot digitization and should be labeled approximate.

The upstream transformer-study branch contains an actual CSV export of TensorBoard `rewards/step`. Its seed-0 `rlgames-lstm-sapg` series reaches frame 874,217,472 with raw reward 360.943 and a trailing-25-million-frame mean of 359.038. At frame 80,019,456 it records raw reward 42.344 and smoothed reward 40.231. This study used 6,144 environments and an easier 654-shape small-cuboid-only distribution; its README explicitly says the old heavy-tool curve is context rather than a direct reproduction target.

The released 366,876,642-byte zip contains only `config.yaml` and `model.pth`, not event history. CPU inspection of the checkpoint recorded epoch 218,836, frame 86,049,816,576, `last_mean_rewards` 13,603.997, and 20 optimizer entries whose Adam step counters were all 1,750,000. The configuration identifies seed 0 and a prior checkpoint path, so this is a resumed/fine-tuned endpoint rather than evidence for the trajectory in Figure 8.

## Comparability to the connectome run

### Verified release settings and memory-preserving comparison

Reinspection of the archived release YAML confirms exploration scale 0.005, force scale 2, force probability range [0.001, 0.1], decay 0.99, decay interval 0.08, and forces only when lifted. Torque and velocity-impulse settings are absent; the environment loader defaults their scales to zero. `configs/connectome/suites/adaptation_1b_release_settings.yaml` explicitly applies these settings and is the active two-policy run. Reproduce it with `.venv/bin/python scripts/run_connectome_suite.py --config configs/connectome/suites/adaptation_1b_release_settings.yaml`.

Correction to earlier units: `forceScale` and `torqueScale` multiply Gaussian noise and object mass in `env.py`; they are not fixed forces in newtons or torques in newton-metres. The prior descriptions of 20 N/2 Nm overstated what those configuration fields mean.

The active run reproduces the released effective 393,216-transition update phase using two consecutive 12,288-environment, horizon-16 rollouts collected before any weight update. This avoids an oversized PhysX scene: at 24,576 simultaneous environments, full and microbatched training either OOMed before an update or repeatedly faulted in GPU contact preparation near epoch 24. The two rollout halves preserve horizon-16 GAE and use the same sampled off-policy SAPG block before concatenation and one global sequence shuffle. Simulator and recurrent state continue across the collection boundary, so this matches update scheduling and GAE horizon but has half as many simultaneous environment identities as the release.

Actor and critic retain 98,304-sample logical minibatches. Physical 24,576-sample chunks use sample-weighted accumulation, including the enlarged final SAPG minibatch, followed by one gradient clip and Adam step. The KL scheduler retains its once-per-mini-epoch cadence. A completed smoke gate records exactly eight actor Adam steps per 393,216-equivalent update phase, and live epoch-10 checkpoints record 80 steps. Focused numerical regressions confirm accumulated central-critic parameters and the optimizer count match a single logical update.

If rollout storage itself does not fit, offloading buffers or distributing a single policy across both GPUs can preserve the original logical batch. Simply collecting longer rollouts in half as many environments changes GAE and rollout semantics. For actor attribution, rerun the original LSTM and connectome variants with the same chosen memory strategy, seeds, data budget, optimizer schedule, and held-out evaluation protocol. A matched reduced-scale LSTM control is scientifically useful even when reproducing the original full-scale schedule is infeasible.

The live connectome jobs emit `rewards/step` on an environment-frame axis, which is the right local series to compare. A 2026-09-13 snapshot found adapters-only at frame 81,199,104 and reward 53.314, and neuron-gains at frame 83,361,792 and reward 56.820. The study-branch seed-0 LSTM value near 80 million frames was about 42.3 raw, but this is illustrative only because its object distribution and seed differ.

The stopped exploratory connectome contract was not optimizer-step matched to the release: it used 12,288 environments, exploration scale 0.002, force scale 20 and torque scale 2. The replacement contract restores the release perturbations and eight actor/critic steps per 393,216 fresh frames through rollout accumulation. Paper Table I and the released checkpoint still differ in their physical perturbation configuration, so comparisons must identify which source is being matched and report environment steps, optimizer steps, seed count, environment distribution, and resolved settings together.

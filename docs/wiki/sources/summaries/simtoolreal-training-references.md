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

The live connectome jobs emit `rewards/step` on an environment-frame axis, which is the right local series to compare. A 2026-09-13 snapshot found adapters-only at frame 81,199,104 and reward 53.314, and neuron-gains at frame 83,361,792 and reward 56.820. The study-branch seed-0 LSTM value near 80 million frames was about 42.3 raw, but this is illustrative only because its object distribution and seed differ.

The active connectome contract is not optimizer-step matched to the paper. It uses 12,288 environments rather than 24,576 and scales the minibatch to retain four minibatches per epoch. It therefore performs eight actor updates per 196,608 frames, twice the actor-update density of the paper-size geometry. It also uses `expl_reward_coef_scale: 0.002` and perturbations of 20 N/2 Nm, whereas paper Table I reports 0.005 and 5 N/0.5 Nm. The released checkpoint uses exploration scale 0.005 and force scale 2, showing an additional paper-versus-release mismatch. Comparisons must report environment steps, optimizer steps, seed count, environment distribution, and these resolved settings together.

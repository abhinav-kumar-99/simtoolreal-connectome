# One-million-step adaptation pilot

Five Triton-backed connectome adaptation policies completed 983,040 environment steps each; none reached a demonstrated evaluation waypoint after this 40-update pilot.

Last updated: 2026-09-13

Related: [Experiment workflow](../workflows/connectome-experiments.md), [Adaptation modes](../concepts/connectome-adaptation.md), [Sparse backends](sparse-backends.md)

## Training contract

The pilot used seed 42 and ordinary single-GPU SAPG processes, with two policies trained concurrently on physical GPUs 0 and 1. The exact paper geometry of 24,576 environments, 4,096 environments per exploration block, and 98,304 actor/critic minibatches OOMed on both 24 GB GPUs at the first critic update, after 29.79 and 29.09 seconds. Those generated failure artifacts remain under `train_dir/connectome/adaptation_1m_triton/`.

The completed configuration scaled all three batch dimensions by two: 12,288 environments, six 2,048-environment exploration blocks, 49,152 actor/critic minibatches, horizon and sequence length 16, two mini-epochs, and five epochs. This preserves four minibatches per epoch and all paper optimizer, reward, randomization, and SAPG settings. Five epochs equal 983,040 environment steps, below the strict 1,000,000 cap.

Each actor received 40 Adam updates: five epochs times two mini-epochs times four minibatches. The asymmetric critic followed the same 40-update schedule. Checkpoint Adam counters independently verify 40 for every trainable actor tensor.

## Training results

Final TensorBoard values are reported at logged frame 786,432; checkpoints record the post-epoch frame count of 983,040. Reward is the repository's raw mean episode return and its configured 0.01-shaped value.

| Adaptation | GPU | Train wall time, s | Raw reward | Shaped reward | Success ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| Adapters only | 0 | 31.405 | 52.927 | 0.5293 | 0.000 |
| Neuron gains | 1 | 30.716 | 52.947 | 0.5295 | 0.000 |
| Low rank, rank 4 | 0 | 30.820 | 52.933 | 0.5293 | 0.000 |
| Independent edges | 1 | 29.492 | 52.913 | 0.5291 | 0.000 |
| Gains plus dynamics | 0 | 29.680 | 52.928 | 0.5293 | 0.000 |

All checkpoints reached epoch 5, contained optimizer state, recorded 983,040 frames, and reloaded through `deployment.RlPlayer` to emit finite `(1, 29)` actions.

## Closed-loop evaluation

The requested three videos per policy use `claw_hammer/swing_down`, `long_screwdriver/spin_vertical`, and `blue_brush/sweep_forward`. One deterministic rollout was run per case. Paper Task Progress used a configured-base tolerance of 0.02; repository `avg_goal_pct` used 0.01. The inherited `keypointScale: 1.5` made the implemented maximum-keypoint thresholds 0.03 m and 0.015 m. Both were 0% for every policy and case.

The mean raw rollout rewards across the three selected cases at the 2 cm definition were 23.853 (adapters), 23.331 (gains), 19.613 (low rank), 24.310 (edgewise), and 25.836 (gains plus dynamics). The 1 cm runs produced identical rewards because no policy reached even the first waypoint under either threshold. These rewards are descriptive trajectory-rollout returns, not evidence of task completion.

This is not the paper's full evaluation protocol: the paper averages Task Progress over five rollouts and the repository contains 24 object-task combinations. The pilot uses three selected cases and one rollout per case to satisfy the requested video set. Conclusions about adaptation quality require materially longer training and the full matched evaluation cohort.

## Interpreting the optimization budget

The connectome changes the actor parameterization and permitted trainable weights; `a2c_continuous.py` still applies the SAPG/PPO actor objective and Adam. A frozen biological circuit is not a policy pretrained on robot tool use. Fewer trainable parameters alone therefore do not establish that fewer or more passes over each rollout are appropriate.

The pilot collected only five rollout batches, corresponding to 80 control steps per environment slot before accounting for resets. More fresh rollout batches and more optimization passes over an existing batch are distinct interventions. A proposed next experiment is to retain two mini-epochs while increasing the fresh-experience budget, then compare actor mini-epochs 2 versus 4 with the critic schedule held fixed and matched environment-step and wall-time reporting. This is a proposal, not a launched experiment.

Live event-file inspection found final logged `info/kl` values of 0.00123–0.00142 and `info/last_lr` of 0.0038443 across the five policies. The initial configured actor learning rate is 0.0001. `a2c_common.py` calls the adaptive scheduler after every mini-epoch; `schedulers.py` multiplies the learning rate by 1.5 when KL falls below half the 0.016 threshold. Increasing mini-epochs therefore changes scheduler frequency as well as data reuse. The current loop has no KL early-stop condition, and `dataset.update_mu_sigma` refreshes the KL reference between passes, so logged KL is not a fixed rollout-policy trust-region bound. A controlled reuse experiment should account for those details and monitor fixed-reference policy drift and clipping.

Reference: [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) describes alternating environment sampling with multiple minibatch optimization passes.

## Artifacts

- Training summary: `train_dir/connectome/adaptation_1m_triton_12k/suite_results.json`
- Evaluation summary: `evals/connectome/adaptation_1m_triton_12k/summary.json`
- Videos: `evals/connectome/adaptation_1m_triton_12k/paper_task_progress/<policy>/<object>/<task>/rollout.mp4`

All 15 videos are nonempty H.264 MP4s with 100 frames, 10 seconds duration, and 400 by 226 resolution. Beginning, middle, and final frames from a sampled rollout were visually inspected and showed distinct live simulator states.

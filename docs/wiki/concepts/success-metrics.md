# Training and Evaluation Success Metrics

TensorBoard training-success tags summarize the most recently completed random-goal episodes, while deterministic trajectory Task Progress is a separate post-training evaluation metric.

Last updated: 2026-09-13

Related: [Billion-step adaptation run](../analyses/adaptation-1b-run.md), [Experiment workflow](../workflows/connectome-experiments.md)

## What one training success means

For each environment, the task computes the Euclidean distance between corresponding object and goal keypoints and takes the furthest-keypoint distance. A state is near the goal when that maximum is at most `success_tolerance * keypointScale`. The active configuration starts at `successTolerance: 0.075` with `keypointScale: 1.5`, so the implemented maximum-keypoint threshold is 0.1125 m, not 0.075 m. `scalars/success_tolerance` logs the unscaled value. The tolerance curriculum can multiply the base tolerance by 0.9 down to 0.01, corresponding to an implemented 0.015 m threshold.

The active suites require ten consecutive near-goal control steps for one success. A success increments the environment's counter and advances to another random goal without ending the whole episode. At a full environment reset, that counter is copied to `prev_episode_successes` and then cleared. Most training curves therefore describe the last completed episode retained in each environment slot; they are not a count of new successes during the current update phase.

## SAPG block layout

The current 100-billion-frame jobs partition 12,288 environments into six contiguous blocks of 2,048. Every block uses the same simulator, extrinsic task reward, success tolerance, ten-step success rule, and 50-goal episode ceiling. The blocks differ in the scalar coefficient supplied to the policy and in the coefficient multiplying policy entropy in the PPO loss. With the released-checkpoint-matched `expl_reward_coef_scale: 0.005`, the exact layout is:

| Block | Environment indices | Policy coefficient ID | Entropy coefficient |
| --- | --- | ---: | ---: |
| 0 | `0:2048` | 50 | 0.0025 |
| 1 | `2048:4096` | 40 | 0.0020 |
| 2 | `4096:6144` | 30 | 0.0015 |
| 3 | `6144:8192` | 20 | 0.0010 |
| 4 | `8192:10240` | 10 | 0.0005 |
| 5 | `10240:12288` | 0 | 0 |

The policy mean, coefficient-conditioned action standard deviation, actor value, and central critic can all depend on the coefficient ID. The entropy coefficient is computed as `linspace(0.5, 0, 6) * 0.005`; it is a loss coefficient, not a separate success threshold or an additive environment reward in this entropy configuration. The base SAPG YAML uses scale `0.002`, but the current jobs override it to `0.005` to match the released checkpoint.

## TensorBoard tags

| Tag stem | Exact population and aggregation | Interpretation |
| --- | --- | --- |
| `scalars/success_tolerance` | Current unscaled curriculum tolerance | Difficulty context; multiply by `keypointScale` for the implemented distance threshold. |
| `mean_successes` | Mean `prev_episode_successes` across all 12,288 environments | Average goals reached in the most recently completed episode stored by each environment. |
| `success_ratio` | All-environment `mean_successes / maxConsecutiveSuccesses` | Fraction of the 50-goal training ceiling. It is a fraction, not a percentage. |
| `mean_success_ratio` | The same all-environment mean divided by 50 | A duplicate of `success_ratio` in the current implementation. |
| `successes` | Mean over the retained 2,048-environment block 5 | Zero-entropy-coefficient block version of average completed-episode successes. |
| `successes_median`, `successes_max` | Median and maximum over block 5 | Distribution diagnostics for that zero-entropy-coefficient block. |
| `successes_per_block/block_0` ... `block_5` | Mean over each 2,048-environment SAPG exploration block | Exploration diagnostic. Block 0 has the largest entropy-loss coefficient; block 5 has zero. |

The observer filters the ordinary `successes` tensor to block 5 but computes `success_ratio` inside the environment before filtering, across all six blocks. This is why `successes / 50` need not equal `success_ratio`. With 2,048 environments, block means move in increments of 1/2,048; all-environment means move in increments of 1/12,288. Block 5 being selected for these ordinary training summaries should not be conflated with deployment: `deployment/rl_player.py` supplies coefficient ID 50, corresponding to block 0, and deterministic evaluation then uses that conditioned policy's mean action.

Every `/frame`, `/iter`, and `/time` suffix is an alias written with the same scalar value and the same environment-frame `global_step`. The suffix does not select a different x-axis. TensorBoard's Step, Wall, and Relative controls determine whether the display uses environment frames or elapsed time.

`true_objective_mean` and `true_objective_max` are related curriculum/PBT scores, not success rates. Before the tolerance reaches its target, the score prioritizes normalized tolerance progress and adds `0.01 * successes`; after reaching the target it uses successes plus the completed-curriculum offset.

## Deterministic trajectory evaluation

Milestone evaluation is not currently written into the training TensorBoard event files. It is stored under `evals/connectome/` as per-case `eval.json` and aggregate `summary.json` files. The evaluator runs the Gaussian mean action, counts demonstrated trajectory waypoints reached, and reports `task_progress_pct = 100 * successful_waypoints / total_waypoints`.

The evaluation YAML's `success_tolerance_m` is passed to the environment as the unscaled base tolerance. With the inherited `keypointScale: 1.5`, configured values 0.02 and 0.01 produce implemented maximum-keypoint thresholds of 0.03 m and 0.015 m respectively. Existing artifact labels call these the paper 2 cm and repository 1 cm settings because those are their configured base tolerances; consumers should record the 1.5 scale when reporting the actual threshold used by code.

For learning-curve diagnosis, plot `success_ratio` together with `scalars/success_tolerance` and inspect per-block curves for exploration effects. For final policy comparison, use matched deterministic Task Progress evaluations rather than treating the training success curves as evaluation success rates.

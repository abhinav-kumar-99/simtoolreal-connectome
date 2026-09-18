# Training and Evaluation Success Metrics

TensorBoard training-success tags summarize the most recently completed random-goal episodes, while deterministic trajectory Task Progress is a separate post-training evaluation metric.

Last updated: 2026-09-18

Related: [Billion-step adaptation run](../analyses/adaptation-1b-run.md), [Experiment workflow](../workflows/connectome-experiments.md)

## What one training success means

For each environment, the task computes the Euclidean distance between corresponding object and goal keypoints and takes the furthest-keypoint distance. A state is near the goal when that maximum is at most `success_tolerance * keypointScale`. The active configuration starts at `successTolerance: 0.075` with `keypointScale: 1.5`, so the implemented maximum-keypoint threshold is 0.1125 m, not 0.075 m. `scalars/success_tolerance` logs the unscaled value. The tolerance curriculum can multiply the base tolerance by 0.9 down to 0.01, corresponding to an implemented 0.015 m threshold.

The active suites require ten consecutive near-goal control steps for one success. A success increments the environment's counter and advances to another random goal without ending the whole episode. At a full environment reset, that counter is copied to `prev_episode_successes` and then cleared. Most training curves therefore describe the last completed episode retained in each environment slot; they are not a count of new successes during the current update phase.

## Training curricula

### Goal-tolerance curriculum: active but not yet advancing

This is the only curriculum that can currently change task difficulty. It starts with unscaled tolerance `0.075` and targets `0.01`. Every 3,000 vector-environment control steps, it is eligible to multiply the current tolerance by `0.9`, clipped at the target, but only if `prev_episode_successes.mean() >= 3` across all 12,288 environments. `prev_episode_successes` is copied from each environment only when that environment's episode ends, so this is an all-environment average over most recently completed episodes rather than the current rollout's successes. If the performance gate fails, the code keeps checking on later control steps and does not move the last-update marker.

The sequence begins `0.075 -> 0.0675 -> 0.06075 -> 0.054675 -> ... -> 0.0101314 -> 0.01`, requiring 20 successful advances. Because the geometric threshold is `tolerance * keypointScale` and `keypointScale=1.5`, this corresponds to tightening the maximum corresponding-keypoint distance from 0.1125 m to 0.015 m. Ten consecutive control steps inside that threshold produce one goal success and a new random goal; the environment episode can accumulate up to 50 goals.

With 12,288 environments, 3,000 control steps correspond to 36,864,000 environment frames. Even if the performance gate passed continuously from startup, reaching the target would therefore require at least 737,280,000 frames. The tanh-squashed KL-0.016 run had already passed the first time gate but still logged tolerance `0.075` at frame 361,955,328: mean completed-episode successes was only 0.00407, versus the required 3.0. Its frame-353,697,792 checkpoint recorded `frame_since_restart: 28784`, `last_curriculum_update: 0`, mean successes 0.00423 and maximum per-environment successes 2. Because a failed performance check leaves the last-update marker unchanged, the environment now checks every control step and will tighten immediately from `0.075` to `0.0675` on the first step whose all-environment mean reaches three. A single environment reaching three is insufficient. After an advance, the next one again requires both another 3,000 control steps and the same performance gate. At the recent 90,000-103,000 environment-frame/s rate, that spacing is roughly six to seven minutes, but the first advance has no predictable wall-clock time because performance is the active blocker.

The dense reward coefficients do not form a stage schedule. Tightening tolerance primarily changes the success/goal-advance condition and associated success bonus; the continuous lifting, fingertip, keypoint, and action terms remain configured throughout training. `evalSuccessTolerance` would overwrite the computed tolerance after the scheduler for evaluation-only configuration, but it is `null` in the active training task profile. Full recovery environment state includes current tolerance and its last-update control step, so a stateful resume can preserve this curriculum.

### Tyler curriculum: counter present, effective behavior disabled

`tyler_curriculum_scale` is a second, wall-clock/performance-gated mechanism intended to move from easy (`0`) to hard (`1`). It increments by `0.01` only when all-environment mean completed successes divided by 50 is strictly greater than `0.6`--that is, mean successes is greater than 30--and more than five wall-clock minutes have elapsed since the previous increment. At uninterrupted qualifying performance, 100 increments would require at least 500 minutes.

Depending on configuration, this scale can progressively remove object velocity, palm velocity, or extra privileged actor observations, either by deterministic scaling or dropout. It can also interpolate arm/hand moving averages and DOF speed toward configured final values. In both live compact resolved configurations, every `turn_off_*` and `turn_off_*_slowly` flag is false, `use_obs_dropout` is false, and all three final controller values are null. Therefore the logged scale currently has no effect even if it were to advance; it remains zero because success ratio is also far below 0.6. Unlike the goal-tolerance state, the Tyler scale is not included in `get_env_state` and initializes from `init_tyler_curriculum_scale` on construction, currently zero.

“Enabled” therefore has two answers. In the legacy Isaac Gym environment, `_update_tyler_curriculum()` is called every control step without an enable guard and its scale is logged, so the updater machinery is active. However, the standard `SimToolReal.yaml`, `origin/main`, the transformer-study branch, the standard `SimToolRealLSTMAsymmetric`/SAPG composition, and the authors' released pretrained `config.yaml` all resolve the three observation-removal modes to false, dropout to false, and the controller final values to null. No tracked launch or configuration override enables a consumer. Thus the standard paper-era Isaac Gym training configuration is functionally Tyler-curriculum-disabled even though the counter/update code runs. The newer recommended Isaac Sim implementation contains no Tyler-curriculum code in this repository.

This does not apply to the goal-tolerance curriculum: that mechanism is genuinely active in both standard Isaac Gym training and the current connectome runs, subject to its `mean successes >= 3` advancement gate.

### Fixed mechanisms that are not curricula

The six SAPG blocks use different fixed entropy-loss coefficients throughout training; they are a simultaneous exploration population, not successive stages. Object/pose randomization, released force perturbations, reward coefficients, network plasticity mode, and optimizer/update timing are also fixed by the run YAML rather than advanced by performance. Milestone deterministic evaluation does not feed results back into either curriculum.

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

A live event-file audit of the seed-42 official LSTM run confirmed this rather
than relying only on source inspection: all 771 available `rewards/step`,
`rewards/iter`, and `rewards/time` entries had identical scalar values and
identical steps from 196,608 through 151,584,768. Their event timestamps spanned
2,128.256 seconds. The few-microsecond timestamp differences among the aliases
only reflect three sequential writer calls. Thus selecting the `/time` tag
while TensorBoard's x-axis remains `Step` still plots environment frames.
Select TensorBoard's `Relative` axis for elapsed wall time or `Wall` for absolute
timestamps; either axis works with the `/step` tag because wall time is event
metadata, not encoded by the tag suffix.

### `true_objective`

`true_objective_mean` and `true_objective_max` are curriculum/PBT ranking diagnostics. They are not the environment reward, PPO objective, success rate, or deterministic Task Progress. PBT is disabled in the current suites, so the value is logged but does not select or replace policies.

The staged LF entropy handoff deliberately uses `true_objective_mean/frame` as its user-selected comparison metric even though PBT itself is disabled. At each 1.25B gate it takes the arithmetic mean of every logged scalar sample in the matched inclusive frame window `[1.15B,1.25B]`; the run must first log a point at or beyond 1.25B. This is an average of equally spaced TensorBoard samples, not an episode-weighted mean or a deterministic evaluation result.

For current success tolerance `tau`, initial tolerance `tau_0`, target tolerance `tau_*`, and current-episode goal count `s`, the active configuration computes

```text
progress = (tau_0 - tau) / (tau_0 - tau_*)

true_objective = progress + 0.01 * s   if tau > tau_*
                 1 + s                 if tau <= tau_*
```

Here `tau_0=0.075` and `tau_*=0.01`. The design makes stricter curriculum stages dominate the score while using a small success-count term to distinguish policies before the target tolerance. Once the target is reached, success count becomes the primary unit. This uses the unscaled tolerance; the separate `keypointScale=1.5` affects the actual geometric success test but not this interpolation.

The `s` in this formula is `self.successes`: each environment's in-progress episode count. This differs from the ordinary `successes`, `mean_successes`, and `success_ratio` TensorBoard summaries, which use `prev_episode_successes` from the last completed episode. `true_objective_mean` averages the current score over all 12,288 environments and `true_objective_max` takes their maximum. Consequently, `successes_max` can be nonzero while `true_objective_max` is zero if a previous episode succeeded but no current episode has yet succeeded.

The tolerance curriculum is eligible to reduce `tau` by multiplying it by 0.9 every 3,000 vector-environment control steps, but only when mean completed-episode successes is at least three. Both live compact policies still logged `tau=0.075` at inspection, so `progress=0` and `true_objective=0.01*s`. At frame 653,918,208 the gains run logged mean `0.000004883` and max `0.01`, meaning some current environment had one success; at frame 102,039,552 the adapters-only run logged both as zero even though its previous-episode `successes_max` was one.

## Deterministic trajectory evaluation

Milestone evaluation is not currently written into the training TensorBoard event files. It is stored under `evals/connectome/` as per-case `eval.json` and aggregate `summary.json` files. The evaluator runs the Gaussian mean action, counts demonstrated trajectory waypoints reached, and reports `task_progress_pct = 100 * successful_waypoints / total_waypoints`.

The current deployment contract is coefficient ID 50, the block-0 conditioned policy. This is inherited from the official SimToolReal `deployment/rl_player.py`, and both real deployment and this repository's video evaluator pass through the same local `RlPlayer`; videos therefore already show the deterministic mean of the currently deployable policy member. If a different coefficient ID is selected later, it must be changed as one shared deployment/evaluation contract rather than only in video generation. Selection among IDs should use a validation cohort, leaving the final evaluation-object cohort for unbiased reporting.

The [2026-09-15 system audit](../analyses/system-audit-2026-09-15.md) makes this distinction experimentally important: the strongest training-return run has zero block-0 completed goals in its recent tail, while ordinary return describes block 5. Across 17 clipped-policy milestones through 4.25B, mean deterministic Task Progress was 0.74074% except for 1.48148% at 2.75B. The nine 1,952-tanh and 18 fresh 262-tanh milestones were all 0.74074%. Evaluate all six IDs on the same validation cohort before concluding that ID-0 training improvements cannot transfer to deterministic control; current artifacts do not establish that another ID succeeds. None of the main histories had advanced the initial tolerance curriculum at that snapshot.

The evaluation YAML's `success_tolerance_m` is passed to the environment as the unscaled base tolerance. With the inherited `keypointScale: 1.5`, configured values 0.02 and 0.01 produce implemented maximum-keypoint thresholds of 0.03 m and 0.015 m respectively. Existing artifact labels call these the paper 2 cm and repository 1 cm settings because those are their configured base tolerances; consumers should record the 1.5 scale when reporting the actual threshold used by code.

For learning-curve diagnosis, plot `success_ratio` together with `scalars/success_tolerance` and inspect per-block curves for exploration effects. For final policy comparison, use matched deterministic Task Progress evaluations rather than treating the training success curves as evaluation success rates.

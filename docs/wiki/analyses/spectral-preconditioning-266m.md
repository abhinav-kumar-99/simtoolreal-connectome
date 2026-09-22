# Spectral preconditioning at 266M frames

On the live `00_spectral_r4_a-0p5_k4_seed42` run, float32 spectral preconditioning is changing LoRA gradients every update, while the effective recurrent spectrum is still the anatomical operator.

Last updated: 2026-09-21

Related: [Adaptation](../concepts/connectome-adaptation.md), [Experiment workflow](../workflows/connectome-experiments.md)

## Source

TensorBoard run `00_spectral_r4_a-0p5_k4_seed42` on port 6008, event file `events.out.tfevents.1790029767.armageddon`, through frame 266,207,232. The run is rank-4 signed LoRA, `learn_dynamics: true`, `alpha: -0.5`, floor ratio `1e-3`, refresh every update. `spectral/` has 14 logged points; `spectral_preconditioning/` has about 1,355.

## Evidence

Preconditioning is on and is not the identity. `alpha` stays `-0.5`. Multiplier mean stays `1`. Multiplier min/max/std stay about `0.0596` / `1.884` / `0.678`. `sigma_max` stays `3.644`, `sigma_min` stays `0`, and `sigma_floor` stays `0.00364`. Target-gradient cosine stays about `0.76`. Source-gradient cosine stays about `0.61–0.66`. The target gradient norm ratio falls from `0.37` to about `0.15` by frame 13M and stays there. The source ratio declines from `0.56` to `0.30` by frame 266M.

The effective spectrum is not moving. Baseline `sigma_max` is `3.64342`. Effective `sigma_max` stays inside `3.642–3.644`, a ratio within about `0.04%` of 1. The spectral-radius ratio drops from `1.000` to `0.992` by frame 19M and then sits near `0.9915`. Top singular values are flat.

Signed LoRA has moved a little. `adaptation/synaptic_delta_rms` grows from `8e-6` to `0.0059` at frame 13M, `0.0141` at frame 133M, and `0.0145` at frame 266M. Relative edge scale extremes reach about `0.80` and `1.19`. `losses/synaptic_plasticity_reg` tracks mean `Delta^2` and is `2.1e-4` at the last point.

`rewards/step` starts at `108`, falls to `28` by frame 14M, and is `103` at frame 266M. That is early-training recovery, not a task result.

## Current relevance

The preconditioner is a nearly fixed filter of the initial MaleCNS spectrum. Rank-4 relative updates at `1.4%` RMS do not move the singular spectrum, so refreshing the SVD every update repeats the same geometry. Whether that fixed geometry helps dexterity is not identifiable from these tags.

# Full-CNS visual reservoir performance

Inference-only kernels and larger environment batches are measured against the same 64x36, render-every-four, nine-update tanh policy.

Last updated: 2026-09-16

Related: [Visual reservoir](../concepts/visual-reservoir.md), [Sparse backends](sparse-backends.md)

## Implementation and correctness boundary

`connectome.backend_options.frozen_inference: true` selects reusable neuron-major buffers and suppresses the recurrent-input and preactivation writes needed only for backward. It requires the frozen cached tanh reservoir and `triton_fused`; ordinary trainable/adapted policies keep the existing autograd path. Returned hidden states are copied so later calls cannot overwrite caller-owned recurrent states. Device moves and checkpoint loads invalidate the runner cache. `cuda_graph: true` optionally captures only the fixed-shape neural passes, not camera rendering, input encoding, simulation or PPO.

`task.env.policyVision.fusedLuminance` reuses a stacked RGBA buffer and fuses channel conversion and luminance reduction. `connectome.fixed_input_encoder.retina.fused` fuses bilinear image sampling, contrast coding and sensory scatter. These are separate stages: the camera still exposes full grayscale observations. There is no learned visual adapter, precision reduction, neural-update reduction or image-rate change. `simtoolreal_shared/vision_ops.py` is an imported CUDA helper, not an entrypoint. Standard encoders remain the default for existing profiles.

## Measurements

Raw artifacts live under `profiles/connectome/visual_optimizations/`. The GPU-1 visual trainer and watcher were stopped for serial tests, while the independent GPU-0 Gaussian trainer was preserved. CUDA-event medians measure the full 165,122-cell, nine-update recurrence after warmup:

| Environments | Original (ms) | Frozen buffers (ms) | Frozen + capture (ms) |
| --- | ---: | ---: | ---: |
| 384 | 45.46 | 41.94 | 41.96 |
| 768 | 91.88 | 84.54 | 85.29 |
| 1,536 | 186.09 | 171.02 | 171.71 |

All recurrent comparisons had zero maximum difference in these tests, including changed inputs and partial environment resets. Image kernels passed FP32 tolerance checks: luminance `atol=2e-7, rtol=2e-6`; retina `atol=1e-5, rtol=2e-5`. Luminance fell from .094 to .035 ms and retinal encoding from .117 to .057 ms at batch 384. Those fractions of a millisecond do not establish a meaningful end-to-end gain.

With synchronized environment-only timing at batch 384, rendered steps averaged 222.8 ms and cached-image steps 23.2 ms; the separate fused run measured 211.9 and 19.3 ms. Each is only 24 measured steps after eight warmups. The difference between those runs is much larger than the image-kernel savings and must not be attributed entirely to fusion. Synchronization changes overlap compared with ordinary training. The camera helper itself averaged 99.9/90.2 ms, but excludes later image-access waits; it is not the complete camera cost.

Training cases run 12 actual PPO epochs and reload the resulting checkpoint. Medians discard the first two epochs. Batch changes retain six equally sized SAPG blocks, horizon 16 and minibatches of four times the environment count. This preserves update geometry but changes samples per update and is not a matched learning comparison. Short speed tests do not demonstrate task learning.

| Training case | Median total FPS |
| --- | ---: |
| Original, 384 environments | 3,252.5 |
| Frozen buffers, 384 | 3,349.0 |
| Frozen + capture, 384 | 3,321.5 |
| Frozen + capture + image fusion, 384 | 3,324.0 |
| All options, 768 | 3,693.5 |
| All options, 1,536 | 3,755.0 |
| Frozen + image fusion, no capture, 768 | 3,688.0 |

All seven cases completed their 12 epochs and finite-action deployment reload. The selected production setup is 768 environments, frozen buffers, fused image operations and no capture. It is 13.4% faster than the baseline in this short sweep. Capture showed no material benefit; 1,536 buys only 1.8% more throughput than the selected setup while doubling reservoir-state and camera counts. Neither tiny capture differences nor image-fusion gains are established beyond noise by one short run per case.

The combined network, retinal and configuration regression suite passed 127 tests, including ordinary autograd backend tests and new frozen/captured rollout, cache invalidation and fixed-image equivalence tests. The optimized two-environment visual audit passed independent-camera, nonblank-image and goal-image-leakage assertions. The 16-step dark/light probe reached maximum motor delta `2.335e-5` and mean `2.233e-6`; this remains a weak but measurable visual signal, not demonstrated visual task competence. The raw 64x36 image was visually inspected.

`ppo_full_cns_tanh_vision_fast_100b.yaml` selects six 128-environment SAPG blocks and 3,072-sample actor/critic minibatches. Full checkpoint saves occur every 100 epochs, retaining the previous approximately 1.23M-frame interval. It resumes epoch 318/frame 1,953,792 from the previous run's best full checkpoint using `resume_training_state`: actor, critic, both optimizers, normalization statistics and counters are retained; simulator episodes and recurrent rollout states reset. The newer 2,002,944-frame milestone is inference-only and cannot preserve both optimizers/critic weights. Existing run artifacts are not overwritten. Gaussian/SAPG/KL settings, 64x36 camera at 15 Hz, all 165,122 neurons, nine updates at 60 Hz, goals and proprioception are unchanged.

## Reproduction

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CC=/usr/bin/gcc CXX=/usr/bin/g++ .venv/bin/python scripts/benchmark_visual_reservoir.py --config configs/connectome/profiling/visual_optimizations.yaml
CUDA_VISIBLE_DEVICES=1 CC=/usr/bin/gcc CXX=/usr/bin/g++ .venv/bin/python scripts/audit_visual_reservoir.py --config configs/connectome/full_cns_visual_fast_audit.yaml
.venv/bin/python scripts/run_connectome_suite.py --config configs/connectome/suites/ppo_full_cns_tanh_vision_fast_100b.yaml
.venv/bin/python scripts/run_connectome_milestone_evaluation.py --config configs/connectome/evaluation/ppo_full_cns_tanh_vision_fast_100b_milestones.yaml
```

The entrypoint runs isolated kernel, environment and training subprocesses and emits raw logs, numerical checks, JSON timings, suite checkpoints and a `summary.json`. The YAML owns `cases`, `batches`, `warmup`, `iterations`, `steps`, `epochs`, `discard_epochs` (default two), artifact/config paths and `output_directory`. `resume: true` skips completed cases; use a fresh output directory for a new timing trial. The base suite assigns physical GPU 1 to trainers; standalone kernel/environment cases use `gpu` (default 1). Do not run this contention-sensitive sweep alongside a live GPU-1 trainer or evaluation watcher. `run_connectome_suite.py` remains the normal training entrypoint; the benchmark constructs short suite YAMLs rather than exposing training knobs through CLI flags.

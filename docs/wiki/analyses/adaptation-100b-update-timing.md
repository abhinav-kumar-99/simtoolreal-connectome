# 100-Billion-Step Gains Update-Timing Comparison

Two neuron-gains policies isolate optimizer-update timing over a strict 100-billion-environment-step budget. Both actors learn the same input/output adapters and topology-preserving neuron gains, use the fused Triton recurrence, seed 42, released-checkpoint exploration and perturbation settings, and one independent physical RTX 4090 per policy. They do not train the original LSTM.

Last updated: 2026-09-13

Related: [Billion-step adaptation run](adaptation-1b-run.md), [Experiment workflow](../workflows/connectome-experiments.md)

## Training contract

The YAML entry point is `configs/connectome/suites/adaptation_100b_gains_update_timing.yaml`. The two fully composed Hydra configurations differ only in rollout accumulation, logical/physical minibatch sizes, and the epoch count needed to match total environment exposure:

| Case | GPU | Fresh frames per update phase | Logical minibatch | Physical microbatch | Phases | Scheduled actor and critic calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `gains_new_update_timing` | 0 | 393,216 | 98,304 | 24,576 | 254,313 | 2,034,504 each |
| `gains_old_update_timing` | 1 | 196,608 | 49,152 | full logical batch | 508,626 | 4,069,008 each |

Each phase runs four logical minibatches for each of two PPO mini-epochs. The new timing therefore schedules one optimizer call per 49,152 fresh frames, matching the released policy's update density through two accumulated `12,288 x 16` rollouts. The old timing schedules one call per 24,576 fresh frames, matching the earlier stopped exploratory jobs. The new timing retains 24,576-sample physical gradient chunks because larger chunks already OOMed on both 24 GB GPUs; accumulation preserves the 98,304-sample logical optimizer batch.

Both cases execute 99,999,940,608 environment steps, 59,392 below the ceiling. A further complete phase would exceed 100 billion. As with previous runs, scheduled mixed-precision actor calls can exceed applied Adam steps if `GradScaler` suppresses a non-finite update; applied counts must be measured from checkpoints rather than assumed.

## Milestone evaluation

Training writes an atomic, inference-only snapshot whenever a completed update phase first crosses a 250-million-frame target. The snapshot includes the actor and normalization state but omits optimizer, rollout, and simulator state. This keeps 800 milestone files small enough for the available disk while ordinary rolling/final checkpoints retain complete recovery state.

Because weights change only after a complete phase, a snapshot's actual frame can follow its nominal target by at most 392,832 frames for the new timing or 196,224 for the old timing. The filename and status manifest record both values. The final nominal 100-billion target points to the last legal phase at 99,999,940,608 frames.

`scripts/run_connectome_milestone_evaluation.py` watches the checkpoint directories, resumes from `milestone_status.json`, retries failed cells, and delegates each checkpoint to the existing evaluation parent. Its contract is `configs/connectome/evaluation/adaptation_100b_gains_milestones.yaml`. Every policy/milestone gets one mean-action episode for `sharpie_marker/write_c`, `flat_eraser/wipe_smile`, and `flat_spatula/flip_over`, producing three H.264 videos at 800x450 and 20 FPS plus paper Task Progress and raw/shaped reward results. The worker rejects sampled-action video collection.

The end-to-end smoke contracts are `configs/connectome/suites/milestone_checkpoint_smoke.yaml` and `configs/connectome/evaluation/milestone_checkpoint_smoke.yaml`. The smoke produced two 1.9 MB inference snapshots, reloaded both with actor normalization state, and generated one nonempty mean-action 800x450 video per snapshot. Ordinary smoke recovery checkpoints were 19 MB because they retained optimizer and environment state.

## Operations

Run training from the repository root:

```bash
.venv/bin/python scripts/run_connectome_suite.py \
  --config configs/connectome/suites/adaptation_100b_gains_update_timing.yaml
```

Run the persistent evaluator in a separate process:

```bash
.venv/bin/python scripts/run_connectome_milestone_evaluation.py \
  --config configs/connectome/evaluation/adaptation_100b_gains_milestones.yaml
```

The training YAML owns both profiles, GPU assignment, update geometry, 100B cap, checkpoint interval, released perturbations, seed, and output roots. The evaluation YAML owns policy-to-GPU mapping, watcher cadence, the three fixed eval cases, mean-action selection, paper tolerance, resolution, frame sampling, and output root. The helper module `simtoolreal_shared/milestone_checkpoints.py` centralizes target scheduling and filename parsing; it is imported by training and evaluation rather than launched directly.

Generated training artifacts live under `train_dir/connectome/adaptation_100b_gains_update_timing/`. Evaluations live under `evals/connectome/adaptation_100b_gains_update_timing_milestones/`. Completion requires both training children to exit successfully, final recovery checkpoints to pass deployment reload, and all 400 milestones per policy to appear in the evaluator status manifest with three videos each.

## Live launch

The two training children were launched at approximately 2026-09-13 22:31 UTC in tmux session `connectome-100b`. The persistent video watcher runs in `connectome-100b-eval`, and TensorBoard runs in `connectome-100b-tensorboard` on port 6008 with a 10,000-scalar display reservoir. Raw TensorBoard Step values remain comparable environment-frame coordinates; the UI will still downsample the complete 254,313/508,626-point histories.

Both resolved configurations report `operator_backend: triton_fused`, `weight_mode: neuron_gains`, and the intended update geometries. Initial steady epochs were approximately 4.1--4.5 seconds for new timing and 1.9--2.1 seconds for old timing, projecting roughly 12--13 days if throughput remains stable. At that rate the first 250-million-frame snapshots and videos should arrive after roughly 45 minutes. These are launch estimates, not completion evidence.

# Approved 1,952-neuron training runs

The exact path-plus-sensory-premotor candidate was launched as matched neuron-gains and adapters-only policies with old optimizer timing; gains failed numerically and adapters-only was stopped after its actor updates became invalid.

Last updated: 2026-09-14

Related: [Selection evidence](front-leg-pathway-coverage.md), [Timing comparison](adaptation-100b-update-timing.md), [Experiment workflow](../workflows/connectome-experiments.md)

## Exact circuit and interfaces

The user approved **1,952**, without the additional feedback/coordination candidates discussed afterward. Preparation reproduces the 1,285 seed union, the 1,596-cell shortest-path footprint, and all 356 additional direct sensory-to-front-motor intermediates. No count quota, degree ranking, additional feedback cells, or isolate pruning is applied. The result has 33,720 directed edges at >=5 contacts and 44 isolated cells. The broader 4,778 candidate remains a reference, not the training graph.

`configs/connectome/malecns_1952.yaml` pins source hashes, selection masks, expected stage counts, and SHA256 of sorted little-endian int64 body IDs (`d6e3d907840e3f88d17b2c3807e9387c625efd9f3508027001725f410e7c5abc`). A differing shortest-path tie-break result fails identity validation rather than silently changing the candidate. SciPy 1.10.1 reproduced the approved set.

Population interfaces explicitly use the original selection masks: 384 proprioceptive/tactile sensory cells receive the 128 robot sensory features; 157 selected descending cells receive 12 goal features plus the 32-dimensional SAPG embedding; 135 front-leg motor cells feed the 29-action readout. Other retained cells participate through recurrence, not additional direct adapter ports. Dense adapters remain bias-free. The observation vector is unchanged and does not gain true tactile measurements merely by including tactile neurons.

Public unsigned reference counts are signed by source consensus neurotransmitter, with the existing model convention of ACh positive and all other labels negative. The manifest records 1,019 ACh, 604 GABA, 272 glutamate, 53 unclear and four missing labels. Unknown-label signs and glutamatergic inhibition are model assumptions, not per-cell physiological validation. Reference edges retain the audit's explicit VNC ROI/confidence scope; this is not an exact induced subset of the original upstream signed CSV. Global spectral normalization gives raw radius 493.6887786 and scale 0.0020255676113113096, targeting radius one before learned gains.

The artifact is `data/connectomes/processed/malecns_1952/biological.npz`, SHA256 `bdd88d94570b918634ed165486b80915c5fc3ff3bb1e472b4f8a9ee14b6d38e2`. Its directory includes `manifest.json`, annotated `neurons.csv` with stage membership, and unsigned selected `edges.csv.gz`. Preparation refuses to replace an existing different NPZ. The original 4,310 artifact is untouched.

## Training and default timing

`SimToolRealConnectome1952GainsSAPG` inherits adapters plus neuron-gain adaptation, frozen leak/bias dynamics, and fused Triton recurrence. The new job starts fresh at seed 42 on physical GPU 0; it does not transfer the original graph's checkpoint. Task reward, perturbations, exploration, horizon/sequence length 16, and budget match the existing old-timing run.

`SimToolRealConnectome1952AdaptersSAPG` uses the same artifact, interfaces, Triton recurrence, seed, environment contract, and old update timing on physical GPU 1, but freezes incoming/outgoing gains at one as well as leak and bias. Its recurrent operator therefore remains exactly the prepared spectrally normalized matrix while only adapters, heads, SAPG embeddings, and action log standard deviations learn.

- 12,288 environments, six blocks of 2,048.
- One rollout per phase: 196,608 fresh frames; `rollout_accumulation_steps: 1`.
- Actor and critic logical/physical minibatches both 49,152, with two mini-epochs: eight scheduled optimizer calls each per phase.
- 508,626 phases yield 99,999,940,608 frames, below the 100-billion cap.
- Inference milestone every 250 million frames; ordinary recovery saves every 3,000 epochs, best-save eligibility after 100 epochs.

The shared `SimToolRealConnectomeSAPG.yaml` now defaults to accumulation one and 49,152 logical minibatches. Physical microbatch defaults interpolate their corresponding logical minibatch, so small smoke overrides still compose correctly. Historical suites retain their explicit timing overrides; those are opt-in experiment contracts, not the new profile default. The launch YAML explicitly pins the full old environment/update geometry.

## Validation and launch evidence

Twenty-six focused tests passed across compact selection, original graph preparation, profile/suite configuration and front-leg auditing. Tests check seed/isolate retention, graph direction, complete two-edge motif inclusion, absence of padding, input-order stability, explicit sign policies, exact live body IDs/CSR magnitudes/population masks, and resolved old timing.

The full-size smoke used 12,288 environments and the intended 49,152-sample physical minibatches. It completed two phases and 393,216 frames in 33.04 seconds of child training, then reloaded the final recovery checkpoint with finite `(1, 29)` deployment actions. All saved model tensors were finite; all actor Adam states had step 16, and both gain tensors had changed from zero initialization. Evidence: `train_dir/connectome/adaptation_1952_smoke/suite_results.json` and its per-run `verification.json`. This is an execution gate, not evidence of learned dexterity.

The full run started around 2026-09-14 00:10 UTC (September 13 local time), in tmux `connectome-1952`: coordinator PID 3211542, training PID 3211659. At the initial check, TensorBoard served 18 actor/critic loss points through logged frame 3,342,336 with finite values. The old graph's GPU-1 training PID 3172351 remained alive. The stopped new-timing job's artifacts were preserved; its watcher was replaced by an old-only watcher, retaining the existing evaluation history. The compact watcher is in `connectome-1952-eval`, configured for three mean-action videos per 250M-frame milestone. Training is ongoing, not complete.

At the user's direction, the 4,310-cell GPU-1 trainer and its dedicated watcher were stopped after the training log reached frame 928,579,584. Its milestone/checkpoint/evaluation artifacts remain in place; the latest full recovery checkpoint is `last/model.pth` at frame 904,396,800 and the latest inference milestone is 750,059,520. No directory was removed or overwritten.

The fresh adapters-only replacement started in tmux `connectome-1952-adapters`: coordinator PID 3248406 and trainer PID 3248466. The milestone watcher is PID 3248583 in `connectome-1952-adapters-eval`. The resolved configuration confirms `weight_mode: adapters_only`, `learn_dynamics: false`, `operator_backend: triton_fused`, the 1,952/33,720 graph, 12,288 environments, 49,152 minibatches, one rollout per phase, seed 42, and the 100B cap. Its first audit found 160 finite scalar tags and `rewards/step=34.874` at frame 1,769,472. TensorBoard 6008 discovered the new run under the shared parent log directory.

## 2026-09-14 numerical-failure status

The gains trainer exited nonzero after 11,574 phases and 2,275,344,384 logged frames. The immediate exception was `RuntimeError: normal expects all elements of std >= 0.0` while sampling the rollout action. Its last complete recovery checkpoint is epoch 11,400/frame 2,241,331,200 with 91,164 applied actor Adam steps; the last inference milestone is 2,250,178,560 and all nine available milestones have their three mean-action videos. The standard SAPG configuration uses an identity `sigma_activation`, so its coefficient-conditioned action standard deviations are not positivity constrained. Several learned sigma entries had crossed below zero before the exception.

The adapters-only trainer remained process-alive on GPU 1 after `losses/a_loss` and `info/kl` became `NaN` at approximately frame 3.19 billion. At the user's direction, its trainer and dedicated watcher were interrupted at 2026-09-14 07:20 EDT. The final log coordinate is epoch 17,066/frame 3,355,115,520. Its last complete recovery checkpoint is epoch 17,000/frame 3,342,336,000 with 129,745 applied actor steps versus 136,000 scheduled calls, showing 6,255 mixed-precision-suppressed calls. The applied count had not increased since the earlier epoch-16,800 checkpoint. Its first 13 milestones through 3,250,126,848 have all 39 requested mean-action videos. All artifacts are preserved.

TensorBoard remains on port 6008. Both compact trainers are stopped; the failed gains watcher remains alive waiting for checkpoints that will not arrive, while the adapters watcher is stopped. A separately launched eligibility trainer appeared on physical GPU 0 after the status audit and was not touched by the adapters shutdown.

## Commands and configuration ownership

Run from the repository root. Both compact processes are stopped: **do not rerun either launch command into its populated output directory**. `on_existing: fail` protects those directories; a repaired experiment needs a new suite name and output directory or an explicit, validated resume contract.

```bash
# Prepare/verify the exact graph only (does not train).
.venv/bin/python scripts/prepare_malecns_connectome.py --config configs/connectome/malecns_1952.yaml

# Full-size two-phase training and checkpoint reload gate.
.venv/bin/python scripts/run_connectome_suite.py --config configs/connectome/suites/adaptation_1952_smoke.yaml

# Historical gains launch command; the recorded process failed and is not running.
.venv/bin/python scripts/run_connectome_suite.py --config configs/connectome/suites/adaptation_1952_100b.yaml

# Persistent gains milestone evaluator, already launched separately.
.venv/bin/python scripts/run_connectome_milestone_evaluation.py --config configs/connectome/evaluation/adaptation_1952_milestones.yaml

# Historical adapters-only launch command; the recorded process was stopped after invalid updates.
.venv/bin/python scripts/run_connectome_suite.py --config configs/connectome/suites/adaptation_1952_adapters_100b.yaml

# Persistent adapters-only milestone evaluator, already launched separately.
.venv/bin/python scripts/run_connectome_milestone_evaluation.py --config configs/connectome/evaluation/adaptation_1952_adapters_milestones.yaml
```

The preparation entrypoint dispatches by `preparation_kind`. Its new helper `simtoolreal_shared/compact_connectome.py` performs hash verification, path/motif selection, population indexing, signing, normalization, and artifact/manifest writing; import it rather than launching it directly. Its inputs are the pinned completed audit tables and public neurotransmitter Feather file. Missing sources must be restored from the pinned snapshot; rerunning the audit may change the compressed-file hash even with identical logical rows, so do not bypass the hash check without verifying provenance and selected IDs.

The suite entrypoint handles preparation, device isolation, resolved YAML, logs, checkpoints, and post-exit deployment verification. Important training YAML keys are `train_profiles`, `gpu_assignments`, `num_envs`, `sapg_block_size`, `rollout_accumulation_steps`, logical/physical minibatch sizes, `epochs`, `max_frames`, milestone interval and `output_directory`. The evaluation entrypoint watches snapshots and invokes the existing video evaluator; its YAML owns paths, GPU, polling interval, mean-action cases, and video settings. The stopped large run used `configs/connectome/evaluation/adaptation_100b_old_only_milestones.yaml`; that watcher is no longer active.

TensorBoard remains at **http://localhost:6008**, watching `train_dir/connectome/adaptation_100b_gains_update_timing`. The compact gains and adapters-only runs are nested under `compact_1952/` and `compact_1952_adapters/`; the API confirmed both are visible with 160 scalar tags. All stopped histories remain visible. Training/evaluation artifacts are ignored and are not committed.

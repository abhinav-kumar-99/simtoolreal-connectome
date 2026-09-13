# Billion-Step Adaptation Run

The adapters-only and neuron-gains policies use the validated 12,288-environment single-GPU geometry, the default fused Triton recurrence, and otherwise retain the capped pilot's SAPG and task settings. They run concurrently, with one independent SAPG child on each physical GPU.

Last updated: 2026-09-13

Related: [Experiment workflow](../workflows/connectome-experiments.md), [One-million-step pilot](adaptation-1m-pilot.md)

## Training contract

The YAML entry point is `configs/connectome/suites/adaptation_1b.yaml`. Its ordered case-to-device mapping is:

| Policy | Train profile | Physical GPU |
| --- | --- | ---: |
| adapters-only | `SimToolRealConnectomeSAPG` | 0 |
| neuron-gains | `SimToolRealConnectomeGainsSAPG` | 1 |

Each epoch collects `12,288 x 16 = 196,608` fresh environment transitions. The run uses 5,086 complete epochs, producing 999,948,288 transitions below the strict 1,000,000,000-step cap; another complete epoch would exceed it. With two mini-epochs and four minibatches, each actor is scheduled for 40,688 optimizer updates.

Both jobs use seed 42, six 2,048-environment SAPG blocks, 49,152-sample actor and critic minibatches, and the same dense-reward and object-randomization overrides as the completed one-million-step pilot. Generated checkpoints, event files, resolved configs, timing metadata, and logs live under `train_dir/connectome/adaptation_1b_triton_12k/` and remain ignored.

## Operations

Launch the training parent from the repository root:

```bash
.venv/bin/python scripts/run_connectome_suite.py --config configs/connectome/suites/adaptation_1b.yaml
```

The suite launcher prepares/verifies the pinned graph, assigns the two ordered cases to separate GPU queues, launches ordinary single-GPU SAPG children, and verifies the final checkpoints. It writes a final suite summary only after both children exit successfully.

TensorBoard watches the suite directory:

```bash
.venv/bin/python -m tensorboard.main \
  --logdir train_dir/connectome/adaptation_1b_triton_12k \
  --host 0.0.0.0 --port 6007
```

Port 6007 is used on the development host because port 6006 is occupied. The operational state and final measurements belong in the project log; a launched process is not a completed experiment.

## Live launch

Both jobs were launched at 2026-09-13 12:17 EDT in the durable tmux session `connectome-1b`; TensorBoard was launched in `connectome-tensorboard`. The adapters-only and neuron-gains children both completed multiple epochs, occupied approximately 22.1 GB on GPU 0 and 21.7 GB on GPU 1, respectively, and emitted event files that TensorBoard discovered. Initial epoch times were approximately 1.9--2.1 seconds, projecting about 2.7--3.0 hours to the requested epoch count if throughput remains stable. This is launch evidence only; final checkpoint verification and elapsed timing remain pending.

For external progress references and important configuration mismatches, see [SimToolReal training references](../sources/summaries/simtoolreal-training-references.md). In particular, matching by environment steps does not match optimizer updates: this 12,288-environment run performs twice as many actor updates per transition as the published 24,576-environment shape.

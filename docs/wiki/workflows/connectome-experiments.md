# Connectome Experiment Workflow

Experiments are owned by YAML contracts and proceed through data, profile, smoke, and full-training gates.

Last updated: 2026-09-13

Related: [Overview](../overview.md), [Actor](../concepts/connectome-actor.md)

## Gates

1. Prepare and validate the pinned graph artifacts.
2. Run unit tests and sparse-versus-dense actor checks.
3. Profile LSTM and connectome forward/backward resource use.
4. Run a two-epoch, 384-environment Isaac Gym SAPG smoke test and reload its checkpoint.
5. Launch matched multi-seed training only after the smoke artifacts are accepted.


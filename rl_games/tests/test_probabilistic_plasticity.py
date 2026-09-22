from __future__ import annotations

from copy import deepcopy

import gym
import numpy as np
import pytest
import torch
from scipy import sparse

from rl_games.algos_torch.connectome_network_builder import ConnectomeBuilder
from rl_games.common.custom_utils import shuffle_batch, swap_and_flatten01
from rl_games.common.datasets import PPODataset
from rl_games.common.experience import ExperienceBuffer


@pytest.fixture()
def probabilistic_artifact(tmp_path):
    sources = np.asarray([0, 0, 1, 2, 2, 3, 4, 5], dtype=np.int64)
    destinations = np.asarray([2, 4, 3, 4, 5, 6, 5, 6], dtype=np.int64)
    values = np.asarray(
        [0.2, -0.1, 0.3, 0.4, -0.2, 0.5, 0.1, -0.4],
        dtype=np.float32,
    )
    matrix = sparse.coo_matrix(
        (values, (destinations, sources)), shape=(7, 7)
    ).tocsr()
    path = tmp_path / "probabilistic_connectome.npz"
    np.savez_compressed(
        path,
        schema_version=np.asarray([1], dtype=np.int64),
        crow_indices=matrix.indptr.astype(np.int64),
        col_indices=matrix.indices.astype(np.int64),
        values=matrix.data.astype(np.float32),
        raw_values=matrix.data.astype(np.float32),
        body_ids=np.arange(7, dtype=np.int64),
        sensory_indices=np.asarray([0, 1], dtype=np.int64),
        front_proprioceptors_indices=np.asarray([0], dtype=np.int64),
        front_tactile_indices=np.asarray([1], dtype=np.int64),
        descending_indices=np.asarray([2, 3], dtype=np.int64),
        motor_indices=np.asarray([4, 5], dtype=np.int64),
    )
    return path


def _params(
    artifact,
    *,
    rank=2,
    budget=5,
    group_size=1,
    preconditioning=True,
    inference_mode="sampled",
):
    return {
        "name": "connectome_actor_critic",
        "connectome": {
            "artifact_path": str(artifact),
            "graph_variant": "test",
            "operator_backend": "triton_fused",
            "dtype": "float32",
            "expected": {
                "neurons": 7,
                "edges": 8,
                "sensory_neurons": 2,
                "descending_neurons": 2,
                "motor_neurons": 2,
            },
            "observations": {
                "policy_size": 5,
                "sensory_size": 2,
                "goal_size": 3,
                "sensory_ranges": [[0, 2]],
                "goal_ranges": [[2, 5]],
            },
            "sapg_embedding_size": 4,
            "population_adapters": {"bias": False},
            "interface_projections": {
                "architecture": "linear",
                "hidden_size": 16,
                "activation": "elu",
            },
            "dynamics": {
                "activation": "tanh",
                "beta": 0.9,
                "gain_bounds": [0.25, 4.0],
                "initial_gain": 1.0,
                "initial_leak": 0.5,
                "initial_bias": 0.0,
                "neural_updates": 3,
            },
            "adaptation": {
                "weight_mode": "latent_probabilistic",
                "learn_dynamics": True,
                "rank": rank,
                "probabilistic": {
                    "topology_prior_error_rate": 0.01,
                    "candidate_nonedge_budget": budget,
                    "topology_sample_group_size": group_size,
                    "target_information_nats_per_neuron": 1.0,
                    "dual_lr": 0.1,
                    "inference_topology_mode": inference_mode,
                },
            },
            "spectral_monitoring": {"enabled": True, "interval": 1, "top_k": 3},
            "spectral_preconditioning": {
                "enabled": preconditioning,
                "alpha": -0.5,
                "singular_value_floor_ratio": 1.0e-3,
                "refresh_every_ppo_updates": 1,
            },
            "plasticity_monitoring": {"enabled": True, "interval": 1},
        },
        "space": {
            "continuous": {
                "mu_activation": "None",
                "sigma_activation": "None",
                "fixed_sigma": "coef_cond",
            }
        },
    }


def _build(artifact, **kwargs):
    builder = ConnectomeBuilder()
    builder.load(_params(artifact, **kwargs))
    return builder.build(
        "probabilistic",
        actions_num=2,
        input_shape=(6,),
        num_seqs=4,
        value_size=1,
        type="extra_param",
        coef_ids=torch.tensor([50.0, 0.0]),
        coef_id_idx=5,
    )


def _observations(batch, device="cpu"):
    observations = torch.randn(batch, 6, device=device)
    observations[:, 5] = torch.tensor(
        [50.0, 0.0] * ((batch + 1) // 2), device=device
    )[:batch]
    return observations


def test_baseline_recovery_layout_scales_and_information(probabilistic_artifact):
    network = _build(probabilistic_artifact, rank=2)
    assert network.plasticity_state.shape == (7, 11)
    assert torch.count_nonzero(network.plasticity_state) == 0
    assert not network.synaptic_post_anchor.requires_grad
    assert not network.topology_post_anchor.requires_grad
    assert network.log_tau.item() == 0.0
    assert set(name for name, _ in network.named_parameters()).isdisjoint(
        {"edge_u", "edge_v", "leak_raw", "recurrent_bias", "log_intrinsic_gain"}
    )

    torch.testing.assert_close(network.intrinsic_gains(), torch.ones(7))
    torch.testing.assert_close(network.leaks(), torch.full((7,), 0.5))
    torch.testing.assert_close(
        network.substep_leaks(),
        torch.full((7,), 1.0 - 0.5 ** (1.0 / 3.0)),
    )
    torch.testing.assert_close(network.recurrent_biases(), torch.zeros(7))

    readout = network.probabilistic_support_readout()
    observed = readout["is_original"]
    torch.testing.assert_close(
        readout["conditional_values"][observed],
        readout["baseline"][observed],
    )
    torch.testing.assert_close(
        readout["conditional_values"][~observed],
        torch.zeros_like(readout["conditional_values"][~observed]),
    )
    expected_prior = torch.where(
        observed,
        torch.full_like(readout["posterior_logits"], 0.99),
        torch.full_like(readout["posterior_logits"], 0.01),
    )
    torch.testing.assert_close(
        torch.sigmoid(readout["posterior_logits"]), expected_prior
    )
    latent, topology, total = network.probabilistic_information_estimate()
    assert latent.item() == pytest.approx(0.0, abs=1e-8)
    assert topology.item() == pytest.approx(0.0, abs=1e-6)
    assert total.item() == pytest.approx(0.0, abs=1e-6)

    rows = torch.repeat_interleave(
        torch.arange(network.neuron_count),
        network.crow_indices[1:] - network.crow_indices[:-1],
    )
    global_rms = network.recurrent_values.square().mean().sqrt()
    for neuron in range(network.neuron_count):
        incoming = network.recurrent_values[rows == neuron]
        expected = (
            incoming.square().mean().sqrt()
            if incoming.numel()
            else global_rms
        )
        torch.testing.assert_close(
            network.probabilistic_bias_scale[neuron], expected
        )


def test_pairwise_decoder_uses_post_pre_orientation(probabilistic_artifact):
    network = _build(probabilistic_artifact, rank=2)
    slices = network._latent_slices
    with torch.no_grad():
        network.synaptic_post_anchor.zero_()
        network.topology_post_anchor.zero_()
        network.synaptic_post_anchor[4] = torch.tensor([2.0, 0.0])
        network.topology_post_anchor[4] = torch.tensor([0.0, 3.0])
        network.plasticity_state[1, slices.synaptic_pre] = torch.tensor(
            [5.0, 0.0]
        )
        network.plasticity_state[1, slices.topology_pre] = torch.tensor(
            [0.0, 7.0]
        )
    forward_id = torch.tensor([4 * 7 + 1])
    reverse_id = torch.tensor([1 * 7 + 4])
    forward = network.probabilistic_pair_readout(forward_id)
    reverse = network.probabilistic_pair_readout(reverse_id)
    assert forward["synaptic_score"].item() == pytest.approx(
        10.0 / np.sqrt(2.0)
    )
    assert forward["topology_score"].item() == pytest.approx(
        21.0 / np.sqrt(2.0)
    )
    assert reverse["synaptic_score"].item() == pytest.approx(0.0)
    assert reverse["topology_score"].item() == pytest.approx(0.0)


def test_stateless_sampling_reproducibility_and_grouping(
    probabilistic_artifact,
):
    from rl_games.algos_torch.connectome_probabilistic_plasticity import (
        hard_bernoulli_gates,
        stateless_uniform,
    )

    edge_ids = torch.arange(49)
    logits = torch.zeros(49)
    seed_a = torch.tensor([11])
    seed_b = torch.tensor([12])
    first = hard_bernoulli_gates(logits, seed_a, edge_ids)
    torch.testing.assert_close(
        first, hard_bernoulli_gates(logits, seed_a, edge_ids)
    )
    assert not torch.equal(
        first, hard_bernoulli_gates(logits, seed_b, edge_ids)
    )
    torch.testing.assert_close(
        stateless_uniform(seed_a, edge_ids),
        stateless_uniform(seed_a, edge_ids),
    )

    independent = _build(probabilistic_artifact, group_size=1)
    independent.configure_probabilistic_runtime(42)
    independent_seeds = independent.new_topology_seeds(6, global_rank=0)
    assert independent_seeds.unique().numel() == 6
    rank_one = _build(probabilistic_artifact, group_size=1)
    rank_one.configure_probabilistic_runtime(42)
    assert not torch.equal(
        independent_seeds, rank_one.new_topology_seeds(6, global_rank=1)
    )

    grouped = _build(probabilistic_artifact, group_size=2)
    grouped.configure_probabilistic_runtime(42)
    grouped_seeds = grouped.new_topology_seeds(6, global_rank=0)
    assert torch.equal(grouped_seeds[::2], grouped_seeds[1::2])
    assert grouped_seeds[::2].unique().numel() == 3


def test_candidate_support_is_shared_unique_and_invalidates_caches(
    probabilistic_artifact,
):
    network = _build(probabilistic_artifact, budget=6)
    network.configure_probabilistic_runtime(9)
    old_candidates = network.probabilistic_candidate_nonedge_ids.clone()
    network._backend_graph = object()
    network._spectral_precond = {"sentinel": True}
    network.refresh_probabilistic_support()
    candidates = network.probabilistic_candidate_nonedge_ids
    assert candidates.numel() == candidates.unique().numel() == 6
    assert not torch.isin(
        candidates, network.probabilistic_original_edge_ids
    ).any()
    rows = candidates // network.neuron_count
    cols = candidates % network.neuron_count
    assert not torch.any(rows == cols)
    assert torch.isin(
        network.probabilistic_original_edge_ids,
        network.probabilistic_support_canonical_ids,
    ).all()
    assert network._backend_graph is None
    assert network._spectral_precond is None
    assert not torch.equal(old_candidates, candidates)

    environment_count = 32
    edge_count = network.probabilistic_support_canonical_ids.numel()
    for name, tensor in network.state_dict().items():
        assert tuple(tensor.shape) != (edge_count, environment_count), name
        assert "gate" not in name


def test_information_controller_tau_and_dual(probabilistic_artifact):
    from rl_games.algos_torch.connectome_probabilistic_plasticity import (
        bernoulli_kl_from_logits,
        gaussian_information,
    )

    network = _build(probabilistic_artifact)
    assert gaussian_information(
        network.plasticity_state, network.log_tau
    ).item() == pytest.approx(0.0)
    logits = torch.tensor([-4.0, 0.0, 3.0])
    torch.testing.assert_close(
        bernoulli_kl_from_logits(logits, logits), torch.zeros(3), atol=1e-7, rtol=0
    )
    start = network.information_dual_weight().item()
    network.update_information_dual(2.0)
    above = network.information_dual_weight().item()
    network.update_information_dual(0.0)
    below = network.information_dual_weight().item()
    assert above > start
    assert below < above
    assert below > 0.0

    with torch.no_grad():
        network.plasticity_state.fill_(0.2)
    latent = gaussian_information(network.plasticity_state, network.log_tau)
    latent.backward()
    assert network.log_tau.grad is not None
    assert torch.isfinite(network.log_tau.grad)


def test_expected_topology_is_rejected_as_runtime_inference_mode(
    probabilistic_artifact,
):
    with pytest.raises(ValueError, match="diagnostics-only"):
        _build(probabilistic_artifact, inference_mode="expected")


def test_spectral_preconditioning_targets_pairwise_slices_only(
    probabilistic_artifact,
):
    from rl_games.algos_torch.connectome_spectral_preconditioning import (
        apply_spectral_preconditioner,
    )

    network = _build(probabilistic_artifact, preconditioning=True)
    network.refresh_spectral_preconditioner()
    cache = network._spectral_precond
    raw = torch.randn_like(network.plasticity_state)
    network.plasticity_state.grad = raw.clone()
    intrinsic_before = raw[:, :3].clone()
    slices = network._latent_slices
    post_before = torch.cat(
        (
            raw[:, slices.synaptic_post],
            raw[:, slices.topology_post],
        ),
        dim=1,
    )
    pre_before = torch.cat(
        (
            raw[:, slices.synaptic_pre],
            raw[:, slices.topology_pre],
        ),
        dim=1,
    )
    expected_post = apply_spectral_preconditioner(
        post_before, cache["P"], cache["multipliers"]
    )
    expected_pre = apply_spectral_preconditioner(
        pre_before, cache["Q"], cache["multipliers"]
    )
    network.precondition_lora_gradients()
    grad = network.plasticity_state.grad
    torch.testing.assert_close(grad[:, :3], intrinsic_before)
    torch.testing.assert_close(
        torch.cat(
            (
                grad[:, slices.synaptic_post],
                grad[:, slices.topology_post],
            ),
            dim=1,
        ),
        expected_post,
    )
    torch.testing.assert_close(
        torch.cat(
            (
                grad[:, slices.synaptic_pre],
                grad[:, slices.topology_pre],
            ),
            dim=1,
        ),
        expected_pre,
    )
    assert network.log_tau.grad is None

    optimizer = torch.optim.Adam(network.parameters(), lr=1.0e-3)
    optimizer.step()
    state_before = deepcopy(optimizer.state[network.plasticity_state])
    pointer = network.plasticity_state.data_ptr()
    network.refresh_spectral_preconditioner()
    assert network.plasticity_state.data_ptr() == pointer
    for key in ("step", "exp_avg", "exp_avg_sq"):
        torch.testing.assert_close(
            optimizer.state[network.plasticity_state][key],
            state_before[key],
        )

    disabled = _build(probabilistic_artifact, preconditioning=False)
    disabled_raw = torch.randn_like(disabled.plasticity_state)
    disabled.plasticity_state.grad = disabled_raw.clone()
    disabled.precondition_lora_gradients()
    torch.testing.assert_close(disabled.plasticity_state.grad, disabled_raw)


def test_checkpoint_round_trip_includes_support_dual_and_counters(
    probabilistic_artifact,
):
    network = _build(probabilistic_artifact)
    network.configure_probabilistic_runtime(123)
    network.refresh_probabilistic_support()
    network.update_information_dual(2.0)
    network.new_topology_seeds(4, 0)
    with torch.no_grad():
        network.plasticity_state.normal_(std=0.01)
        network.log_tau.fill_(0.2)
    state = deepcopy(network.state_dict())
    required = {
        "plasticity_state",
        "log_tau",
        "information_dual_log_weight",
        "probabilistic_candidate_nonedge_ids",
        "probabilistic_support_canonical_ids",
        "probabilistic_kl_nonedge_ids",
        "probabilistic_candidate_refresh_counter",
        "probabilistic_topology_seed_counter",
        "probabilistic_update_counter",
    }
    assert required.issubset(state)
    assert not any("gate" in key for key in state)

    restored = _build(probabilistic_artifact)
    restored.load_state_dict(state)
    for key in required:
        torch.testing.assert_close(restored.state_dict()[key], state[key])
    assert restored._backend_graph is None
    assert restored._spectral_precond is None


def test_probabilistic_diagnostics_expose_required_tensorboard_tags(
    probabilistic_artifact,
):
    network = _build(probabilistic_artifact)
    network.configure_probabilistic_runtime(4)
    network.refresh_probabilistic_support()
    tags = network.compute_probabilistic_plasticity_diagnostics()
    required = {
        "probabilistic_plasticity/information/latent_nats",
        "probabilistic_plasticity/information/topology_nats",
        "probabilistic_plasticity/information/total_nats",
        "probabilistic_plasticity/information/nats_per_neuron",
        "probabilistic_plasticity/information/target_nats_per_neuron",
        "probabilistic_plasticity/information/dual_weight",
        "probabilistic_plasticity/information/tau",
        "probabilistic_plasticity/topology/original_edge_probability_mean",
        "probabilistic_plasticity/topology/nonedge_probability_mean",
        "probabilistic_plasticity/topology/original_edge_probability_p05",
        "probabilistic_plasticity/topology/original_edge_probability_p50",
        "probabilistic_plasticity/topology/original_edge_probability_p95",
        "probabilistic_plasticity/topology/nonedge_probability_p95",
        "probabilistic_plasticity/topology/sample_edge_count_mean",
        "probabilistic_plasticity/topology/sample_edge_count_std",
        "probabilistic_plasticity/topology/sample_removed_original_edges_mean",
        "probabilistic_plasticity/topology/sample_new_edges_mean",
        "probabilistic_plasticity/candidates/nonedge_budget",
        "probabilistic_plasticity/candidates/posterior_mass_fraction",
        "probabilistic_plasticity/candidates/turnover_fraction",
        "probabilistic_plasticity/synapse/conditional_weight_rms",
        "probabilistic_plasticity/synapse/original_edge_delta_rms",
        "probabilistic_plasticity/synapse/original_sign_flip_fraction",
        "probabilistic_plasticity/synapse/new_edge_weight_rms",
        "probabilistic_plasticity/latent/rms",
        "probabilistic_plasticity/latent/intrinsic_rms",
        "probabilistic_plasticity/latent/synaptic_rms",
        "probabilistic_plasticity/latent/topology_rms",
    }
    assert required.issubset(tags)
    assert all(np.isfinite(value) for value in tags.values())


def test_topology_seed_buffer_accumulation_shuffle_and_recurrent_batches():
    from rl_games.common.a2c_common import ContinuousA2CBase

    env_info = {
        "agents": 1,
        "action_space": gym.spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        ),
        "observation_space": gym.spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        ),
    }
    buffer = ExperienceBuffer(
        env_info,
        {
            "num_actors": 4,
            "horizon_length": 3,
            "has_central_value": False,
            "use_action_masks": False,
        },
        "cpu",
        aux_tensor_dict={
            "topology_seed": {"shape": (), "dtype": np.int64}
        },
    )
    seeds = torch.tensor([101, 202, 303, 404], dtype=torch.int64)
    for step in range(3):
        buffer.update_data("topology_seed", step, seeds)
    flattened = buffer.get_transformed_list(
        swap_and_flatten01, ["topology_seed"]
    )["topology_seed"]
    assert flattened.dtype == torch.int64
    assert torch.equal(flattened.reshape(4, 3), seeds[:, None].expand(4, 3))

    first_rollout = {
        "returns": torch.arange(12).reshape(12, 1).float(),
        "topology_seed": flattened.clone(),
        "played_frames": 12,
        "step_time": 1.0,
        "rnn_states": [torch.zeros(1, 4, 2)],
    }
    second_rollout = {
        "returns": torch.arange(12, 24).reshape(12, 1).float(),
        "topology_seed": flattened.clone() + 1000,
        "played_frames": 12,
        "step_time": 2.0,
        "rnn_states": [torch.ones(1, 4, 2)],
    }
    harness = ContinuousA2CBase.__new__(ContinuousA2CBase)
    batch = ContinuousA2CBase._concatenate_rollouts(
        harness, [first_rollout, second_rollout]
    )
    assert batch["played_frames"] == 24
    assert batch["step_time"] == 3.0
    assert torch.equal(batch["topology_seed"][:12], flattened)
    assert torch.equal(batch["topology_seed"][12:], flattened + 1000)
    assert batch["rnn_states"][0].shape[1] == 8
    shuffled = shuffle_batch(batch, 3)
    seed_matrix = shuffled["topology_seed"].reshape(8, 3)
    assert torch.all(seed_matrix == seed_matrix[:, :1])

    dataset = PPODataset(
        batch_size=24,
        minibatch_size=6,
        is_discrete=False,
        is_rnn=True,
        device="cpu",
        seq_length=3,
    )
    dataset.update_values_dict(
        {
            "returns": shuffled["returns"],
            "topology_seed": shuffled["topology_seed"],
            "rnn_states": shuffled["rnn_states"],
        }
    )
    reconstructed = []
    for index in range(len(dataset)):
        item = dataset[index]
        seed_matrix = item["topology_seed"].reshape(-1, 3)
        assert torch.all(seed_matrix == seed_matrix[:, :1])
        reconstructed.append(item["topology_seed"])
    assert torch.equal(torch.cat(reconstructed), shuffled["topology_seed"])


def test_leader_follower_duplication_preserves_source_topology_ids():
    from rl_games.common.custom_utils import filter_leader

    # Two policy blocks, two environments per block, three timesteps per env.
    original = torch.tensor(
        [10, 10, 10, 11, 11, 11, 20, 20, 20, 21, 21, 21],
        dtype=torch.int64,
    )
    repeated = torch.cat((original, original))
    filtered = filter_leader(
        repeated,
        orig_len=len(original),
        repeat_idxs=[0, 1],
        num_blocks=2,
    )
    assert torch.equal(filtered[: len(original)], original)
    # The added leader/follower block retains the source block's circuit IDs.
    assert torch.equal(filtered[len(original) :], original[:6])


def _probabilistic_ddp_worker(
    rank, world_size, init_path, artifact_path, result_directory
):
    import torch.distributed as dist

    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_path}",
        rank=rank,
        world_size=world_size,
    )
    try:
        torch.manual_seed(100 + rank)
        network = _build(
            artifact_path, budget=6, preconditioning=True
        )
        # Mirrors the trainer's initial rank-0 model-state broadcast.
        for tensor in network.state_dict().values():
            dist.broadcast(tensor, 0)
        network.configure_probabilistic_runtime(77)
        if rank == 0:
            candidates, kl_ids, stats = (
                network.propose_probabilistic_support()
            )
            stats_tensor = torch.tensor(
                [
                    stats["posterior_mass_fraction"],
                    stats["turnover_fraction"],
                ],
                dtype=torch.float64,
            )
        else:
            candidates = torch.empty_like(
                network.probabilistic_candidate_nonedge_ids
            )
            kl_ids = torch.empty_like(network.probabilistic_kl_nonedge_ids)
            stats_tensor = torch.empty(2, dtype=torch.float64)
        dist.broadcast(candidates, 0)
        dist.broadcast(kl_ids, 0)
        dist.broadcast(stats_tensor, 0)
        network.apply_probabilistic_support(
            candidates,
            kl_ids,
            posterior_mass_fraction=float(stats_tensor[0]),
            turnover_fraction=float(stats_tensor[1]),
        )

        _latent, _topology, total = network.probabilistic_full_information()
        per_neuron = total / network.neuron_count
        dist.all_reduce(per_neuron)
        per_neuron.div_(world_size)
        if rank == 0:
            network.update_information_dual(float(per_neuron))
        dist.broadcast(network.information_dual_log_weight, 0)

        if rank == 0:
            network.refresh_spectral_preconditioner()
            cache = network._spectral_precond
        else:
            n = network.neuron_count
            cache = {
                "P": torch.empty(n, n),
                "Q": torch.empty(n, n),
                "multipliers": torch.empty(n),
                "singular_values": torch.empty(n),
            }
        for key in ("P", "Q", "multipliers", "singular_values"):
            dist.broadcast(cache[key], 0)
        torch.save(
            {
                "support": network.probabilistic_support_canonical_ids,
                "kl": network.probabilistic_kl_nonedge_ids,
                "dual": network.information_dual_log_weight,
                "p": cache["P"],
                "q": cache["Q"],
            },
            f"{result_directory}/rank_{rank}.pt",
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not torch.distributed.is_available(), reason="torch.distributed unavailable"
)
def test_two_rank_support_dual_and_spectral_basis_are_identical(
    probabilistic_artifact, tmp_path
):
    import torch.multiprocessing as mp

    init_path = tmp_path / "gloo_init"
    result_directory = tmp_path / "results"
    result_directory.mkdir()
    mp.spawn(
        _probabilistic_ddp_worker,
        args=(
            2,
            str(init_path),
            str(probabilistic_artifact),
            str(result_directory),
        ),
        nprocs=2,
        join=True,
    )
    rank_zero = torch.load(
        result_directory / "rank_0.pt", weights_only=True
    )
    rank_one = torch.load(
        result_directory / "rank_1.pt", weights_only=True
    )
    for key in ("support", "kl", "dual", "p", "q"):
        torch.testing.assert_close(rank_zero[key], rank_one[key])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_gated_triton_forward_backward_matches_straight_through_reference():
    from rl_games.algos_torch.connectome_ops import Graph
    from rl_games.algos_torch.connectome_probabilistic_plasticity import (
        dense_gated_recurrent_reference,
    )
    from rl_games.algos_torch.connectome_triton import fused_gated_step

    torch.manual_seed(7)
    device = torch.device("cuda")
    neuron_count, batch = 4, 5
    crow = torch.tensor([0, 2, 3, 5, 6], device=device)
    col = torch.tensor([1, 2, 0, 1, 3, 2], device=device)
    rows = torch.repeat_interleave(
        torch.arange(neuron_count, device=device), crow[1:] - crow[:-1]
    )
    canonical = rows * neuron_count + col
    graph = Graph(crow, col, canonical)
    base_values = torch.randn(6, device=device)
    base_logits = torch.randn(6, device=device)
    base_hidden = torch.randn(batch, neuron_count, device=device)
    incoming = torch.rand(neuron_count, device=device) + 0.5
    outgoing = torch.rand(neuron_count, device=device) + 0.5
    leak = torch.rand(neuron_count, device=device) * 0.5 + 0.2
    intrinsic = torch.rand(neuron_count, device=device) + 0.5
    bias = torch.randn(neuron_count, device=device) * 0.1
    sensory = torch.randn(batch, 2, device=device)
    descending = torch.randn(batch, 1, device=device)
    sensory_indices = torch.tensor([0, 2], device=device)
    descending_indices = torch.tensor([3], device=device)
    seeds = torch.tensor([11, 22, 33, 44, 55], device=device)
    beta = 0.9
    probe = torch.randn(batch, neuron_count, device=device)

    def run(kind):
        values = base_values.clone().requires_grad_()
        logits = base_logits.clone().requires_grad_()
        hidden = base_hidden.clone().requires_grad_()
        if kind == "triton":
            output = fused_gated_step(
                graph,
                values,
                logits,
                seeds,
                hidden,
                incoming,
                outgoing,
                leak,
                intrinsic,
                bias,
                sensory,
                descending,
                sensory_indices,
                descending_indices,
                beta,
            )
        else:
            recurrent = dense_gated_recurrent_reference(
                rows,
                col,
                values,
                logits,
                hidden,
                seeds,
                canonical,
                outgoing,
            )
            drive = torch.zeros_like(hidden)
            drive[:, sensory_indices] += sensory
            drive[:, descending_indices] += descending
            preactivation = intrinsic * (
                beta * incoming * recurrent + drive
            ) + bias
            output = (1.0 - leak) * hidden + leak * torch.tanh(
                preactivation
            )
        (output * probe).sum().backward()
        return output, hidden.grad, values.grad, logits.grad

    actual = run("triton")
    expected = run("reference")
    for name, got, want in zip(
        ("forward", "hidden", "conditional", "topology"),
        actual,
        expected,
    ):
        torch.testing.assert_close(
            got, want, atol=2.0e-5, rtol=2.0e-4, msg=name
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_network_reuses_rollout_topology_across_neural_updates(
    probabilistic_artifact,
):
    network = _build(probabilistic_artifact, budget=5).cuda()
    network.configure_probabilistic_runtime(5)
    network.refresh_probabilistic_support()
    with torch.no_grad():
        slices = network._latent_slices
        network.synaptic_post_anchor.normal_(std=1.0)
        network.topology_post_anchor.normal_(std=2.0)
        network.plasticity_state[:, slices.synaptic_pre].normal_(std=1.0)
        network.plasticity_state[:, slices.topology_pre].normal_(std=2.0)
    observations = _observations(4, "cuda")
    seeds = torch.tensor([1, 2, 3, 4], device="cuda")
    default_state = tuple(
        state.cuda() for state in network.get_default_rnn_state()
    )
    output_one = network(
        {
            "obs": observations,
            "rnn_states": default_state,
            "topology_seed": seeds,
            "seq_length": 1,
            "is_train": False,
        }
    )
    output_two = network(
        {
            "obs": observations,
            "rnn_states": default_state,
            "topology_seed": seeds,
            "seq_length": 1,
            "is_train": False,
        }
    )
    for first, second in zip(output_one[:3], output_two[:3]):
        torch.testing.assert_close(first, second)
    different = network(
        {
            "obs": observations,
            "rnn_states": default_state,
            "topology_seed": seeds + 100,
            "seq_length": 1,
            "is_train": False,
        }
    )
    assert not torch.equal(output_one[0], different[0])
    with pytest.raises(ValueError, match="constant"):
        network(
            {
                "obs": _observations(4, "cuda"),
                "rnn_states": (
                    torch.zeros(1, 2, 7, device="cuda"),
                ),
                "topology_seed": torch.tensor(
                    [1, 2, 1, 3], device="cuda"
                ),
                "seq_length": 2,
                "is_train": True,
            }
        )

    map_network = _build(
        probabilistic_artifact, budget=5, inference_mode="map"
    ).cuda()
    map_network.refresh_probabilistic_support()
    map_state = tuple(
        state.cuda() for state in map_network.get_default_rnn_state()
    )
    with torch.no_grad():
        map_first = map_network(
            {
                "obs": observations,
                "rnn_states": map_state,
                "topology_inference": True,
                "seq_length": 1,
                "is_train": False,
            }
        )
        map_second = map_network(
            {
                "obs": observations,
                "rnn_states": map_state,
                "topology_inference": True,
                "seq_length": 1,
                "is_train": False,
            }
        )
    for first, second in zip(map_first[:3], map_second[:3]):
        torch.testing.assert_close(first, second)

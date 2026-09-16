import numpy as np
import pytest
import torch
from scipy import sparse
from simtoolreal_shared.retina import RetinalPopulationEncoder
from simtoolreal_shared.visual_connectome import distances


class Body(torch.nn.Module):
    def forward(self, x):
        out = x.new_zeros((len(x), 4))
        out[:, 0] = x[:, 0]
        return out


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA kernels')
def test_fused_vision_matches_reference_and_preserves_proprioception():
    from simtoolreal_shared.vision_ops import rgba_luminance
    rgba = torch.randint(0, 256, (3, 36, 64, 4), device='cuda', dtype=torch.uint8)
    rgb = rgba[..., :3].float() / 255.
    gray = (rgb * rgb.new_tensor([.299, .587, .114])).sum(-1)
    out = torch.empty_like(gray)
    assert rgba_luminance(rgba, out).data_ptr() == out.data_ptr()
    torch.testing.assert_close(out, gray, atol=2e-7, rtol=2e-6)
    # Include corner and interior sampling, extra trailing SAPG column and a
    # noncontiguous row stride, as occur in policy observation views.
    obs = torch.rand(3, 2307, device='cuda')[:, :2306]
    regular = RetinalPopulationEncoder(Body(), [1, 2, 3],
        [[-1, -1], [1, 1], [.24, -.57]], image_start=1, height=36, width=64).cuda()
    fused = RetinalPopulationEncoder(Body(), [1, 2, 3],
        [[-1, -1], [1, 1], [.24, -.57]], image_start=1, height=36, width=64, fused=True).cuda()
    with torch.no_grad():
        actual = fused(obs)
        torch.testing.assert_close(actual, regular(obs), atol=1e-5, rtol=2e-5)
        torch.testing.assert_close(actual[:, 0], obs[:, 0], atol=0, rtol=0)
    with pytest.raises(RuntimeError, match='no_grad'):
        fused(obs)


def test_retina_uses_spatial_pixels_and_preserves_proprioception():
    encoder = RetinalPopulationEncoder(Body(), [1, 2], [[-1, -1], [1, 1]],
                                       image_start=1, height=2, width=2)
    observations = torch.tensor([[.7, 0., .5, .5, 1.], [-.2, 1., .5, .5, 0.]])
    drive = encoder(observations)
    torch.testing.assert_close(drive[:, 0], observations[:, 0])
    torch.testing.assert_close(drive[:, 1], -drive[:, 2])
    assert drive[0, 1] > 0 and drive[1, 1] < 0
    assert (drive[:, 3] == 0).all()
    assert not list(encoder.parameters())


def test_directed_paths_use_post_pre_orientation():
    graph = sparse.coo_matrix((np.ones(3, dtype=bool), ([1, 2, 3], [0, 1, 2])), shape=(5, 5)).tocsr()
    assert distances(graph, [0], 4).tolist() == [0, 1, 2, 3, 5]
    assert distances(graph.T.tocsr(), [3], 4).tolist() == [3, 2, 1, 0, 5]


def test_visual_profile_has_no_object_state_shortcut():
    from pathlib import Path
    from hydra import compose, initialize_config_dir
    root = Path(__file__).resolve().parents[2]
    with initialize_config_dir(version_base=None, config_dir=str(root / 'isaacgymenvs/cfg')):
        cfg = compose(config_name='config', overrides=['task=SimToolRealVisionProprio',
                      'train=SimToolRealVisualReservoirGaussianSAPG'])
    assert list(cfg.task.env.obsList) == ['joint_pos', 'joint_vel', 'prev_action_targets',
                                         'goal_keypoints_world', 'camera_luminance']
    net = cfg.train.params.network.connectome
    assert net.dynamics.activation == 'tanh'
    assert net.dynamics.neural_updates == 9
    assert net.observations.policy_size == 99 + cfg.task.env.policyVision.width * cfg.task.env.policyVision.height
    assert not cfg.train.params.config.normalize_input
    assert net.reservoir_readout.enabled
    assert net.fixed_input_encoder.retina.image_start == 99


def test_multirate_low_resolution_visual_profile_contract():
    from pathlib import Path
    from hydra import compose, initialize_config_dir
    root = Path(__file__).resolve().parents[2]
    with initialize_config_dir(version_base=None, config_dir=str(root / 'isaacgymenvs/cfg')):
        cfg = compose(
            config_name='config',
            overrides=[
                'task=SimToolRealVisionProprio64x36R4',
                'train=SimToolRealFullCNSVisualReservoirGaussianSAPGVision64x36R4',
            ],
        )
    vision = cfg.task.env.policyVision
    net = cfg.train.params.network.connectome
    assert (vision.width, vision.height, vision.renderInterval) == (64, 36, 4)
    assert net.observations.policy_size == 99 + 64 * 36 == 2403
    assert net.observations.context_size == 29 + 64 * 36 == 2333
    assert list(net.observations.context_ranges) == [[58, 87], [99, 2403]]
    assert (net.fixed_input_encoder.retina.height,
            net.fixed_input_encoder.retina.width) == (36, 64)
    assert net.fixed_input_encoder.retina.image_start == 99


def test_fast_visual_profile_preserves_camera_and_neural_timing():
    from pathlib import Path
    from hydra import compose, initialize_config_dir
    root = Path(__file__).resolve().parents[2]
    with initialize_config_dir(version_base=None, config_dir=str(root / 'isaacgymenvs/cfg')):
        cfg = compose(config_name='config', overrides=[
            'task=SimToolRealVisionProprio64x36R4',
            'train=SimToolRealFullCNSVisualReservoirGaussianSAPGFast'])
    net = cfg.train.params.network.connectome
    assert net.backend_options.frozen_inference
    assert not net.backend_options.cuda_graph
    assert net.fixed_input_encoder.retina.fused
    assert net.dynamics.neural_updates == 9
    assert net.reservoir_readout.enabled
    assert net.observations.policy_size == 2403
    assert cfg.task.env.policyVision.renderInterval == 4

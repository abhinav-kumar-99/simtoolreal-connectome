import numpy as np
import torch
from scipy import sparse
from simtoolreal_shared.retina import RetinalPopulationEncoder
from simtoolreal_shared.visual_connectome import distances


class Body(torch.nn.Module):
    def forward(self, x):
        out = x.new_zeros((len(x), 4))
        out[:, 0] = x[:, 0]
        return out


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

import numpy as np
import pytest
import torch

from simtoolreal_shared.activity_trace import ActivityRecorder

from . import test_connectome_network as fixtures
from .test_beta_multirate import network
from .test_connectome_network import _observations

artifact_path = fixtures.artifact_path


@pytest.mark.parametrize("backend", ["native_csr", "triton_fused"])
def test_substep_recording_preserves_actions_states_and_checkpoint(
    artifact_path, tmp_path, backend
):
    if backend == "triton_fused" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = "cuda" if backend == "triton_fused" else "cpu"
    net = network(artifact_path, backend=backend).to(device)
    obs = _observations(1).to(device)
    initial = torch.randn(1, 1, 7, device=device)
    inputs = {"obs": obs, "rnn_states": (initial.clone(),)}
    checkpoint_keys = set(net.state_dict())
    with torch.no_grad():
        reference = net(inputs)
        recorder = ActivityRecorder(net, artifact_path)
        recorder.start_episode(0, initial)
        recorder.start_step(0)
        recorder.video_frame(0)
        recorded = net(inputs)
        recorder.finish_step()
        recorder.close()
    for left, right in zip(reference[:3], recorded[:3]):
        torch.testing.assert_close(left, right, atol=0, rtol=0)
    torch.testing.assert_close(reference[3][0], recorded[3][0], atol=0, rtol=0)
    assert len(recorder.states) == 5
    np.testing.assert_array_equal(
        recorder.states[-1], recorded[3][0].cpu().numpy().reshape(-1)
    )
    assert set(net.state_dict()) == checkpoint_keys
    assert net.activity_observer is None
    recorder.save(tmp_path / "trace.npz", {})
    with np.load(tmp_path / "trace.npz") as trace:
        assert trace["tick"].tolist() == [0, 1, 2, 3, 4]


def test_observation_hook_rejects_training_and_incompatible_backend(artifact_path):
    net = network(artifact_path)
    recorder = ActivityRecorder(net, artifact_path)
    with pytest.raises(RuntimeError, match="no-grad"):
        net._step(_observations(1), torch.zeros(1, 7))
    recorder.close()
    net.backend_options["frozen_inference"] = True
    with pytest.raises(ValueError, match="frozen_inference"):
        ActivityRecorder(net, artifact_path)

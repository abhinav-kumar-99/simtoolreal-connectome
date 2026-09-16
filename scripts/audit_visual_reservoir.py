#!/usr/bin/env python3
"""YAML-owned camera leakage and visual-path sensitivity check."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'rl_games'))
import yaml
import imageio.v2 as imageio
from isaacgym import gymapi
import torch
from deployment.isaac.isaac_env import create_env
from deployment.rl_player import RlPlayer
from isaacgymenvs.utils.rendering import render_camera_sensors_for_current_step


def run(cfg):
    output = Path(cfg['output_directory'])
    output.mkdir(parents=True, exist_ok=True)
    env = create_env(cfg['policy_config_path'], device='cuda:0', headless=True,
        overrides={'task.env.numEnvs': 2, 'task.env.useObsDelay': False, 'task.env.capture_video': False})
    obs = env.step(torch.zeros((2, env.num_acts), device=env.device))[0]['obs'].clone()
    def capture():
        render_camera_sensors_for_current_step(env.gym, env.sim, env.device)
        return env.gym.get_camera_image(env.sim, env.envs[0], env.policy_camera_handles[0], gymapi.IMAGE_COLOR).reshape(54, 96, 4)[..., :3].copy()
    before = capture()
    camera_distinct = bool((obs[0, 99:] - obs[1, 99:]).abs().max() > 0)
    assert camera_distinct, 'Environment cameras unexpectedly identical'
    env.goal_states[:, :3] += .2
    env.root_state_tensor[env.goal_object_indices, :7] = env.goal_states[:, :7]
    env.deferred_set_actor_root_state_tensor_indexed([env.goal_object_indices])
    env.set_actor_root_state_tensor_indexed()
    after = capture()
    import numpy as np
    assert np.array_equal(before, after), 'Goal visualization leaked into policy camera'
    assert before.std() > 1, 'Blank policy camera'
    imageio.imwrite(output / 'policy_camera.png', before)
    render_camera_sensors_for_current_step(env.gym, env.sim, env.device)
    frame = env.gym.get_camera_image(env.sim, env.envs[0], env.camera_handle, gymapi.IMAGE_COLOR)
    imageio.imwrite(output / 'evaluation_camera.png', frame.reshape(env.camera_properties.height, env.camera_properties.width, 4)[..., :3])
    player = RlPlayer(num_observations=env.num_obs, num_actions=env.num_acts,
        config_path=cfg['policy_config_path'], checkpoint_path=cfg['checkpoint_path'], device='cuda:0', num_envs=2)
    network = player.player.model.a2c_network
    def motor_for(image_value):
        player.reset()
        x = obs.clone()
        x[:, 99:] = image_value
        for _ in range(int(cfg['sensitivity_control_steps'])):
            player.get_normalized_action(x, deterministic_actions=True)
        return player.player.states[0][0, :, network.motor_indices].clone()
    with torch.no_grad():
        dark, light = motor_for(0.), motor_for(1.)
    delta = (dark - light).abs()
    assert torch.isfinite(delta).all() and delta.max() > 1e-8, 'No measurable camera-to-motor response'
    result = dict(goal_image_identical=True, camera_std=float(before.std()),
        environment_cameras_distinct=camera_distinct,
        motor_visual_delta_max=float(delta.max()), motor_visual_delta_mean=float(delta.mean()),
        responsive_motor_cells=int((delta.max(0).values > 1e-8).sum()),
        neurons=network.neuron_count, camera_shape=list(before.shape),
        sensitivity_control_steps=cfg['sensitivity_control_steps'])
    (output / 'audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    env.gym.destroy_sim(env.sim)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    with open(args.config) as stream:
        run(yaml.safe_load(stream))

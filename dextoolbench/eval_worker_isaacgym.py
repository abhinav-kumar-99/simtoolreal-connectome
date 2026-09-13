#!/usr/bin/env python3
"""Run one headless DexToolBench evaluation case from a YAML contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import yaml

# Isaac Gym must be imported before torch.
from isaacgym import gymapi  # noqa: F401
import torch

from deployment.isaac.isaac_env import create_env
from deployment.rl_player import RlPlayer
from isaacgymenvs.utils.rendering import render_camera_sensors_for_current_step

TABLE_Z = 0.38


def _load_yaml(path: Path) -> dict:
    with path.open() as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise TypeError(f"Expected YAML mapping in {path}")
    return config


def _capture_frame(env) -> np.ndarray:
    render_camera_sensors_for_current_step(env.gym, env.sim, env.device)
    rgba = env.gym.get_camera_image(
        env.sim,
        env.envs[env.index_to_view],
        env.camera_handle,
        gymapi.IMAGE_COLOR,
    )
    frame = rgba.reshape(
        env.camera_properties.height, env.camera_properties.width, 4
    )[..., :3]
    return np.ascontiguousarray(frame)


def run(config: dict) -> dict:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    config_path = Path(config["policy_config_path"])
    checkpoint_path = Path(config["checkpoint_path"])
    output_path = Path(config["output_path"])
    trajectory_path = Path(config["trajectory_path"])
    with trajectory_path.open() as stream:
        trajectory = json.load(stream)
    trajectory["start_pose"][2] += float(config.get("z_offset", 0.03))
    trajectory["goals"] = trajectory["goals"][:: int(config["downsample_factor"])]

    env = create_env(
        config_path=str(config_path),
        headless=True,
        device=device,
        overrides={
            "task.env.resetPositionNoiseX": 0.0,
            "task.env.resetPositionNoiseY": 0.0,
            "task.env.resetPositionNoiseZ": 0.0,
            "task.env.randomizeObjectRotation": False,
            "task.env.resetDofPosRandomIntervalFingers": 0.0,
            "task.env.resetDofPosRandomIntervalArm": 0.0,
            "task.env.resetDofVelRandomInterval": 0.0,
            "task.env.tableResetZRange": 0.0,
            "task.env.objectName": config["object_name"],
            "task.env.numEnvs": 1,
            "task.env.envSpacing": 0.4,
            "task.env.capture_video": False,
            "task.env.useFixedGoalStates": True,
            "task.env.fixedGoalStates": trajectory["goals"],
            "task.env.useActionDelay": False,
            "task.env.useObsDelay": False,
            "task.env.useObjectStateDelayNoise": False,
            "task.env.objectScaleNoiseMultiplierRange": [1.0, 1.0],
            "task.env.resetWhenDropped": False,
            "task.env.armMovingAverage": 0.1,
            "task.env.evalSuccessTolerance": float(config["success_tolerance"]),
            "task.env.successSteps": 1,
            "task.env.fixedSizeKeypointReward": True,
            "task.env.asset.table": config["table_urdf"],
            "task.env.tableResetZ": TABLE_Z,
            "task.env.useFixedInitObjectPose": True,
            "task.env.objectStartPose": trajectory["start_pose"],
            "task.env.startArmHigher": True,
            "task.env.forceScale": 0.0,
            "task.env.torqueScale": 0.0,
            "task.env.linVelImpulseScale": 0.0,
            "task.env.angVelImpulseScale": 0.0,
            "task.env.forceOnlyWhenLifted": True,
            "task.env.torqueOnlyWhenLifted": True,
            "task.env.linVelImpulseOnlyWhenLifted": True,
            "task.env.angVelImpulseOnlyWhenLifted": True,
            "task.env.forceProbRange": [0.0001, 0.0001],
            "task.env.torqueProbRange": [0.0001, 0.0001],
            "task.env.linVelImpulseProbRange": [0.0001, 0.0001],
            "task.env.angVelImpulseProbRange": [0.0001, 0.0001],
        },
    )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    env.set_env_state(checkpoint[0]["env_state"])
    policy = RlPlayer(
        num_observations=140,
        num_actions=29,
        config_path=str(config_path),
        checkpoint_path=str(checkpoint_path),
        device=device,
        num_envs=1,
    )

    episode_results = []
    all_video_frames = []
    zero_action = torch.zeros((1, 29), device=device)
    obs = env.step(zero_action)[0]["obs"]
    max_steps = int(config.get("max_steps", env.cfg["env"]["episodeLength"] + 1))
    for episode in range(int(config["num_episodes"])):
        policy.reset()
        if episode:
            obs = env.step(zero_action)[0]["obs"]
        episode_reward = 0.0
        frames = []
        steps = 0
        done = False
        while not done and steps < max_steps:
            if bool(config["record_video"]) and steps % int(
                config["video_frame_interval"]
            ) == 0:
                frames.append(_capture_frame(env))
            action = policy.get_normalized_action(obs, deterministic_actions=True)
            obs_dict, reward, dones, _ = env.step(action)
            obs = obs_dict["obs"]
            episode_reward += float(reward[0].item())
            done = bool(dones[0].item())
            steps += 1
        successes = int(env.successes[0].item())
        task_progress = 100.0 * successes / env.max_consecutive_successes
        episode_results.append(
            {
                "episode": episode,
                "steps": steps,
                "raw_reward": episode_reward,
                "shaped_reward": episode_reward * 0.01,
                "successful_waypoints": successes,
                "total_waypoints": int(env.max_consecutive_successes),
                "task_progress_pct": task_progress,
                "terminated": done,
            }
        )
        all_video_frames.extend(frames)

    video_path = None
    if all_video_frames:
        video_path = Path(config["video_path"])
        video_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(
            video_path,
            all_video_frames,
            fps=int(config["video_fps"]),
            macro_block_size=2,
        )

    result = {
        "policy": config["policy"],
        "metric": config["metric"],
        "success_tolerance_m": float(config["success_tolerance"]),
        "object_category": config["object_category"],
        "object_name": config["object_name"],
        "task_name": config["task_name"],
        "episodes": episode_results,
        "mean_raw_reward": float(np.mean([x["raw_reward"] for x in episode_results])),
        "mean_shaped_reward": float(
            np.mean([x["shaped_reward"] for x in episode_results])
        ),
        "mean_task_progress_pct": float(
            np.mean([x["task_progress_pct"] for x in episode_results])
        ),
        "video_path": str(video_path) if video_path else None,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(_load_yaml(args.config)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

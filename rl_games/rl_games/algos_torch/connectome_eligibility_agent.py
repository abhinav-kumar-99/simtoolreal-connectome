"""Single-policy, online TD actor-critic using local connectome eligibility."""
from __future__ import annotations

import math
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from rl_games.algos_torch import model_builder, torch_ext
from rl_games.algos_torch.connectome_eligibility import LocalEligibility
from rl_games.common import vecenv
from simtoolreal_shared.milestone_checkpoints import crossed_milestone_targets, milestone_checkpoint_name


class ConnectomeEligibilityAgent:
    def __init__(self, base_name, params):
        self.config = c = params["config"]
        self.settings = e = dict(c["eligibility"])
        if c.get("multi_gpu", False):
            raise ValueError("Eligibility currently supports a single process/GPU")
        if e["bootstrap_timeouts"]:
            raise ValueError("Only finite-horizon terminal semantics are implemented; bootstrap_timeouts must be false")
        if c.get("rollout_accumulation_steps", 1) != 1:
            raise ValueError("Eligibility does not accumulate/replay PPO rollouts")
        if c.get("normalize_value", False):
            raise ValueError("Eligibility uses an unnormalized separate TD value predictor")
        if float(e["sigma"]) <= 0 or not math.isfinite(float(e["sigma"])):
            raise ValueError("eligibility.sigma must be finite and positive")
        if int(e["critic_width"]) <= 0 or not 0 < float(e["critic_lr"]) < float("inf"):
            raise ValueError("Critic width and learning rate must be positive")
        if int(c["horizon_length"]) <= 0 or int(c["save_frequency"]) <= 0:
            raise ValueError("horizon_length and save_frequency must be positive")
        self.device = torch.device(c.get("device", "cuda:0"))
        self.ppo_device, self.num_agents = self.device, 1  # Observer API compatibility.
        self.global_rank, self.has_central_value = 0, False
        self.num_actors = int(c["num_actors"])
        self.games_to_track = int(c.get("games_to_track", 100))
        self.vec_env = c.get("vec_env")
        if self.vec_env is None:
            self.vec_env = vecenv.create_vec_env(c["env_name"], self.num_actors, **c.get("env_config", {}))
        info = self.vec_env.get_env_info()
        self.obs_size = int(info["observation_space"].shape[0])
        self.action_size = int(info["action_space"].shape[0])
        self.action_low = torch.as_tensor(info["action_space"].low, device=self.device)
        self.action_high = torch.as_tensor(info["action_space"].high, device=self.device)
        self.model = model_builder.ModelBuilder().load(params).build({
            "actions_num": self.action_size, "input_shape": (self.obs_size + 1,),
            "num_seqs": self.num_actors, "value_size": 1,
            "normalize_value": False, "normalize_input": bool(c.get("normalize_input", True)),
            "type": "extra_param", "coef_ids": torch.linspace(50, 0, 6), "coef_id_idx": self.obs_size,
        }).to(self.device)
        self.net = self.model.a2c_network
        with torch.no_grad():
            self.net.sigma.fill_(math.log(float(e["sigma"])))
        self.engine = LocalEligibility(self.net, self.num_actors, e)
        # Feedforward critic sees normalized robot observations and detached
        # current recurrent features; it never differentiates through the actor.
        width = int(e["critic_width"])
        self.critic = nn.Sequential(nn.Linear(self.obs_size + self.net.neuron_count, width), nn.Tanh(),
                                    nn.Linear(width, width), nn.Tanh(), nn.Linear(width, 1)).to(self.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=float(e["critic_lr"]))
        self.frame, self.epoch_num = 0, 0
        self.elapsed_before = 0.0
        self.run_dir = Path(c.get("train_dir", "runs")) / c["name"]
        self.nn_dir = self.run_dir / "nn"
        self.nn_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(str(self.run_dir / "summaries"))
        self.observer = c["features"]["observer"]
        self.observer.before_init(base_name, c, c["name"])
        self.observer.after_init(self)

    @torch.no_grad()
    def normalized(self, observation, update=False):
        raw = observation["obs"] if isinstance(observation, dict) else observation
        raw = raw.to(self.device).float()
        augmented = torch.cat((raw, raw.new_full((raw.shape[0], 1), 50)), dim=-1)
        self.model.train(update)
        return self.model.norm_obs(augmented)

    def features(self, obs, hidden):
        return torch.cat((obs[:, :self.obs_size], hidden), dim=-1).detach()

    def set_weights(self, weights):
        incoming = dict(weights["model"])
        current = self.model.state_dict()
        for name in ("crow_indices", "col_indices", "recurrent_values",
                     "sensory_indices", "descending_indices", "motor_indices"):
            key = "a2c_network." + name
            if not torch.equal(incoming[key].to(current[key].device), current[key]):
                raise ValueError(f"Checkpoint graph identity differs: {name}")
        # PPO's value normalizer is unused by this actor and separate critic.
        incoming = {key: value for key, value in incoming.items() if not key.startswith("value_mean_std.")}
        self.model.load_state_dict(incoming)
        # Weights-only initialization deliberately keeps the configured noise.
        with torch.no_grad():
            self.net.sigma.fill_(math.log(float(self.settings["sigma"])))

    def restore(self, path):
        state = torch_ext.load_checkpoint(path)
        state = state[0] if 0 in state else state
        trainer = state.get("trainer_state", {})
        if trainer.get("algorithm") != "connectome_eligibility":
            raise ValueError("Resume requires a full eligibility checkpoint; use weights mode for PPO/inference checkpoints")
        self.engine.load_state_dict(trainer["eligibility"])
        self.set_weights(state)
        self.critic.load_state_dict(trainer["critic"])
        self.critic_optimizer.load_state_dict(trainer["critic_optimizer"])
        self.frame, self.epoch_num = int(state["frame"]), int(state["epoch"])
        self.elapsed_before = float(trainer["elapsed_seconds"])
        if state.get("env_state") is not None:
            self.vec_env.set_env_state({"rewards_episode": {}, **{
                key: state["env_state"][key] for key in ("success_tolerance", "last_curriculum_update")
                if key in state["env_state"]
            }})
        print("Eligibility resume: restored weights, critic, feedback and counters; starting fresh episodes with zero traces")

    def save(self, filename, elapsed, inference_only=False):
        state = {"model": self.model.state_dict(), "epoch": self.epoch_num, "frame": self.frame}
        if self.model.normalize_input:
            state["running_mean_std"] = self.model.running_mean_std.state_dict()
        if not inference_only:
            state["trainer_state"] = {"algorithm": "connectome_eligibility", "eligibility": self.engine.state_dict(),
                                      "critic": self.critic.state_dict(), "critic_optimizer": self.critic_optimizer.state_dict(),
                                      "elapsed_seconds": elapsed}
            env_state = self.vec_env.get_env_state()
            # SimToolReal's full state includes unfinished physics episodes.
            # Restore curriculum scalars only, consistent with zeroed traces.
            state["env_state"] = ({"rewards_episode": {}, **{
                key: env_state[key] for key in ("success_tolerance", "last_curriculum_update") if key in env_state
            }} if isinstance(env_state, dict) else None)
        path = self.nn_dir / filename
        temporary = path.with_suffix(".tmp")
        torch.save({0: state}, temporary)
        temporary.replace(path)

    def train(self):
        self.vec_env.reset()
        hidden = torch.zeros(self.num_actors, self.net.neuron_count, device=self.device)
        episode_returns = torch.zeros(self.num_actors, device=self.device)
        episode_lengths = torch.zeros_like(episode_returns)
        max_frames = int(self.config.get("max_frames", -1))
        horizon = int(self.config["horizon_length"])
        started = time.perf_counter()
        try:
            while self.epoch_num < int(self.config["max_epochs"]) and (max_frames < 0 or self.frame + self.num_actors <= max_frames):
                epoch_start, previous_frame = time.perf_counter(), self.frame
                self.epoch_num += 1
                metrics, returns, lengths = [], [], []
                for _ in range(horizon):
                    if max_frames >= 0 and self.frame + self.num_actors > max_frames:
                        break
                    observation, reset_ids = self.vec_env.reset_done()
                    hidden[reset_ids] = 0
                    self.engine.reset(reset_ids)
                    self.vec_env.set_train_info(self.frame, self)
                    with torch.no_grad():
                        obs = self.normalized(observation, update=True)
                        mu, log_std, _, states = self.net({"obs": obs, "rnn_states": (hidden.unsqueeze(0),)})
                        next_hidden = states[0][0]
                        std = log_std.exp()
                        sample = mu + std * torch.randn_like(mu)
                        if not torch.isfinite(sample).all():
                            raise FloatingPointError("Non-finite sampled policy action")
                        score = (sample - mu) / std.square()
                        self.engine.observe(obs, hidden, next_hidden, score)
                        # The score belongs to the unclipped latent Gaussian
                        # sample; the environment action is its clipped transform.
                        action = self.action_low + (sample.clamp(-1, 1) + 1) * .5 * (self.action_high - self.action_low)
                    value = self.critic(self.features(obs, next_hidden)).squeeze(-1)
                    next_obs, reward, done, infos = self.vec_env.step(action)
                    reward = reward.to(self.device).flatten()
                    done = done.to(self.device).bool().flatten()
                    with torch.no_grad():
                        normalized_next = self.normalized(next_obs)
                        # A critic-only lookahead, not another policy/control step.
                        _, _, _, bootstrap_states = self.net({"obs": normalized_next, "rnn_states": (next_hidden.unsqueeze(0),)})
                        bootstrap = self.critic(self.features(normalized_next, bootstrap_states[0][0])).squeeze(-1)
                        shaped = self.config["reward_shaper"](reward)
                        target = shaped + float(self.settings["gamma"]) * torch.where(done, 0., bootstrap)
                        td = target - value.detach()
                    stats = self.engine.update(td)
                    self.critic_optimizer.zero_grad(set_to_none=True)
                    loss = torch.nn.functional.smooth_l1_loss(value, target)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Non-finite critic loss")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.settings["max_grad_norm"], error_if_nonfinite=True)
                    self.critic_optimizer.step()
                    stats["critic_loss"] = loss.item()
                    stats["reward_mean"] = reward.mean().item()
                    stats["action_clip_fraction"] = (sample.abs() > 1).float().mean().item()
                    stats["hidden_saturation_fraction"] = (next_hidden.abs() > .95).float().mean().item()
                    metrics.append(stats)
                    self.frame += self.num_actors
                    episode_returns += reward
                    episode_lengths += 1
                    ids = done.nonzero(as_tuple=False).flatten()
                    returns.extend(episode_returns[ids].tolist())
                    lengths.extend(episode_lengths[ids].tolist())
                    episode_returns[ids], episode_lengths[ids] = 0, 0
                    self.observer.process_infos(infos, ids)
                    hidden = next_hidden
                    hidden[ids] = 0
                    self.engine.reset(ids)
                elapsed = self.elapsed_before + time.perf_counter() - started
                for name in metrics[0]:
                    value = sum(row[name] for row in metrics) / len(metrics)
                    self.writer.add_scalar("eligibility/" + name, value, self.frame)
                self.writer.add_scalar("eligibility/updates", self.engine.updates, self.frame)
                self.writer.add_scalar("rewards/transition", sum(row["reward_mean"] for row in metrics) / len(metrics), self.frame)
                self.writer.add_scalar("eligibility/incoming_gain_mean", self.net.incoming_gains().mean().item(), self.frame)
                self.writer.add_scalar("eligibility/outgoing_gain_mean", self.net.outgoing_gains().mean().item(), self.frame)
                self.writer.add_scalar("performance/env_frames_per_second", (self.frame - previous_frame) / (time.perf_counter() - epoch_start), self.frame)
                self.writer.add_scalar("performance/elapsed_seconds", elapsed, self.frame)
                if returns:
                    mean_return = sum(returns) / len(returns)
                    self.writer.add_scalar("rewards/step", mean_return, self.frame)
                    self.writer.add_scalar("rewards/time", mean_return, int(elapsed))
                    self.writer.add_scalar("episode_lengths/step", sum(lengths) / len(lengths), self.frame)
                self.observer.after_print_stats(self.frame, self.epoch_num, elapsed)
                interval = int(self.config.get("inference_checkpoint_interval_frames", 0))
                for milestone in crossed_milestone_targets(previous_frame, self.frame, interval):
                    self.save(milestone_checkpoint_name(milestone, self.frame, self.epoch_num), elapsed, inference_only=True)
                if self.epoch_num % int(self.config["save_frequency"]) == 0:
                    self.save("last.pth", elapsed)
                self.writer.flush()
                print(f"Eligibility epoch={self.epoch_num} frames={self.frame} updates={self.engine.updates} elapsed={elapsed:.2f}s", flush=True)
            elapsed = self.elapsed_before + time.perf_counter() - started
            if int(self.config.get("inference_checkpoint_interval_frames", 0)) > 0:
                self.save(milestone_checkpoint_name(self.frame, self.frame, self.epoch_num), elapsed, inference_only=True)
            self.save("last.pth", elapsed)
            return self.num_actors
        finally:
            self.writer.close()

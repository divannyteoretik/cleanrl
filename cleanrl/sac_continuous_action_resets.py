# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import datetime
import json
import math
import os
import random
import time
from collections import deque
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

# in "The Primacy Bias in RL", they reset actor completely when training SAC

"""
actor params:
fc1.weight torch.Size([256, 27])
fc1.bias torch.Size([256])
fc2.weight torch.Size([256, 256])
fc2.bias torch.Size([256])
fc_mean.weight torch.Size([8, 256])
fc_mean.bias torch.Size([8])
fc_logstd.weight torch.Size([8, 256])
fc_logstd.bias torch.Size([8])
"""


RESET_COEF_DICT = {
    "actor": {"fc1": 0.5, "fc2": 0.5, "fc_mean": 1, "fc_logstd": 1},
}

RESET_COEF_DICT = {
    "actor": {"fc1": 1, "fc2": 1, "fc_mean": 1, "fc_logstd": 1},
}


def reset_actor(actor):
    for m in actor.modules():
        print(m)
        if "Actor" in m.__str__():
            continue
        m.reset_parameters()


# sort weights by abs, use reset_coef as a percentile
# for resets
def reset_by_abs(param, rand_part, reset_coef, mask=None):
    if reset_coef == 1:
        # no need for sorting
        param.data = rand_part
        return

    if mask is None:
        mask = param.data

    # sorting by abs value
    mask = mask.cpu().numpy().flatten().tolist()
    mask_idxs = [*range(len(mask))]
    mask_with_idxs = [*zip(mask, mask_idxs)]
    if "max" in args.reset_type:
        print("resetting max values")
        sorted_mask_wi = sorted(mask_with_idxs, key=lambda x: np.abs(x[0]), reverse=True)
    else:
        sorted_mask_wi = sorted(mask_with_idxs, key=lambda x: np.abs(x[0]))
    sorted_mask, sorted_idxs = list(zip(*sorted_mask_wi))

    data_flatten = param.data.flatten()

    n_weights = len(data_flatten)
    n_reset_weights = int(n_weights * reset_coef)

    reset_idxs = list(sorted_idxs[:n_reset_weights])
    data_flatten[reset_idxs] = rand_part.flatten()[reset_idxs]


def reset_by_val(param, rand_part, reset_coef, mask=None):
    # sort weights, reset min and max values
    if reset_coef == 1:
        # no need for sorting
        param.data = rand_part
        return

    if mask is None:
        mask = param.data

    mask_flatten = mask.flatten()
    mask_sorted, idxs_sorted = torch.sort(mask_flatten)
    n_weights = len(mask_flatten)
    n_reset_weights = int(n_weights * reset_coef)
    first_idxs = idxs_sorted[: n_reset_weights // 2]
    last_idxs = idxs_sorted[n_reset_weights // 2 :]

    data_flatten = param.data.flatten()
    data_flatten[first_idxs] = rand_part.flatten()[first_idxs]
    data_flatten[last_idxs] = rand_part.flatten()[last_idxs]


# https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/linear.py#L105
def reset_weights(param, reset_coef, exp_avg=None):
    print("resetting weights")
    with torch.no_grad():
        param_copy = param.data.clone()
        rand_part = torch.nn.init.kaiming_uniform_(param_copy, a=math.sqrt(5))
        print(args.reset_type)
        if args.reset_type == "full":
            print(f"performing full reset with coef {reset_coef}")
            param.data = reset_coef * rand_part + (1 - reset_coef) * param.data
        elif "wnorm_abs" in args.reset_type:
            print("using abs for resets")
            reset_by_abs(param, rand_part, reset_coef, exp_avg)
        elif "wnorm_val" in args.reset_type:
            print("NOT using abs for resets")
            reset_by_val(param, rand_part, reset_coef, exp_avg)
        else:
            print(args.reset_type)


def reset_bias(param, bound, reset_coef, exp_avg=None):
    with torch.no_grad():
        param_copy = param.data.clone()
        rand_part = nn.init.uniform_(param_copy, -bound, bound)
        if args.reset_type == "full":
            # rand_part = bias_const = 0
            param.data = reset_coef * rand_part + (1 - reset_coef) * param.data
        if "wnorm_abs" in args.reset_type:
            reset_by_abs(param, rand_part, reset_coef, exp_avg)
        if "wnorm_val" in args.reset_type:
            reset_by_val(param, rand_part, reset_coef, exp_avg)


@dataclass
class Args:
    # I don't understand the n_updates_per_train_step part
    # https://github.com/evgenii-nikishin/rl_with_resets/blob/main/discrete_control/agents/rainbow_agent.py#L502C15-L502C94
    n_updates_per_step: int = 1
    n_reset_steps: int = int(1e6 + 1)
    sps_interval: int = 10
    n_last_evals: int = 100
    run_name: str = ""
    reset_type: str = "full"
    no_reset_bias: int = 0
    avg_return_history_step: int = 1000
    heavy_priming_iters: int = 0
    # optimize_actor_on_priming = False
    optimize_actor_on_priming = True
    optimize_q_on_priming = False

    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "Hopper-v4"
    """the environment id of the task"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    learning_starts: int = 5e3
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 1e-3
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""
    noise_clip: float = 0.5
    """noise clip parameter of the Target Policy Smoothing Regularization"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.single_action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
        )

    def set_bias_bounds(self):
        self.bias_bounds = {}
        for mname, m in self.named_modules():
            if len(mname) == 0:
                continue
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            self.bias_bounds[mname] = bound

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


if __name__ == "__main__":
    import stable_baselines3 as sb3

    datenow = datetime.datetime.now()
    print(f"starting at {datenow}")

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:
poetry run pip install "stable_baselines3==2.0.0a1"
"""
        )

    args = tyro.cli(Args)

    reset_str = f"{args.n_reset_steps}"
    if args.n_reset_steps >= args.total_timesteps:
        reset_str = "noresets"

    args.reset_str = reset_str

    run_name = args.run_name
    if len(run_name) == 0:
        run_name = f"{args.env_id}__{args.exp_name}_{args.total_timesteps}_r{reset_str}"
        run_name += f"_rr{args.n_updates_per_step}"
        run_name += f"__{args.seed}__{int(time.time())}"
    print(run_name)

    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, run_name)])
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    max_action = float(envs.single_action_space.high[0])

    actor = Actor(envs).to(device)
    actor.set_bias_bounds()

    qf1 = SoftQNetwork(envs).to(device)
    qf2 = SoftQNetwork(envs).to(device)
    qf1_target = SoftQNetwork(envs).to(device)
    qf2_target = SoftQNetwork(envs).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())

    q_params = []
    qf1_dict = dict(qf1.named_parameters())
    for k, v in qf1_dict.items():
        res_param = {"params": v, "name": f"qf1_{k}"}
        q_params.append(res_param)
    qf2_dict = dict(qf2.named_parameters())
    for k, v in qf2_dict.items():
        res_param = {"params": v, "name": f"qf2_{k}"}
        q_params.append(res_param)

    actor_params = []
    actor_dict = dict(actor.named_parameters())
    for k, v in actor_dict.items():
        print(k, v.data.shape)
        res_param = {"params": v, "name": f"actor_{k}"}
        actor_params.append(res_param)

    # q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    # actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)
    q_optimizer = optim.Adam(q_params, lr=args.q_lr)
    actor_optimizer = optim.Adam(actor_params, lr=args.policy_lr)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)

    sps = 0
    last_sps_measuring_time = time.time()
    last_sps_step = 0

    avg_returns = deque(maxlen=args.n_last_evals)
    avg_return_history = {}
    need_add_avg_return = False
    next_avg_return_history_step = 0

    is_first_learning_step = True
    for global_step in range(args.total_timesteps + 1):
        # print(f"GLOBAL STEP: {global_step}; {args.heavy_priming_iters} {args.learning_starts}")

        if global_step >= next_avg_return_history_step and not need_add_avg_return:
            need_add_avg_return = True
            next_avg_return_history_step += args.avg_return_history_step

        if args.heavy_priming_iters > 0 and is_first_learning_step:
            args.learning_starts = args.batch_size

        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
            actions = actions.detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                datenow = datetime.datetime.now()
                print(f"{datenow} global_step={global_step}, episodic_return={info['episode']['r']}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                avg_returns.append(info["episode"]["r"])
                if need_add_avg_return:
                    avg_ret = round(np.average(avg_returns), 4)
                    print(f"adding average return: step {global_step} return: {avg_ret}")
                    avg_return_history[global_step] = avg_ret
                    need_add_avg_return = False
                break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        # if global_step > args.learning_starts:
        if global_step % args.n_reset_steps == 0:
            # reset_actor(actor) # works, but not flexible

            actor_opt_state = actor_optimizer.state_dict()["state"]

            for i, state in actor_opt_state.items():
                param_group = actor_optimizer.param_groups[i]
                if "name" not in param_group:
                    print("WARNING: Adam param without name; skipping")
                    continue

                # WARNING: it's 1-lentgh array in this particular env/model,
                # not sure whether that's always the case
                param_name = param_group["name"]
                layer_name = param_name.replace("actor_", "").split(".")[0]

                param = param_group["params"][0]

                print(param_name, param.shape, state["exp_avg"].shape)

                exp_avg = None

                if "expavg" in args.reset_type:
                    exp_avg = state["exp_avg"]

                reset_coef = RESET_COEF_DICT["actor"][layer_name]
                # reset_coef = 1
                if "weight" in param_name:
                    print(f"resetting {param_name} with coef {reset_coef}")
                    reset_weights(param, reset_coef, exp_avg)
                if "bias" in param_name:
                    print(f"resetting {param_name} with coef {reset_coef}")
                    print(actor.bias_bounds)
                    bound = actor.bias_bounds[layer_name]
                    if not args.no_reset_bias:
                        reset_bias(param, bound, reset_coef, exp_avg)

        if global_step <= args.learning_starts:
            continue

        n_updates_per_step = args.n_updates_per_step
        is_heavy_priming_step = args.heavy_priming_iters > 0 and is_first_learning_step
        if is_heavy_priming_step:
            n_updates_per_step = args.heavy_priming_iters
            is_first_learning_step = False

        for cur_update_step in range(n_updates_per_step):
            data = rb.sample(args.batch_size)
            if is_heavy_priming_step and cur_update_step % 100 == 0:
                print(f"heavy priming; epoch {cur_update_step}; batch_size: {len(data.dones.flatten())}")

            optimize_q_on_priming = is_heavy_priming_step and args.optimize_q_on_priming
            optimize_q = optimize_q_on_priming or (not is_heavy_priming_step)
            if optimize_q:
                # print("optimizing q")
                with torch.no_grad():
                    next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_observations)
                    qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                    qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                    min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                    next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (
                        min_qf_next_target
                    ).view(-1)

                qf1_a_values = qf1(data.observations, data.actions).view(-1)
                qf2_a_values = qf2(data.observations, data.actions).view(-1)
                qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
                qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
                qf_loss = qf1_loss + qf2_loss

                # optimize the model
                q_optimizer.zero_grad()
                qf_loss.backward()
                q_optimizer.step()

            optimize_actor_on_priming = cur_update_step % args.policy_frequency == 0
            optimize_actor_on_priming = optimize_actor_on_priming and is_heavy_priming_step
            optimize_actor_on_priming = optimize_actor_on_priming and args.optimize_actor_on_priming

            optimize_actor = global_step % args.policy_frequency and not is_heavy_priming_step
            optimize_actor = optimize_actor or optimize_actor_on_priming
            if optimize_actor:
                for _ in range(
                    args.policy_frequency
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    # print("optimizing actor")
                    pi, log_pi, _ = actor.get_action(data.observations)
                    qf1_pi = qf1(data.observations, pi)
                    qf2_pi = qf2(data.observations, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(data.observations)
                        alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

            # update the target networks
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

        if time.time() - last_sps_measuring_time > args.sps_interval:
            n_steps = global_step - last_sps_step
            sps = round(n_steps / args.sps_interval, 4)
            last_sps_step = global_step
            last_sps_measuring_time = time.time()

        if global_step % 100 == 0:
            print(f"{datenow} global_step={global_step}")
            writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
            writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
            writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
            writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
            writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
            writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
            writer.add_scalar("losses/alpha", alpha, global_step)
            # print("SPS:", int(global_step / (time.time() - start_time)))
            print("SPS:", sps)
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
            if args.autotune:
                writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

            if global_step % 1000 != 0:
                continue

            def add_param_group_info(param_group, state):
                param_name = param_group["name"]
                param = param_group["params"][0]
                exp_avg = state["exp_avg"]
                print("adding histograms")
                try:
                    with torch.no_grad():
                        corr_data = [param.data.flatten(), param.grad.flatten(), exp_avg.flatten()]
                        corr = torch.corrcoef(torch.stack(corr_data))
                except:
                    print("can't compute corr")

                try:
                    writer.add_scalar(f"corr/{param_name}_w_grads", corr[0][1], global_step)
                    writer.add_scalar(f"corr/{param_name}_w_expavg", corr[0][2], global_step)
                    writer.add_scalar(f"corr/{param_name}_expavg_grads", corr[1][2], global_step)
                    writer.add_histogram(f"wts/{param_name}", param.data, global_step)
                    writer.add_histogram(f"grad/{param_name}", param.grad, global_step)
                    writer.add_histogram(f"eavg/{param_name}", exp_avg, global_step)
                except:
                    print("can't add histogram, probably after weight resets")

            actor_opt_state = actor_optimizer.state_dict()["state"]
            for i, state in actor_opt_state.items():
                param_group = actor_optimizer.param_groups[i]
                if "name" not in param_group:
                    print("WARNING: Adam param without name; skipping")
                    continue

                add_param_group_info(param_group, state)

            q_opt_state = q_optimizer.state_dict()["state"]
            for i, state in q_opt_state.items():
                param_group = q_optimizer.param_groups[i]
                if "name" not in param_group:
                    print("WARNING: Adam param without name; skipping")
                    continue

                add_param_group_info(param_group, state)

    last_returns = [*map(float, list(avg_returns))]

    envs.close()
    writer.close()

    with open(f"{run_name}_res.txt", "w") as f:
        res = np.average(last_returns)
        res = round(res, 4)
        f.write(str(res))

    with open(f"{run_name}_res_all.json", "w") as f:
        json.dump(last_returns, f)

    with open(f"{run_name}_avg_return_history.json", "w") as f:
        hist = {k: float(v) for k, v in avg_return_history.items()}
        json.dump(hist, f)

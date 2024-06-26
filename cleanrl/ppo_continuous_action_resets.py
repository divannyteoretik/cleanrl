# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import json
import os
import random
import time
from collections import deque
from dataclasses import dataclass
from typing import Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

# reset steps
# reset type: full, based on norms of weights, or based on adam moments
# reset module: none, actor, critic, both
# reset coef: how much do we reset towards random (0 - no reset, 1 - totally random)

"""
all params:

critic.0.weight 1 torch.Size([64, 376])
critic.0.bias 1 torch.Size([64])
critic.2.weight 1 torch.Size([64, 64])
critic.2.bias 1 torch.Size([64])
critic.4.weight 1 torch.Size([1, 64])
critic.4.bias 1 torch.Size([1])

actor_mean.0.weight 1 torch.Size([64, 376])
actor_mean.0.bias 1 torch.Size([64])
actor_mean.2.weight 1 torch.Size([64, 64])
actor_mean.2.bias 1 torch.Size([64])
actor_mean.4.weight 1 torch.Size([17, 64])
actor_mean.4.bias 1 torch.Size([17])
actor_logstd 1 torch.Size([1, 17])
"""

RESET_COEF_DICT = {
    "critic.0": 0.25,
    "critic.2": 0.5,
    "critic.4": 1,
    "actor_mean.0": 0.25,
    "actor_mean.2": 0.5,
    "actor_mean.4": 1,
    "actor_logstd": 1,
}

RESET_COEF_DICT_TMP = {}
for k, v in RESET_COEF_DICT.items():
    if "_logstd" in k:
        RESET_COEF_DICT_TMP[k] = v
        continue
    RESET_COEF_DICT_TMP[f"{k}.bias"] = v
    RESET_COEF_DICT_TMP[f"{k}.weight"] = v
RESET_COEF_DICT = RESET_COEF_DICT_TMP
print(RESET_COEF_DICT)


@dataclass
class Args:
    n_reset_steps: int = 1000
    reset_type: str = "full"
    reset_module: Tuple[str, ...] = ("critic",)
    no_reset_bias: int = 0
    heavy_priming_iters: int = 0
    avg_return_history_step: int = 1000

    n_last_evals: int = 100
    run_name: str = ""

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
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "HalfCheetah-v4"
    """the id of the environment"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    # num_steps: int = 2048
    num_steps: int = 256
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def make_env(env_id, idx, capture_video, run_name, gamma):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


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


def reset_param(param, rand_part, reset_coef, exp_avg=None):
    print(f"performing {args.reset_type} reset on {param_name} with coef {reset_coef}")
    with torch.no_grad():
        if args.reset_type == "full":
            param.data = reset_coef * rand_part + (1 - reset_coef) * param.data
        if "wnorm_abs" in args.reset_type:
            reset_by_abs(param, rand_part, reset_coef, exp_avg)
        if "wnorm_val" in args.reset_type:
            reset_by_val(param, rand_part, reset_coef, exp_avg)


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

        self.bias_const = 0.0
        self.default_std = np.sqrt(2)
        self.weight_stds = {
            "critic.0.weight": self.default_std,
            "critic.2.weight": self.default_std,
            "critic.4.weight": 1.0,
            "actor_mean.0.weight": self.default_std,
            "actor_mean.2.weight": self.default_std,
            "actor_mean.4.weight": 0.01,
        }

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

    reset_coef_str = "_"
    for module, params in RESET_COEF_DICT.items():
        if module not in args.reset_module:
            continue
        reset_coef_str += f"__{module}_"
        for k, v in params.items():
            reset_coef_str += f"{k}x{v}_"

    reset_str = f"{args.n_reset_steps}_{args.reset_type}_{reset_coef_str}"
    if args.no_reset_bias:
        reset_str += "_noresetbias"

    if args.n_reset_steps >= args.total_timesteps:
        reset_str = "noresets"

    args.reset_str = reset_str

    run_name = args.run_name
    if len(run_name) == 0:
        run_name = f"{args.env_id}__{args.exp_name}_{args.total_timesteps}_r{reset_str}"
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

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name, args.gamma) for i in range(args.num_envs)]
    )

    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(envs).to(device)

    param_dict = dict(agent.named_parameters())
    opt_params = []
    for k, v in param_dict.items():
        res_param = {"params": v, "name": k}
        opt_params.append(res_param)

    optimizer = optim.Adam(opt_params, lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    avg_returns = deque(maxlen=args.n_last_evals)
    avg_return_history = {}
    need_add_avg_return = False
    next_avg_return_history_step = 0

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    need_reset = False
    next_reset_step = args.n_reset_steps
    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            if global_step >= next_reset_step and not need_reset:
                need_reset = True
                next_reset_step += args.n_reset_steps

            if global_step >= next_avg_return_history_step and not need_add_avg_return:
                need_add_avg_return = True
                next_avg_return_history_step += args.avg_return_history_step

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                        avg_returns.append(info["episode"]["r"])
                        if need_add_avg_return:
                            avg_ret = round(np.average(avg_returns), 4)
                            print(f"adding average return: step {global_step} return: {avg_ret}")
                            avg_return_history[global_step] = avg_ret
                            need_add_avg_return = False

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []

        heavy_priming = args.heavy_priming_iters > 0 and iteration == 1

        n_updates = args.update_epochs
        if heavy_priming:
            n_updates = args.heavy_priming_iters

        for epoch in range(n_updates):
            if not heavy_priming:
                np.random.shuffle(b_inds)

            batch_size = args.batch_size
            minibatch_size = args.minibatch_size
            if heavy_priming:
                batch_size = 100
                if minibatch_size >= batch_size:
                    minibatch_size = batch_size

            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]
                if heavy_priming and epoch % 100 == 0:
                    print(f"heavy priming; epoch {epoch}; minibatch_size: {len(mb_inds)}")

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                # We loose one update step because of resets right after it
                # maybe it's better to move this code after the resets
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

                # resets here

                # adam state contains values of moments and velocities
                # adam param_groups contains parameters (values of weights + grads)
                # We added names to param_groups previously, and now we can use them
                # to perform resets of particular parts of the network
                adam_dict = optimizer.state_dict()
                adam_state = adam_dict["state"]

                if need_reset:
                    need_reset = False
                    print(f"performing reset ({args.reset_module})")
                    for i, state in adam_state.items():
                        param_group = optimizer.param_groups[i]
                        if "name" not in param_group:
                            print("WARNING: Adam param without name; skipping")
                            continue

                        # WARNING: it's 1-lentgh array in this particular env/model,
                        # not sure whether that's always the case
                        param_name = param_group["name"]
                        param = param_group["params"][0]

                        print(param_name, param.shape, state["exp_avg"].shape)

                        exp_avg = None
                        if "expavg" in args.reset_type:
                            exp_avg = state["exp_avg"]

                        if all(m not in param_name for m in args.reset_module):
                            continue

                        reset_coef = RESET_COEF_DICT[param_name]

                        if "weight" in param_name:
                            std_coef = agent.weight_stds.get(param_name, agent.default_std)
                            param_copy = param.data.clone()
                            rand_part = torch.nn.init.orthogonal_(param_copy, std_coef)
                        if "bias" in param_name or "logstd" in param_name:
                            rand_part = torch.zeros(param.data.shape, device=param.device)

                        reset_param(param, rand_part, reset_coef, exp_avg)

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        print("writing stats to tensorboard")
        for i, state in adam_state.items():
            param_group = optimizer.param_groups[i]
            if "name" not in param_group:
                print("WARNING: Adam param without name; skipping")
                continue

            # WARNING: it's 1-lentgh array in this particular env/model,
            # not sure whether that's always the case
            param_name = param_group["name"]
            param = param_group["params"][0]
            exp_avg = state["exp_avg"]

            with torch.no_grad():
                corr_data = [param.data.flatten(), param.grad.flatten(), exp_avg.flatten()]
                corr = torch.corrcoef(torch.stack(corr_data))

            writer.add_scalar(f"correlations/{param_name}_w_grads", corr[0][1], global_step)
            writer.add_scalar(f"correlations/{param_name}_w_expavg", corr[0][2], global_step)
            writer.add_scalar(f"correlations/{param_name}_expavg_grads", corr[1][2], global_step)

            writer.add_histogram(f"weights/{param_name}", param.data, global_step)
            writer.add_histogram(f"grads/{param_name}", param.grad, global_step)
            writer.add_histogram(f"exp_avgs/{param_name}", exp_avg, global_step)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")
        from cleanrl_utils.evals.ppo_eval import evaluate

        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=Agent,
            device=device,
            gamma=args.gamma,
        )
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

        if args.upload_model:
            from cleanrl_utils.huggingface import push_to_hub

            repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
            repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
            push_to_hub(args, episodic_returns, repo_id, "PPO", f"runs/{run_name}", f"videos/{run_name}-eval")

    envs.close()
    writer.close()
    last_returns = [*map(float, list(avg_returns))]

    with open(f"{run_name}_res.txt", "w") as f:
        res = np.average(last_returns)
        res = round(res, 4)
        f.write(str(res))

    with open(f"{run_name}_res_all.json", "w") as f:
        json.dump(last_returns, f)

    with open(f"{run_name}_avg_return_history.json", "w") as f:
        hist = {k: float(v) for k, v in avg_return_history.items()}
        json.dump(hist, f)

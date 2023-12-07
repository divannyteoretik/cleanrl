# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqnpy
# https://github.com/HamedCoding/DQLFrozenLake/blob/main/main_4x4.py

import os
import random
import time
from dataclasses import dataclass
from gymnasium.spaces.utils import flatten_space, unflatten

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from gymnasium.wrappers.flatten_observation import FlattenObservation
import pandas as pd
import matplotlib.pyplot as plt

@dataclass
class Args:
    exp_name: str = 'dqn'
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
    env_id: str = "FrozenLake-v1"
    """the id of the environment"""
    n_episodes: int = 500
    """total timesteps of the experiments"""
    learning_starts: int = 3
    """timestep to start learning"""
    learning_rate: float = 0.001
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = 10000
    """the replay memory buffer size"""
    gamma: float = 0.9
    """the discount factor gamma"""
    tau: float = 1.0
    """the target network update rate"""
    batch_size: int = 32 #512 #32 lowered from 512 to 64 at 12:35pm on Dec6
    """the batch size of sample from the reply memory"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.01
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.001/3 #0.5
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    train_frequency: int = 1
    """the frequency of training"""
    replay_ratio: int = 1
    """network updates per observation"""
    target_network_frequency: int = 5
    """target_network_frequency"""
    n_tests = 100
    """number of tests"""
    weight_reset = 0 # 0 for no resets


from fl_envs import make_env, make_env_sol


def levenshteinDistance(s1, s2):
    # from https://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Levenshtein_distance#Python
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]


def score_path(args, device, Model, q_network, env_count):
    target_path = make_env_sol(args, env_count)
    with torch.no_grad():
    
        envs, _ = make_env(args, -1, test=True, env_count=env_count)
        obs, _ = envs.reset()
        
        agent = Model(envs).to(device)
        agent.load_state_dict(q_network.state_dict())
        agent.eval()
    
        n_test = 0
        logger = []
        my_path = ''
        
        while True:
            actions = q_network(torch.Tensor(obs).to(device))
            action = int(torch.argmax(actions))
            my_path += str(action)
    
            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, rewards, terminations, truncations, infos = envs.step(action)
            # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
            obs = next_obs
    
            # TRY NOT TO MODIFY: record rewards for plotting purposes
            if "episode" in infos:
                break

        # print (my_path)
        # print (my_path)
        score = levenshteinDistance(target_path, my_path) # score is number of deviations from target path
        return score


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(env.observation_space.shape[0], 10),
            nn.ReLU(),
            nn.Linear(10, flatten_space(env.action_space).shape[0]),
        )

    def forward(self, x):
        return self.network(x)


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)



import stable_baselines3 as sb3
import uuid

def run_trial(args, verbose=False):
    run_name = str(uuid.uuid4())
    
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    
    # TRY NOT TO MODIFY: seeding
    # random.seed(args.seed)
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    # torch.backends.cudnn.deterministic = args.torch_deterministic
    
    device = 'cpu' #torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    
    # env setup
    env_label = 3 # start in map 3 if markov otherwise ignore
    envs, env_label = make_env(args, env_label, False, 0)
    
    q_network = QNetwork(envs).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = QNetwork(envs).to(device)
    target_network.load_state_dict(q_network.state_dict())
    
    rb = ReplayBuffer(
        args.buffer_size,
        flatten_space(envs.observation_space),
        envs.action_space,
        device,
        handle_timeout_termination=False,
    )
    start_time = time.time()
    
    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    episodes = 0
    global_step = 0
    logger = []
    score_list = []
    
    while True:
        global_step += 1
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(args.start_e, args.end_e, args.n_episodes, episodes)
        
        if args.weight_reset > 0 and global_step % args.weight_reset == 0:
            # full reset
            del q_network, optimizer, target_network
            # print ("reset!", global_step)
            q_network = QNetwork(envs).to(device)
            optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
            target_network = QNetwork(envs).to(device)
            target_network.load_state_dict(q_network.state_dict())

        if random.random() < epsilon:
            actions = torch.rand(size=(1, 4))[0]
            action = int(torch.argmax(actions))
        else:
            actions = q_network(torch.Tensor(obs).to(device)).detach()
            action = int(torch.argmax(actions))
    
        next_obs, rewards, terminations, truncations, infos = envs.step(action)
    
        if episodes > args.n_episodes:
            break
        
        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        rb.add(obs, real_next_obs, np.array([[action]]), rewards, terminations, infos)
    
        if 'episode' in infos:
            result = {"episodes": episodes, "length": infos["episode"]["l"][0], 'reward': infos["episode"]["r"][0]}
            envs, env_label = make_env(args, env_label, False, episodes)
            obs, _ = envs.reset() #seed=args.seed)
            episodes += 1
    
        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if episodes > args.learning_starts:
            for rr in range(args.replay_ratio):
                data = rb.sample(args.batch_size)
                with torch.no_grad(): 
                    target_max, _ = target_network(data.next_observations.type(torch.float32)).max(dim=1)
                    td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
                old_val = q_network(data.observations.type(torch.float32)).gather(1, data.actions).squeeze()
                loss = F.mse_loss(td_target, old_val)
        
                if global_step % 100 == 0:
                    writer.add_scalar("losses/td_loss", loss, global_step)
                    writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        
                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    
                # update target 
                if global_step % args.target_network_frequency == 0:
                    for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                        target_network_param.data.copy_(
                            args.tau * q_network_param.data + (1.0 - args.tau) * target_network_param.data
                        )
    
        score_list.append(score_path(args, device, QNetwork, q_network, episodes))

    return q_network, logger, rb, f"runs/{run_name}/{args.exp_name}.cleanrl_model.eval=*", score_list



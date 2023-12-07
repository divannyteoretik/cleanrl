from gymnasium.spaces.utils import flatten_space, unflatten
import gymnasium as gym
from gymnasium.wrappers.flatten_observation import FlattenObservation
import numpy as np


def map_3x3_slippery(test=False):

    map1 = ['SFF', 
            'FFF', 
            'HFG']
    
    if test:
        env = gym.make("FrozenLake-v1", desc=map1, is_slippery=False, max_episode_steps=10)

    else:
        env = gym.make("FrozenLake-v1", desc=map1, is_slippery=True, max_episode_steps=10)
        
    env = FlattenObservation(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    return env, -1


def map_3x3_random(test=False):

    map1 = ['SFF', 
            'FFF', 
            'HFG']

    map2 = ['SFF', 
            'FHF', 
            'FFG']

    map3 = ['SFH', 
            'FFF', 
            'FFG']
    
    if test:
        output_map = map1

    else:
        rand = np.random.uniform()
        if rand < 0.6:
            output_map = map1
        elif rand < 0.9:
            output_map = map2
        else:
            output_map = map3
    
    env = gym.make("FrozenLake-v1", desc=output_map, is_slippery=False, max_episode_steps=10)
    env = FlattenObservation(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    return env, -1


def map_3x3_markov(prev_map, test=False):

    map1 = ['SFF', 
            'FFF', 
            'HFG']

    map2 = ['SFF', 
            'FHF', 
            'FFG']

    map3 = ['SFH', 
            'FFF', 
            'FFG']
    
    if test:
        output_map = map1
        env_label = prev_map
    else:
        rand = np.random.uniform()
        if prev_map == 1:
            if rand < 0.9:
                output_map = map1
                env_label = 1
            else:
                output_map = map2
                env_label = 2
        elif prev_map == 2:
            if rand < 0.05:
                output_map = map3
                env_label = 3
            elif rand < 0.35:
                output_map = map1
                env_label = 1
            else:
                output_map = map2
                env_label = 2
        elif prev_map == 3:
            if rand < 0.9:
                output_map = map3
                env_label = 3
            else:
                output_map = map2
                env_label = 2
        else:
            assert (False and "wtf")
    
    env = gym.make("FrozenLake-v1", desc=output_map, is_slippery=False, max_episode_steps=10)
    env = FlattenObservation(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    return env, env_label


def map_3x3_markov2(prev_map, test=False):

    map1 = ['SFF', 
            'FFF', 
            'HFG']

    map2 = ['SFF', 
            'FHF', 
            'FFG']

    map3 = ['SFH', 
            'FFF', 
            'FFG']
    
    if test:
        output_map = map1
        env_label = prev_map
    else:
        rand = np.random.uniform()
        if prev_map == 1:
            if rand < 0.5:
                output_map = map1
                env_label = 1
            else:
                output_map = map2
                env_label = 2
        elif prev_map == 2:
            if rand < 0.0:
                output_map = map3
                env_label = 3
            elif rand < 0.5:
                output_map = map1
                env_label = 1
            else:
                output_map = map2
                env_label = 2
        elif prev_map == 3:
            if rand < 0.98:
                output_map = map3
                env_label = 3
            else:
                output_map = map2
                env_label = 2
        else:
            assert (False and "wtf")
    
    env = gym.make("FrozenLake-v1", desc=output_map, is_slippery=False, max_episode_steps=10)
    env = FlattenObservation(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    return env, env_label





def map_3x3_markov3(prev_map, test=False):

    map1 = ['SFF', 
            'FFF', 
            'HFG']

    map2 = ['SFF', 
            'FHF', 
            'FFG']

    map3 = ['SFH', 
            'FFF', 
            'FFG']
    
    if test:
        output_map = map1
        env_label = prev_map
    else:
        rand = np.random.uniform()
        if prev_map == 1:
            if rand < 0.5:
                output_map = map1
                env_label = 1
            else:
                output_map = map2
                env_label = 2
        elif prev_map == 2:
            if rand < 0.333:
                output_map = map3
                env_label = 3
            elif rand < 0.667:
                output_map = map1
                env_label = 1
            else:
                output_map = map2
                env_label = 2
        elif prev_map == 3:
            if rand < 0.3:
                output_map = map3
                env_label = 3
            else:
                output_map = map2
                env_label = 2
        else:
            assert (False and "wtf")
    
    env = gym.make("FrozenLake-v1", desc=output_map, is_slippery=False, max_episode_steps=10)
    env = FlattenObservation(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    return env, env_label


def map_3x3_temporal(n):
    if n < 100:
        output_map = ['SFH', 
                      'FHF', 
                      'FFG']
    else:
        output_map = ['SFF', 
                      'FHF', 
                      'HFG']
        
    env = gym.make("FrozenLake-v1", desc=output_map, is_slippery=False, max_episode_steps=10)
    env = FlattenObservation(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    return env, -1


def make_env(args, prev_map, test, env_count=1):
    if args.map_rule == '3x3_slippery':
        return map_3x3_slippery(test)
    elif args.map_rule == '3x3_random':
        return map_3x3_random(test)
    elif args.map_rule == '3x3_markov':
        return map_3x3_markov(prev_map, test)
    elif args.map_rule == '3x3_markov2':
        return map_3x3_markov2(prev_map, test)
    elif args.map_rule == '3x3_markov3':
        return map_3x3_markov3(prev_map, test)
    elif args.map_rule == '3x3_temporal':
        return map_3x3_temporal(env_count)
    else:
        assert (False)
        


def make_env_sol(args, n=None):
    if args.map_rule == '3x3_temporal':
        if n < 100:
            return '1122'
        else:
            return '2211'
    else:
        return '2211' 


    

import hydra
import pandas as pd
import mbrl
# from mbrl.models import ModelEnv
# from mbrl.third_party.pytorch_sac_pranz24 import SAC
from typing import cast
import os
from os.path import join as osp

import torch
from mbrl.third_party.pytorch_sac import VideoRecorder

import yaml
from mbrl.algorithms.bde_mbpo import evaluate

import gymnasium as gym

from mbrl.planning.sac_wrapper import SACAgent
from mbrl.third_party import pytorch_sac_pranz24

import argparse

# import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import omegaconf

import mbrl.env.reward_fns as reward_fns
import mbrl.env.termination_fns as termination_fns
import mbrl.models as models
import mbrl.planning as planning
import mbrl.util.common as common_util
import mbrl.util as util
import mbrl.util.env

from review.utils import *

def evaluate_sample_true(agent, env, num_episodes=25):
    reward_lst = []
    for _ in range(num_episodes):
        obs, _ = env.reset()
        terminated = False
        truncated = False
        episode_reward = 0.0
        while not terminated and not truncated:
            action = agent.act(obs, sample=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
        
        reward_lst.append(episode_reward)

    return reward_lst


def check_reward(
        env,
        bde_agent,
        mbpo_agent,
):

    video_recorder = VideoRecorder(None)
    num_episodes = 25
    bde_re_lst_1 = evaluate_sample_true(bde_agent, env, num_episodes)
    bde_re_lst_2 = [evaluate(env=env, agent=bde_agent, num_episodes=1, video_recorder=video_recorder) for _ in range(num_episodes)]
    mbpo_re_lst_1 = evaluate_sample_true(mbpo_agent, env, num_episodes)
    mbpo_re_lst_2 = [evaluate(env=env, agent=mbpo_agent, num_episodes=1, video_recorder=video_recorder) for _ in range(num_episodes)]

    df = pd.DataFrame(
        {
            "BDE_sample":bde_re_lst_1,
            "BDE":bde_re_lst_2,
            "MBPO_sample":mbpo_re_lst_1,
            "MBPO":mbpo_re_lst_2,
        }
    )
    df['diff_bde'] = np.abs(df["BDE_sample"] - df["BDE"])
    df['diff_mbpo'] = np.abs(df["MBPO_sample"] - df["MBPO"])
    sns.boxplot(df.loc[:,  ['BDE_sample', 'BDE', 'MBPO_sample', 'MBPO']])
    

if __name__ == "__main__":

    # Argument for our experiment
    args = argparse.ArgumentParser(description="Args for Visualization")
    args.add_argument("--N_s", type=int, default=10)
    args = args.parse_args()

    noise_scale = 0.0001
    seed = 1
    rollout_horizon = 12
    idx = 20
    trial_length = 1000

    bde_cfg = create_config(
        seed = seed,
        agent_type="bde", 
        env_name="hopper", 
        trial_length=trial_length,
        N_s=args.N_s
    )
    mbpo_cfg = create_config(
        seed = seed,
        agent_type="mbpo", 
        env_name="hopper", 
        trial_length=trial_length
    )

    log_dir = 'review/'

    # Get agent and model directory
    bde_agent_dir, bde_model_dir = get_model_dir("bde")
    mbpo_agent_dir, mbpo_model_dir = get_model_dir("mbpo")

    torch_generator = torch.Generator(device=bde_cfg.device)
    if bde_cfg.seed is not None:
        torch_generator.manual_seed(bde_cfg.seed)

    env, term_fn, reward_fn = mbrl.util.env.EnvHandler.make_env(bde_cfg)
    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape

    # dynamics model
    dynamics_model = mbrl.util.common.create_one_dim_tr_model(bde_cfg, obs_shape, act_shape)
    rng = np.random.default_rng(seed=bde_cfg.seed)
    
    use_double_dtype = bde_cfg.algorithm.get("normalize_double_precision", False)
    dtype = np.double if use_double_dtype else np.float32
    
    #########  create replay buffer  #########
    replay_buffer = mbrl.util.common.create_replay_buffer(
        bde_cfg,
        obs_shape,
        act_shape,
        rng=rng,
        obs_type=dtype,
        action_type=dtype,
        reward_type=dtype,
        collect_trajectories = True
    )
    ##########################################

    ##########  create model env  ############
    model_env = create_model_env(
        env,
        bde_cfg,
        dynamics_model,
        term_fn,
        torch_generator,
        'bde'
    )
    ##########################################

    ############## create agent ##############
    mbrl.planning.complete_agent_cfg(env, bde_cfg.algorithm.agent)
    bde_agent = SACAgent(
        cast(pytorch_sac_pranz24.SAC, hydra.utils.instantiate(bde_cfg.algorithm.agent))
    )
    mbrl.planning.complete_agent_cfg(env, mbpo_cfg.algorithm.agent)
    mbpo_agent = SACAgent(
        cast(pytorch_sac_pranz24.SAC, hydra.utils.instantiate(mbpo_cfg.algorithm.agent))
    )

    ##########################################

    ############## load model ################
    bde_agent.sac_agent.load_checkpoint(bde_agent_dir)
    mbpo_agent.sac_agent.load_checkpoint(mbpo_agent_dir)

    model_env.dynamics_model.load(bde_model_dir)
    ##########################################

    ############## debug ##############

    agent_kwargs = {
            'sample':False, 'batched':False
    }

    # 确认 bde 的 sample=True 是有问题的 
    # check_reward(env, bde_agent, mbpo_agent)
    # 基本上，bde 的 sample=True 和 sample=False 的结果差异较大

    # TODO: 确认 bde 的 sample=True 和 sample=False 的结果差异较大的原因
    # 1. 是否是因为 action 的差异导致的
    # 确认思路：将 action 的差异可视化
    # diff0 = check_action_diff(env, bde_agent, mbpo_agent, num_episodes=10, action_type=0)
    # diff1 = check_action_diff(env, bde_agent, mbpo_agent, num_episodes=10, action_type=1)
    # diff2 = check_action_diff(env, bde_agent, mbpo_agent, num_episodes=10, action_type=2)
    # diff3 = check_action_diff(env, bde_agent, mbpo_agent, num_episodes=10, action_type=3)

    episode = 0
    episode1 = 0
    while (episode <= 8):
        obs, _ = env.reset()
        terminated = False
        truncated = False
        episode_reward = 0.0
        diff_lst = []
        while not terminated and not truncated:
            diff = []

            action = bde_agent.act(obs, sample=True)
            action_determined = bde_agent.act(obs, sample=False)
            action_mbpo_sample = mbpo_agent.act(obs, sample=True)
            action_mbpo_determined = mbpo_agent.act(obs, sample=False)

            diff.append(np.linalg.norm(action_determined - action))
            diff.append(np.linalg.norm(action_mbpo_determined - action_mbpo_sample))
            diff.append(np.linalg.norm(action_mbpo_determined - action_determined))
            diff_lst.append(diff)

            # TODO: change action
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward

        print("episode_reward: ", episode_reward)
        if episode_reward < 2300:
            episode += 1
            diff_lst = np.vstack(diff_lst)
            plt.figure(dpi=300, figsize=(15, 4))
            plt.plot(diff_lst[:, 0], label="bde_diff_action")
            plt.plot(diff_lst[:, 1], label="mbpo_diff_action")
            # plt.plot(diff_lst[:, 2], label="bde_mbpo_mean_diff_action")
            plt.legend()
            plt.savefig(f"review/check4sample_true/diff_action_le2300_{episode}.png")

        if episode_reward > 3300:
            episode1 += 1
            diff_lst = np.vstack(diff_lst)
            plt.figure(dpi=300, figsize=(15, 4))
            plt.plot(diff_lst[:, 0], label="bde_diff_action")
            plt.plot(diff_lst[:, 1], label="mbpo_diff_action")
            # plt.plot(diff_lst[:, 2], label="bde_mbpo_mean_diff_action")
            plt.legend()
            plt.savefig(f"review/check4sample_true/diff_action_ge3300_{episode1}.png")

    video_recorder = VideoRecorder(None)
    print("##############")
    print("evaluate for bde")
    for _ in range(25):
        print(evaluate(env=env, agent=bde_agent, num_episodes=1, video_recorder=video_recorder))
    print("##############")

    print("##############")
    print("evaluate for mbpo")
    for _ in range(25):
        print(evaluate(env=env, agent=mbpo_agent, num_episodes=1, video_recorder=video_recorder))
    print("##############")
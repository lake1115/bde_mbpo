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

if __name__ == "__main__":

    # Argument for our experiment
    args = argparse.ArgumentParser(description="Args for Visualization")
    args.add_argument("--noise_scale", type=float, default=0)
    args.add_argument("--seed", type=int, default=101)
    args.add_argument("--exp_seed", type=int, default=0)
    args.add_argument("--rollout_horizon", type=int, default=12)
    args.add_argument("--idx", type=int, default=296)
    args.add_argument("--log_dir", type=str, default="noise_results")
    args.add_argument("--env_name", type=str, default="hopper")
    args.add_argument("--N_s", type=int, default=10)
    args.add_argument("--agent_type", type=str, default="bde")
    args = args.parse_args()
    noise_scale = args.noise_scale
    seed = args.seed
    rollout_horizon = args.rollout_horizon
    idx = args.idx
    trial_length = 1000

    cfg = create_config(
        seed = seed,
        agent_type=args.agent_type, 
        env_name="hopper", 
        trial_length=trial_length,
        N_s=args.N_s
    )

    # NOTE: change this to your own directory
    root_dir = 'review/'
    log_dir = osp(root_dir, f'noise_results_seed{args.exp_seed}_rl{rollout_horizon}_idx{idx}_agent_{args.agent_type}')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Get agent and model directory
    agent_dir, model_dir = get_model_dir(args.agent_type,args.exp_seed)

    df_denoise_name = osp(log_dir , f'idx{idx}_seed{args.exp_seed}_rl{rollout_horizon}_df_denoise.csv')
    df_noise_name = osp(log_dir , f'idx{idx}_seed{args.exp_seed}_rl{rollout_horizon}_df_noise.csv')

    torch_generator = torch.Generator(device=cfg.device)
    if cfg.seed is not None:
        torch_generator.manual_seed(cfg.seed)

    env, term_fn, reward_fn = mbrl.util.env.EnvHandler.make_env(cfg)
    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape

    # dynamics model
    dynamics_model = mbrl.util.common.create_one_dim_tr_model(cfg, obs_shape, act_shape)
    rng = np.random.default_rng(seed=cfg.seed)
    
    use_double_dtype = cfg.algorithm.get("normalize_double_precision", False)
    dtype = np.double if use_double_dtype else np.float32
    
    #########  create replay buffer  #########
    replay_buffer = mbrl.util.common.create_replay_buffer(
        cfg,
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
    model_env1 = create_model_env(
        env,
        cfg,
        dynamics_model,
        term_fn,
        torch_generator,
        args.agent_type
    )
    model_env2 = create_model_env(
        env,
        cfg,
        dynamics_model,
        term_fn,
        torch_generator,
        args.agent_type
    )
    ##########################################

    ############## create agent ##############
    # TODO Create agent and load model 是使用什么
    mbrl.planning.complete_agent_cfg(env, cfg.algorithm.agent)
    agent = SACAgent(
        cast(pytorch_sac_pranz24.SAC, hydra.utils.instantiate(cfg.algorithm.agent))
    )
    ##########################################

    ############## load model ################
    agent.sac_agent.load_checkpoint(agent_dir)
    model_env1.dynamics_model.load(model_dir)
    model_env2.dynamics_model.load(model_dir)
    ##########################################

    ############## collect data ##############

    seed_everything(seed)
    rewards = mbrl.util.common.rollout_agent_trajectories(
        env = env,
        steps_or_trials_to_collect = 1,
        agent = agent,
        agent_kwargs = {
            'sample':False, 'batched':False
        },
        trial_length=1000,
        replay_buffer = replay_buffer,
        collect_full_trajectories=True,
        agent_uses_low_dim_obs = False,
        seed= cfg.seed,
    )
    print("collected rewards" , rewards)
    
    video_recorder = VideoRecorder(None)
    ave_reward = evaluate(
        env, agent, num_episodes=5, video_recorder = video_recorder
    )
    print("ave_reward", ave_reward)

    ##########################################

    ########## sample from replay buffer ##########
    # sample from replay buffer
    trans_batch = replay_buffer.sample_trajectory()
    obs = trans_batch.obs[[idx]]
    # action = trans_batch.act[[idx]]
    action = agent.act(obs, batched=True)
    
    env_steps = 0
    log_model_weights = True

    model_state = model_env1.reset(
        initial_obs_batch=cast(np.ndarray, obs),
        return_as_np=True,
    )
    ###############################################

    # rollout
    seed_everything(seed)
    for i in range(rollout_horizon):
        if i < (rollout_horizon - 1): 
            end_factor = False
        else: end_factor=True
        pred_next_obs, pred_rewards, pred_dones, model_state = model_env1.step(
                agent, 
                False, 
                action, 
                model_state, 
                sample=True, 
                end_factor=end_factor,
                epoch=i,
                iteration=env_steps,
                log_dir=log_dir,
                log_model_weights=log_model_weights
            )

    noise = np.random.normal(0, noise_scale, obs.shape)
    model_state = model_env2.reset(
        initial_obs_batch=cast(np.ndarray, obs+noise),
        return_as_np=True,
    )
    action = agent.act(obs+noise, batched=True)
    
    seed_everything(seed)
    # model_env._rng = torch_generator
    for i in range(rollout_horizon):
        if i < (rollout_horizon - 1): 
            end_factor = False
        else: end_factor=True

        pred_next_obs, pred_rewards, pred_dones, model_state = model_env2.step(
                agent, 
                False, 
                action, 
                model_state, 
                sample=True, 
                end_factor=end_factor,
                epoch=i,
                iteration=env_steps,
                log_dir=log_dir,
                log_model_weights=log_model_weights
            )

    # delete model_weights
    df_all = pd.read_csv(os.path.join(log_dir, 'model_weights.csv'))
    df_all = df_all.drop(columns=['Unnamed: 0'])
    df_all = df_all[df_all['epoch'] != 'epoch']
    df_all = df_all.astype({'epoch': 'int32', 'iteration': 'int32'})
    
    df_denoise = df_all.iloc[:rollout_horizon-1, :]
    df_noise = df_all.iloc[rollout_horizon-1:, :]

    df_noise.to_csv(df_noise_name, index=False)
    df_denoise.to_csv(df_denoise_name, index=False)

    plt.figure(figsize=(10, 4), dpi=300)
    
    for i in range(5):
        plt.plot(df_denoise['epoch'], df_denoise[f'{i}'], label=f'{i}')
    plt.legend()
    plt.savefig(osp(log_dir , f'idx{idx}_seed{args.exp_seed}_rl{rollout_horizon}_denoise.png'))
    plt.cla()

    plt.figure(figsize=(10, 4), dpi=300)
    for i in range(5):
        plt.plot(df_noise['epoch'], df_noise[f'{i}'], label=f'{i}')
    plt.legend()
    plt.savefig(osp(log_dir , f'idx{idx}_seed{args.exp_seed}_rl{rollout_horizon}_noise.png'))
    plt.cla()
    os.remove(os.path.join(log_dir, 'model_weights.csv'))


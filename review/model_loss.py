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
    args.add_argument("--rollout_horizon", type=int, default=30)
    args.add_argument("--idx", type=int, default=260)
    args.add_argument("--log_dir", type=str, default="noise_results")
    args.add_argument("--env_name", type=str, default="hopper")
    args.add_argument("--N_s", type=int, default=30)
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

    root_dir = 'review/'
    log_dir = osp(root_dir, f'model_loss_seed{args.exp_seed}_rl{rollout_horizon}_idx{idx}_agent_{args.agent_type}')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Get agent and model directory
    agent_dir, model_dir = get_model_dir(args.agent_type, seed = args.exp_seed)
    
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
    
    ##########  create model env  ############
    model_env = create_model_env(
        env, cfg, dynamics_model, term_fn, torch_generator, agent_type=args.agent_type
    )
    ##########################################

    mbrl.planning.complete_agent_cfg(env, cfg.algorithm.agent)
    agent = SACAgent(
        cast(pytorch_sac_pranz24.SAC, hydra.utils.instantiate(cfg.algorithm.agent))
    )

    ############## load model ################
    agent.sac_agent.load_checkpoint(agent_dir)
    model_env.dynamics_model.load(model_dir)

    ##########################################
    df_model_weights_name = osp(log_dir , f'idx{idx}_seed{args.exp_seed}_rl{rollout_horizon}_df_model_weights.csv')

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

    # trans_batch = replay_buffer.sample_trajectory()
    # # sample from replay buffer
    # obs_batch = trans_batch.obs
    # act_bacth = trans_batch.act
    obs_batch = np.loadtxt(osp(root_dir, 'obs.csv'), delimiter=',')
    act_bacth = np.loadtxt(osp(root_dir, 'act.csv'), delimiter=',')

    obs = obs_batch[[idx]]
    action = act_bacth[[idx]]
    
    env_steps = 0
    log_model_weights = True

    if args.agent_type == 'mbpo':
        obs = obs.repeat(5, 0)
        action = action.repeat(5, 0)

    model_state = model_env.reset(
        initial_obs_batch=cast(np.ndarray, obs),
        return_as_np=True,
    )
    pred_next_obs_lst = []
    seed_everything(seed)
    for i in range(rollout_horizon):
        if args.agent_type == 'bde':
            pred_next_obs, pred_rewards, pred_dones, model_state = model_env.step(
                    agent, 
                    False, 
                    action, 
                    model_state, 
                    sample=False, 
                    end_factor=False,
                    epoch=i,
                    iteration=env_steps,
                    log_dir=log_dir,
                    log_model_weights=True
                )
            pred_next_obs_lst.append(pred_next_obs)
        else:
            pred_next_obs, pred_rewards, pred_dones, model_state = model_env.step(
                    action,
                    model_state,
                    sample=False
                )
            pred_next_obs_lst.append(pred_next_obs[random.randint(0,4),:])

    pred_next_obs_lst = np.vstack(pred_next_obs_lst)
    obs_gt = obs_batch[idx:idx+rollout_horizon]
    print(f"{args.agent_type}_L2 loss:" ,np.linalg.norm(pred_next_obs_lst - obs_gt, axis=1))
    print(f"{args.agent_type}_L2 loss mean:" ,np.mean(np.linalg.norm(pred_next_obs_lst - obs_gt, axis=1)))
    if args.agent_type == 'bde':
        os.remove(os.path.join(log_dir, 'model_weights.csv'))


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

def create_config(seed):
    with open("mbrl/examples/conf/main.yaml") as f:
        cfg = yaml.load(f, yaml.FullLoader)

    with open("mbrl/examples/conf/algorithm/bde_mbpo.yaml") as f:
        algo_cfg = yaml.load(f, yaml.FullLoader)

    with open("mbrl/examples/conf/dynamics_model/gaussian_bde_mlp_ensemble.yaml") as f:
        dyna_model_cfg = yaml.load(f, yaml.FullLoader)
        
    with open("mbrl/examples/conf/overrides/mbpo_hopper.yaml") as f:
        overrides_cfg = yaml.load(f, yaml.FullLoader)

    for k, v in algo_cfg['agent']['args'].items():
        if v.find("overrides") != -1: 
            v = v.replace("${overrides.", "")
            v = v.replace("}", "")  
            algo_cfg['agent']['args'][k] = overrides_cfg[v]
        else: 
            v = v.replace("${", "")
            v = v.replace("}", "")  
            algo_cfg['agent']['args'][k] = cfg[v]
    algo_cfg['freq_train_model'] = overrides_cfg['freq_train_model']

    dyna_model_cfg['device'] = cfg['device']

    cfg["overrides"] = overrides_cfg
    cfg["algorithm"] = algo_cfg
    cfg["dynamics_model"] = dyna_model_cfg
    cfg_dict = cfg

    cfg = omegaconf.OmegaConf.create(cfg_dict)

    cfg.seed = seed
    cfg.overrides.trial_length = 1000
    return cfg

if __name__ == "__main__":
    # python vis_bde.py --noise_scale 0.0001 --seed 101 --rollout_horizon 12 --idx 10

    args = argparse.ArgumentParser(description="Args for Visualization")
    args.add_argument("--seed", type=int, default=101)
    args.add_argument("--rollout_horizon", type=int, default=12)
    args.add_argument("--idx", type=int, default=243)
    args.add_argument("--agent_type", type=str, default="mbpo")
    args = args.parse_args()

    seed = args.seed
    rollout_horizon = args.rollout_horizon
    idx = args.idx

    if args.agent_type == 'bde':
        agent_dir = 'exp/bde_mbpo/bde_hopper_test/gym___Hopper-v4_Ns10_rl0_seed101_2023.10.12_160547/sac.pth'
        model_dir = 'exp/bde_mbpo/bde_hopper_test/gym___Hopper-v4_Ns10_rl0_seed101_2023.10.12_160547'
    elif args.agent_type == 'mbpo':
        agent_dir = 'exp/bde_mbpo/mbpo_hopper_test/seed101/sac.pth'
        model_dir = 'exp/bde_mbpo/mbpo_hopper_test/seed101'
    elif args.agent_type == 'mean':
        agent_dir = 'exp/bde_mbpo/mean_hopper_test/gym___Hopper-v4_Ns10_rl0_seed101_2023.10.13_145811/sac.pth'
        model_dir = 'exp/bde_mbpo/mean_hopper_test/gym___Hopper-v4_Ns10_rl0_seed101_2023.10.13_145811'
    
        
    cfg = create_config(seed=seed)
    
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
    model_env = mbrl.models.ModelEnv(
        env, dynamics_model, term_fn, None, generator=torch_generator
        , alpha = cfg.overrides.bde_alpha
        , N_s = cfg.overrides.N_s
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
    os.chdir('cycle')
    log_dir = os.getcwd()
    df_model_weights_name = osp(log_dir , f'idx{idx}_seed{seed}_rl{rollout_horizon}_df_model_weights.csv')

    # sample from replay buffer
    obs_batch = np.loadtxt('obs.csv', delimiter=',')
    act_bacth = np.loadtxt('act.csv', delimiter=',')
    obs = obs_batch[[idx]]
    action = act_bacth[[idx]]
    
    env_steps = 0
    log_model_weights = True

    model_state = model_env.reset(
        initial_obs_batch=cast(np.ndarray, obs),
        return_as_np=True,
    )

    pred_next_obs_lst = []
    for i in range(rollout_horizon):
        pred_next_obs, pred_rewards, pred_dones, model_state = model_env.step(
                agent, 
                cfg.algorithm.sac_samples_action, 
                action, 
                model_state, 
                sample=True, 
                end_factor=False,
                epoch=i,
                iteration=env_steps,
                log_dir=log_dir,
                log_model_weights=log_model_weights
            )
        pred_next_obs_lst.append(pred_next_obs)
    pred_next_obs_lst = np.vstack(pred_next_obs_lst)
    obs_gt = obs_batch[idx:idx+rollout_horizon]
    print("L2 loss:" , np.mean((pred_next_obs_lst - obs_gt)**2))

    os.remove(os.path.join(log_dir, 'model_weights.csv'))


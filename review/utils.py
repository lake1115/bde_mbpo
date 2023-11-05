import random
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

def create_config(
        seed,
        agent_type="bde",
        env_name="hopper",
        N_s=10,
        trial_length=1000
):

    assert agent_type in ["bde", "mbpo"], "agent_type must be either bde or mbpo or mean"
    assert env_name in ["hopper", "walker", "halfcheetah", "ant", "humanoid"], "env_name must be either hopper, walker, halfcheetah, ant, or humanoid"

    if agent_type == "bde":
        agent_type = "bde_mbpo"

    with open("mbrl/examples/conf/main.yaml") as f:
        cfg = yaml.load(f, yaml.FullLoader)

    with open(f"mbrl/examples/conf/algorithm/{agent_type}.yaml") as f:
        algo_cfg = yaml.load(f, yaml.FullLoader)

    policy_f_name = 'gaussian_bde_mlp_ensemble' if agent_type == 'bde_mbpo' else 'gaussian_mlp_ensemble'
    with open(f"mbrl/examples/conf/dynamics_model/{policy_f_name}.yaml") as f:
        dyna_model_cfg = yaml.load(f, yaml.FullLoader)
        
    with open(f"mbrl/examples/conf/overrides/{agent_type}_{env_name}.yaml") as f:
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
    cfg.overrides.trial_length = trial_length
    cfg.overrides.N_s = N_s

    return cfg

def get_model_dir(
        agent_type,
        seed
):

    assert agent_type in ["bde", "mbpo", "mean"], "agent_type must be either bde or mbpo or mean"

    if agent_type == 'bde':
        agent_dir = f'model/model_achieve/seed{seed}_bde_hopper/sac.pth'
        model_dir = f'model/model_achieve/seed{seed}_bde_hopper'
    elif agent_type == 'mbpo':
        agent_dir = f'model/model_achieve/seed{seed}_mbpo_hopper/sac.pth'
        model_dir = f'model/model_achieve/seed{seed}_mbpo_hopper'
    elif agent_type == 'mean':
        agent_dir = 'exp/bde_mbpo/mean_hopper_test/gym___Hopper-v4_Ns10_rl0_seed101_2023.10.13_145811/sac.pth'
        model_dir = 'model/model_achieve/seed0_mbpo_hopper'

    return agent_dir, model_dir

def create_model_env(
        env, 
        cfg,
        dynamics_model,
        term_fn,
        torch_generator,
        agent_type="bde",
):
    
    assert agent_type in ["bde", "mbpo", "mean"], "agent_type must be either bde or mbpo or mean"

    if agent_type in ["bde", "mean"]:

        model_env = mbrl.models.BDE_ModelEnv(
            env, dynamics_model, term_fn, None, generator=torch_generator
            , alpha = cfg.overrides.bde_alpha
            , N_s = cfg.overrides.N_s
        )
    else:
        model_env = mbrl.models.ModelEnv(
            env, dynamics_model, term_fn, None, generator=torch_generator
        )

    return model_env


def check_action_diff(
        env,
        bde_agent,
        mbpo_agent,
        num_episodes=10,
        action_type=0
):
    
    assert action_type in [0, 1, 2, 3], "action_type must be either 0, 1, 2, or 3"
    diff_mean = np.zeros((num_episodes, 3))

    for episode in range(num_episodes):
        obs, _ = env.reset()
        terminated = False
        truncated = False
        episode_reward = 0.0
        diff_lst = []
        while not terminated and not truncated:
            diff = []

            action_sample = bde_agent.act(obs, sample=True)
            action_determined = bde_agent.act(obs, sample=False)
            action_mbpo_sample = mbpo_agent.act(obs, sample=True)
            action_mbpo_determined = mbpo_agent.act(obs, sample=False)

            diff.append(np.linalg.norm(action_determined - action_sample))
            diff.append(np.linalg.norm(action_mbpo_determined - action_mbpo_sample))
            diff.append(np.linalg.norm(action_mbpo_determined - action_determined))
            diff_lst.append(diff)

            if action_type == 0:
                action = action_sample
            elif action_type == 1:
                action = action_determined
            elif action_type == 2:
                action = action_mbpo_sample
            elif action_type == 3:
                action = action_mbpo_determined

            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward

        diff_mean[episode] = np.mean(np.array(diff_lst), axis=0)
        print(f"episode_reward with action_type{action_type}: ", episode_reward)
    return diff_mean

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

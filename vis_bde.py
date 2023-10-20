import hydra
import pandas as pd
import mbrl
# from mbrl.models import ModelEnv
# from mbrl.third_party.pytorch_sac_pranz24 import SAC
from typing import cast
import os

import torch

import yaml

# from easydict import EasyDict as edict

import gymnasium as gym

from mbrl.planning.sac_wrapper import SACAgent
from mbrl.third_party import pytorch_sac_pranz24

import argparse

# import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import omegaconf

# import mbrl.env.reward_fns as reward_fns
# import mbrl.env.termination_fns as termination_fns
# import mbrl.models as models
# import mbrl.planning as planning
# import mbrl.util.common as common_util
# import mbrl.util as util
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

    ############################################################################################################

    # def find_config(cfg):
    #     for k, v in cfg.items():
    #         if (type(v) == dict):
    #             print()
    #             print(k)
    #             find_config(v)
    #         if (type(v) == str) and (v.startswith("${")):
    #             print(k, v)

    # find_config(cfg)

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
    args.add_argument("--noise_scale", type=float, default=0.0001)
    args.add_argument("--seed", type=int, default=101)
    args.add_argument("--rollout_horizon", type=int, default=25)
    args.add_argument("--idx", type=int, default=20)
    args = args.parse_args()

    noise_scale = args.noise_scale
    seed = args.seed
    rollout_horizon = args.rollout_horizon
    idx = args.idx

    
    log_dir = 'log_model_vis/'
    df_denoise_name = log_dir + f'idx{idx}_seed{seed}_rl{rollout_horizon}_df_denoise.csv'
    df_noise_name = log_dir + f'idx{idx}_seed{seed}_rl{rollout_horizon}_df_noise.csv'

    cfg = create_config(seed=seed)

    torch_generator = torch.Generator(device=cfg.device)
    if cfg.seed is not None:
        torch_generator.manual_seed(cfg.seed)

    env, term_fn, reward_fn = mbrl.util.env.EnvHandler.make_env(cfg)
    # get obs and act shape
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
    model_env = mbrl.models.ModelEnv(
        env, dynamics_model, term_fn, None, generator=torch_generator
        , alpha = cfg.overrides.fogetting_alpha
        , N_s = cfg.overrides.N_s
    )
    ##########################################

    mbrl.planning.complete_agent_cfg(env, cfg.algorithm.agent)
    agent = SACAgent(
        cast(pytorch_sac_pranz24.SAC, hydra.utils.instantiate(cfg.algorithm.agent))
    )

    ############## load model ################
    agent.sac_agent.load_checkpoint('exp/bde_mbpo/bde_hopper_test/gym___Hopper-v4_Ns10_rl0_seed101_2023.10.12_160547/sac.pth')
    model_env.dynamics_model.load('exp/bde_mbpo/bde_hopper_test/gym___Hopper-v4_Ns10_rl0_seed101_2023.10.12_160547')
    
    ############## collect data ##############
    rewards = mbrl.util.common.rollout_agent_trajectories(
        env = env,
        steps_or_trials_to_collect = 1,
        agent = agent,
        agent_kwargs = {
            'sample':cfg.algorithm.sac_samples_action, 'batched':False
        },
        trial_length=1000,
        replay_buffer = replay_buffer,
        collect_full_trajectories=True,
        agent_uses_low_dim_obs = False,
        seed= cfg.seed,
    )
    print("collected rewards" , rewards)
    # replay_buffer.load('exp/bde_mbpo/bde_hopper_test/gym___Hopper-v4_Ns10_rl0_seed101_2023.10.12_160547')
    ##########################################

    # sample from replay buffer
    tans_batch = replay_buffer.sample_trajectory()
    obs = tans_batch.obs[[idx]]
    action = tans_batch.act[[idx]]
    # replay_buffer.sample_trajectory()
    # action = agent.act(obs, sample=cfg.algorithm.sac_samples_action, batched=True)
    
    env_steps = 0
    log_model_weights = True

    model_state = model_env.reset(
        initial_obs_batch=cast(np.ndarray, obs),
        return_as_np=True,
    )

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

    # add noise
    noise = np.random.normal(0, noise_scale, obs.shape)
    
    model_state = model_env.reset(
        initial_obs_batch=cast(np.ndarray, obs+noise),
        return_as_np=True,
    )
    
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
    plt.savefig(log_dir + f'idx{idx}_seed{seed}_rl{rollout_horizon}_denoise.png')
    plt.cla()

    plt.figure(figsize=(10, 4), dpi=300)
    for i in range(5):
        plt.plot(df_noise['epoch'], df_noise[f'{i}'], label=f'{i}')
    plt.legend()
    plt.savefig(log_dir + f'idx{idx}_seed{seed}_rl{rollout_horizon}_noise.png')
    plt.cla()
    os.remove(os.path.join(log_dir, 'model_weights.csv'))


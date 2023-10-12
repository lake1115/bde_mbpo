# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import hydra
import numpy as np
import omegaconf
import torch

import sys, os
sys.path.insert(0,os.getcwd())
os.system('nvidia-smi -q -d Memory | grep -A5 GPU | grep Free > tmp.txt')
memory_gpu = [int(x.split()[2]) for x in open('tmp.txt', 'r').readlines()]
os.environ["CUDA_VISIBLE_DEVICES"] = str(np.argmax(memory_gpu)) 
os.system('rm tmp.txt')

import mbrl.algorithms.mbpo as mbpo
import mbrl.algorithms.pets as pets
import mbrl.algorithms.planet as planet
import mbrl.algorithms.bde_mbpo as bde_mbpo
import mbrl.util.env


@hydra.main(config_path="conf", config_name="main")
def run(cfg: omegaconf.DictConfig):
    env, term_fn, reward_fn = mbrl.util.env.EnvHandler.make_env(cfg)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if cfg.algorithm.name == "pets":
        return pets.train(env, term_fn, reward_fn, cfg)
    if cfg.algorithm.name == "mbpo":
        test_env, *_ = mbrl.util.env.EnvHandler.make_env(cfg)
        return mbpo.train(env, test_env, term_fn, cfg)
    if cfg.algorithm.name == "bde_mbpo":
        test_env, *_ = mbrl.util.env.EnvHandler.make_env(cfg)
        return bde_mbpo.train(env, test_env, term_fn, cfg)
    if cfg.algorithm.name == "planet":
        return planet.train(env, cfg)


if __name__ == "__main__":
    run()

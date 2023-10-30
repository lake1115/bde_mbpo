# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from copy import deepcopy
import os
from typing import Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
from torch.distributions import Categorical

import mbrl.types
from . import ModelEnv
from . import Model
from . import GaussianMeanMLP
from . import GaussianMLP
import pandas as pd

class BDE_ModelEnv(ModelEnv):
    """Wraps a dynamics model into a gym-like environment.

    This class can wrap a dynamics model to be used as an environment. The only requirement
    to use this class is for the model to use this wrapper is to have a method called
    ``predict()``
    with signature `next_observs, rewards = model.predict(obs,actions, sample=, rng=)`

    Args:
        env (gym.Env): the original gym environment for which the model was trained.
        model (:class:`mbrl.models.Model`): the model to wrap.
        termination_fn (callable): a function that receives actions and observations, and
            returns a boolean flag indicating whether the episode should end or not.
        reward_fn (callable, optional): a function that receives actions and observations
            and returns the value of the resulting reward in the environment.
            Defaults to ``None``, in which case predicted rewards will be used.
        generator (torch.Generator, optional): a torch random number generator (must be in the
            same device as the given model). If None (default value), a new generator will be
            created using the default torch seed.
    """

    def __init__(
        self,
        env: gym.Env,
        model: Model,
        termination_fn: mbrl.types.TermFnType,
        reward_fn: Optional[mbrl.types.RewardFnType] = None,
        generator: Optional[torch.Generator] = None,
        alpha: float = 0.5,
        N_s: int = 3,
    ):
        self.dynamics_model = model
        self.termination_fn = termination_fn
        self.reward_fn = reward_fn
        self.device = model.device

        self.observation_space = env.observation_space
        self.action_space = env.action_space

        self._current_obs: torch.Tensor = None
        self._propagation_method: Optional[str] = None
        self._model_indices = None
        if generator:
            self._rng = generator
        else:
            self._rng = torch.Generator(device=self.device)
        self._return_as_np = True

        self.N_s = N_s
        # Torch Tensor [N_s, batch_size, in_size]
        self.dataset = None 
        # Torch Tensor [N_s, batch_size]
        self.dataset_weights = None
        self.alpha = alpha 

    def reset(
        self, initial_obs_batch: np.ndarray, return_as_np: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Resets the model environment.

        Args:
            initial_obs_batch (np.ndarray): a batch of initial observations. One episode for
                each observation will be run in parallel. Shape must be ``B x D``, where
                ``B`` is batch size, and ``D`` is the observation dimension.
            return_as_np (bool): if ``True``, this method and :meth:`step` will return
                numpy arrays, otherwise it returns torch tensors in the same device as the
                model. Defaults to ``True``.

        Returns:
            (dict(str, tensor)): the model state returned by `self.dynamics_model.reset()`.
        """
        if isinstance(self.dynamics_model, mbrl.models.OneDTransitionRewardModel):
            assert len(initial_obs_batch.shape) == 2  # batch, obs_dim
        with torch.no_grad():
            model_state = self.dynamics_model.reset(
                initial_obs_batch.astype(np.float32), rng=self._rng
            )
        self._return_as_np = return_as_np
        return model_state if model_state is not None else {}

    def step(
        self,
        agent,
        sac_samples_action,
        actions: mbrl.types.TensorType,
        model_state: Dict[str, torch.Tensor],
        sample: bool = False,
        end_factor: bool = False,
        epoch: int = 0,
        iteration: int = 0,
        log_dir: str = None,
        log_model_weights: bool = True
    ) -> Tuple[mbrl.types.TensorType, mbrl.types.TensorType, np.ndarray, Dict]:
        """Steps the model environment with the given batch of actions.

        Args:
            agent 
            sac_sample_action 
            actions (torch.Tensor or np.ndarray): the actions for each "episode" to rollout.
                Shape must be ``B x A``, where ``B`` is the batch size (i.e., number of episodes),
                and ``A`` is the action dimension. Note that ``B`` must correspond to the
                batch size used when calling :meth:`reset`. If a np.ndarray is given, it's
                converted to a torch.Tensor and sent to the model device.
            model_state (dict(str, tensor)): the model state as returned by :meth:`reset()`.
            sample (bool): if ``True`` model predictions are stochastic. Defaults to ``False``.

        Returns:
            (tuple): contains the predicted next observation, reward, done flag and metadata.
            The done flag is computed using the termination_fn passed in the constructor.
        """
        assert len(actions.shape) == 2  # batch, action_dim
        with torch.no_grad():
            # if actions is tensor, code assumes it's already on self.device
            if isinstance(actions, np.ndarray):
                if not hasattr(self.dynamics_model.model, 'model_weights'):
                    isPF = False
                elif self.dynamics_model.model.model_weights is None:
                    isPF = True
                    self._init_model_weights(actions.shape[0])
                else: isPF = True

                actions = torch.from_numpy(actions).to(self.device)
            
            # Note: model_state won't change after dynamics_model.sample()
            model_state_copy = {}
            for key, value in model_state.items():
                model_state_copy[key] = value
        
            # model_state_copy = deepcopy(model_state)
            '''
            if epoch ==1:
                print(epoch)
            '''
            try:
                (
                    next_observs,
                    pred_rewards,
                    pred_terminals,
                    next_model_state,
                ) = self.dynamics_model.sample(
                    actions,
                    model_state_copy,
                    deterministic = not sample,
                    rng=self._rng,
                )
            except:
                print("Error: ", epoch, iteration)
                print("model_state_copy: ", model_state_copy)
                print("actions: ", actions)
                self.dynamics_model.sample(
                    actions,
                    model_state_copy,
                    deterministic = not sample,
                    rng=self._rng,
                )
                raise ValueError("NaN in model output")

            rewards = (
                pred_rewards
                if self.reward_fn is None
                else self.reward_fn(actions, next_observs)
            )
            dones = self.termination_fn(actions, next_observs)

            # We determine if it's the first call to step by whether dataset is None or not
            # If it is the first call to step, we need to initialize dataset and dataset_weights, 
            # as well as model_weights
            if not isPF:
                pass
            elif self.dataset is None:
                # dict{str, tensor: [N_s*batch_size, obs_dim]}, [N_s*batch_size, action_dim]
                # (x_0)
                action_dim = actions.shape[-1]
                sampled_actions = actions.repeat(self.N_s, 1, 1).reshape(-1, action_dim)
                sampled_model_state = {}
                current_observs = model_state['obs']
                obs_dim = current_observs.shape[-1]
                sampled_model_state['obs'] = current_observs.repeat(self.N_s, 1, 1).reshape(-1, obs_dim)
                
                del(current_observs)
                sampled_model_state['propagation_indices'] = None #model_state['propagation_indices']

                # [N_s*batch_size, *]
                (_,_,_, sampled_next_model_state) = self.dynamics_model.sample(
                    sampled_actions,
                    sampled_model_state,
                    deterministic=not sample,
                    rng=self._rng,
                )
                # [N_s*batch_size, obs_dim]
                sampled_actions = agent.act(sampled_next_model_state['obs'].cpu().numpy(), 
                                        sample=sac_samples_action, 
                                        batched=True)
                
                sampled_actions = torch.from_numpy(sampled_actions).to(self.device)

                # Update self.dataset: Tensor [N_s, batch_size, in_size]
                self._output_to_x(sampled_next_model_state, sampled_actions)
                self._init_dataset_weights()

            else:
                # [N_s, batch_size]
                j_i = self._sample_j_i() 
                # [N_s, batch_size, in_size]
                x_j_i = self._sample_x_j_i(j_i)
                # dict{str, tensor: [N_s*batch_size, obs_dim]}, [N_s*batch_size, action_dim]
                sampled_model_state, sampled_actions = self._x_to_input(x_j_i)
                current_observs = sampled_model_state['obs']
                
                (sampled_next_model_state,sampled_pred_rewards,_, _) = self.dynamics_model.sample(
                    sampled_actions,
                    sampled_model_state,
                    deterministic=not sample,
                    rng=self._rng,
                )

                sampled_actions = agent.act(sampled_next_model_state.cpu().numpy(), 
                                            sample=sac_samples_action, 
                                            batched=True)
                sampled_actions = torch.from_numpy(sampled_actions).to(self.device)

                # Update self.dataset: Tensor [N_s, batch_size, in_size]
                self._output_to_x({'obs':sampled_next_model_state}, sampled_actions)

                # likelihood: [N_s, batch_size, num_models]
                # log_likelihood: [N_s, batch_size], Update self.dataset_weights
                likelihood, log_likelihood = self._compute_likelihood(next_observs
                                                      , model_state['obs']
                                                      , pred_rewards)
                
                # estimate p_m(y_t|y_{0:t-1})
                # self.dataset_weights: [N_s, batch_size]
                # p_m_y: [batch_size, num_models]
                p_m_y = (likelihood * self.dataset_weights.unsqueeze(-1)).sum(dim=0)

                # Update self.dynamic_model.model.model_weights
                self._update_dynamic_model_weights(p_m_y)
                
                # update self.dataset_weights: [N_s, batch_size] # TODO
                self._update_dataset_weights(log_likelihood)

            if (log_model_weights) and (isPF) and (epoch > 0):
                # log model_weights [batch_size, num_models]
                log_dir = os.path.join(log_dir, "model_weights.csv")
                model_weights = self.dynamics_model.model.model_weights.cpu().numpy()
                
                model_weights = pd.DataFrame(model_weights)
                # index: batch; header: model
                model_weights['epoch'] = epoch
                model_weights['iteration'] = iteration
                model_weights.reset_index(inplace=True)
                model_weights.to_csv(log_dir, mode='a', header=True, index=True)

            if (end_factor == True) and isPF:
                self._set_init_state()
            

            if pred_terminals is not None:
                raise NotImplementedError(
                    "ModelEnv doesn't yet support simulating terminal indicators."
                )

            if self._return_as_np:
                next_observs = next_observs.cpu().numpy()
                rewards = rewards.cpu().numpy()
                dones = dones.cpu().numpy()
            return next_observs, rewards, dones, next_model_state
    
    def _process_model_output(self):
        pass

    def _set_init_state(self):
        '''
        1. Set self.dynamics_model.model.model_weights to None
        2. Set self.dataset to None
        3. Set self.dataset_weights to None
        '''
        self.dynamics_model.model.model_weights = None
        self.dataset = None
        self.dataset_weights = None

    def _init_dataset_weights(self):
        '''
        Initialize self.dataset_weights: [N_s, batch_size]
        '''
        self.dataset_weights = torch.ones(self.N_s, self.dataset.shape[1]) / self.N_s
        self.dataset_weights = self.dataset_weights.to(self.device)

    def _init_model_weights(self, batch_size: int):
        '''
        Initialize self.model_weights
        '''
        # num_models = self.dynamics_model.model
        model_weights = torch.softmax(torch.ones(batch_size, len(self.dynamics_model.model.elite_models)),dim=1)
        self.dynamics_model.model.model_weights = model_weights.to(self.device)
        

    def _sample_j_i(self) -> torch.Tensor:
        # [N_s, batch_size]
        j_i = []
        for i in range(self.dataset_weights.shape[1]):
            probs = self.dataset_weights[:, i]
            dist = Categorical(probs) 
            j_i.append(dist.sample(torch.tensor([self.N_s]))) # a list of N_s-length tensor

        j_i = torch.stack(j_i, dim=1)
        return j_i

    def _x_to_input(self, new_dataset: torch.Tensor):
        """
        Args:
            - new_dataset: torch.Tensor [N_s, batch_size, in_size]
        
        Return:
            - model_state: dict{str, tensor: [N_s*batch_size, obs_dim]}
            - actions: [N_s*batch_size, action_dim] 
        """
        model_state = {}
        model_state['obs'] = new_dataset[:, :, :self.observation_space.shape[0]].reshape(-1, self.observation_space.shape[0])
        
        actions = new_dataset[:, :, self.observation_space.shape[0]:]
        model_state['propagation_indices'] = None

        return model_state, actions.reshape(-1, self.action_space.shape[0])

    def _output_to_x(self, model_state: dict, actions: torch.Tensor):
        '''
        Update self.dataset with sampled model_state and actions

        Args: 
            - model_state:  dict{str, tensor: [N_s*batch_size, obs_dim]}
            - actions: [N_s*batch_size, action_dim]
        '''
        self.dataset = torch.cat((model_state['obs'].reshape(self.N_s,-1, self.observation_space.shape[0])
                                  , actions.reshape(self.N_s, -1, self.action_space.shape[0])), dim=-1)

    def _sample_x_j_i(self, j_i: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            - j_i: [N_s, batch_size]
            
        Return:
            - x_j_i: [N_s, batch_size, in_size]
        '''
        N_s = j_i.shape[0]
        batch_size = j_i.shape[1]
        # TODO: find an elegant way to do this
        x_j_i = self.dataset[j_i, 
                             torch.arange(batch_size).repeat(N_s).reshape(N_s, batch_size)]
        return x_j_i
        # raise NotImplementedError

    def _compute_likelihood(self, next_observs: torch.Tensor,
                            model_state: torch.Tensor, 
                            pred_rewards: torch.Tensor):
        '''
        Compute the likelihood of the dataset given the model.
        And the arguments are used to compute y appeared in the likelihood function.

        Args:
            - next_observs: [batch_size, obs_dim]
            - model_state: [batch_size, obs_dim]
            - pred_rewards: [batch_size, ]
        ---
        Return:
            - likelihood: [N_s, batch_size, num_models]
            - log_likelihood: [N_s, batch_size]
        '''

        if len(pred_rewards.shape) < 2:
            pred_rewards = pred_rewards.unsqueeze(-1)
        # y is the concatenation of (next_observs - obs) and pred_rewards
        # y: [N_s*batch_size, output_size]
        y = torch.cat((next_observs - model_state, pred_rewards), dim=-1)
        y = y.repeat(self.N_s, 1, 1).reshape(-1, y.shape[-1])

        # self.dataset: [N_s, batch_size, in_size]
        batch_size = self.dataset.shape[1]
        

        # normalize model_in
        if self.dynamics_model.input_normalizer:
            # model_in: [N_s*batch_size, in_size]
            model_in = self.dataset.reshape(self.N_s*batch_size, -1)
            model_in = self.dynamics_model.input_normalizer.normalize(model_in.float())
            # reshape model_in -> [N_s, batch_size, in_size]
            model_in = model_in.reshape(self.N_s, batch_size, -1)

        # likelihood: [N_s*batch_size, num_models] 
        # log_likelihood: [N_s, batch_size]
        likelihood, log_likelihood = self.dynamics_model.model.compute_likelihood(model_in, y)
        
        return likelihood.reshape(self.N_s, batch_size, -1), log_likelihood

    def _update_dataset_weights(self, likelihood: torch.Tensor):
        '''
        Update self.dataset_weights

        Args:
            - likelihood: [N_s, batch_size, num_models]
        '''

        # [N_s, batch_size]
        self.dataset_weights = torch.nn.functional.softmax(likelihood, dim=0)
        

    def _update_dynamic_model_weights(self, p_m_y: torch.Tensor):
        ''' 
        ---
        Args:
            - p_m_y: [batch_size, num_models]
        '''
        # prev_model_weights: [batch_size, num_models]
        prev_model_weights = self.dynamics_model.model.model_weights
        # forgetting part: [batch_size, num_models]
        forgetting_part = prev_model_weights ** self.alpha / \
            torch.sum(prev_model_weights ** self.alpha, dim=-1, keepdim=True)
        # update model_weights: [batch_size, num_models]
        self.dynamics_model.model.model_weights = forgetting_part * p_m_y / \
            torch.sum(forgetting_part * p_m_y, dim=-1, keepdim=True)

    def render(self, mode="human"):
        pass

    def evaluate_action_sequences(
        self,
        action_sequences: torch.Tensor,
        initial_state: np.ndarray,
        num_particles: int,
    ) -> torch.Tensor:
        """Evaluates a batch of action sequences on the model.

        Args:
            action_sequences (torch.Tensor): a batch of action sequences to evaluate.  Shape must
                be ``B x H x A``, where ``B``, ``H``, and ``A`` represent batch size, horizon,
                and action dimension, respectively.
            initial_state (np.ndarray): the initial state for the trajectories.
            num_particles (int): number of times each action sequence is replicated. The final
                value of the sequence will be the average over its particles values.

        Returns:
            (torch.Tensor): the accumulated reward for each action sequence, averaged over its
            particles.
        """
        with torch.no_grad():
            assert len(action_sequences.shape) == 3
            population_size, horizon, action_dim = action_sequences.shape
            # either 1-D state or 3-D pixel observation
            assert initial_state.ndim in (1, 3)
            tiling_shape = (num_particles * population_size,) + tuple(
                [1] * initial_state.ndim
            )
            initial_obs_batch = np.tile(initial_state, tiling_shape).astype(np.float32)
            model_state = self.reset(initial_obs_batch, return_as_np=False)
            batch_size = initial_obs_batch.shape[0]
            total_rewards = torch.zeros(batch_size, 1).to(self.device)
            terminated = torch.zeros(batch_size, 1, dtype=bool).to(self.device)
            for time_step in range(horizon):
                action_for_step = action_sequences[:, time_step, :]
                action_batch = torch.repeat_interleave(
                    action_for_step, num_particles, dim=0
                )
                _, rewards, dones, model_state = self.step(
                    action_batch, model_state, sample=True
                )
                rewards[terminated] = 0
                terminated |= dones
                total_rewards += rewards

            total_rewards = total_rewards.reshape(-1, num_particles)
            return total_rewards.mean(dim=1)

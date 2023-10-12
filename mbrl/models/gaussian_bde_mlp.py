from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np

import omegaconf
import torch
from torch import nn as nn
from torch.nn import functional as F
import torch.distributions as D
from .gaussian_mlp import GaussianMLP

class GaussianBDEMLP(GaussianMLP):
    def __init__(
        self,
        in_size: int,
        out_size: int,
        device: Union[str, torch.device],
        num_layers: int = 4,
        ensemble_size: int = 1,
        hid_size: int = 200,
        deterministic: bool = False,
        propagation_method: Optional[str] = None,
        learn_logvar_bounds: bool = False,
        activation_fn_cfg: Optional[Union[Dict, omegaconf.DictConfig]] = None,
    ):
        super().__init__(in_size,out_size,device,num_layers,ensemble_size,hid_size,deterministic,propagation_method,learn_logvar_bounds,activation_fn_cfg)

        self.model_weights = None # Torch Tensor [batch_size, num_elite_models]

    '''
    def _maybe_toggle_layers_weights_use_only_elite(self, only_elite: bool):

        if self.elite_models is None:
            return
        else:
            self.model_weights = self.model_weights[self.elite_models]

    def _mth_model_likelihood(self, x: torch.Tensor, y: torch.Tensor, model_lst: Optional[Sequence[int]]):
        """
        Compute the likelihood of the data given the model.
        """
        self._toggle_specific_models(model_lst)
        x = self.hidden_layers(x)
        mean_and_logvar = self.mean_and_logvar(x)
        self._toggle_specific_models(model_lst)

        mean = mean_and_logvar[..., : self.out_size]
        logvar = mean_and_logvar[..., self.out_size :]
        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)
        # TODO: check the dimension of the likelihood
        normal_dist = D.MultivariateNormal(mean, torch.diag(torch.exp(logvar)))
        likelihood = normal_dist.log_prob(y)

        return likelihood
    
    def _toggle_specific_models(self, model_lst: Optional[Sequence[int]]):
        
        if self.num_members > 1 and model_lst is not None:
            for layer in self.hidden_layers:
                # each layer is (linear layer, activation_func)
                layer[0].set_elite(model_lst)
                layer[0].toggle_use_only_elite()
            self.mean_and_logvar.set_elite(model_lst)
            self.mean_and_logvar.toggle_use_only_elite()
        
    '''
    def set_elite(self, elite_indices: Sequence[int]):
        self.elite_models = list(elite_indices)
    def forward(  # type: ignore
        self,
        x: torch.Tensor,
        rng: Optional[torch.Generator] = None,
        propagation_indices: Optional[torch.Tensor] = None,
        use_propagation: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        if use_propagation:
            # model inference
            return self._forward_ensemble(
                x, rng=rng, propagation_indices=propagation_indices
            )
        return self._default_forward(x)

    def _forward_ensemble(
        self,
        x: torch.Tensor,
        rng: Optional[torch.Generator] = None,
        propagation_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the forward pass of the ensemble model.
        """
        if self.model_weights.shape[0] != x.shape[0]:  ## 这里是N_s固定为1了？
            N_s = x.shape[0] // self.model_weights.shape[0]
        else: 
            N_s = 1

        # x: [batch_size, in_size] 
        # -> mean, logvar: [num_elite_models, batch_size, out_size] 
        #               or [num_elite_models, N_s*batch_size, out_size]
        mean, logvar = self._default_forward(x, only_elite=True)
        # self.model_weights: [batch_size, num_elite_models]
        num_elite_models = len(self.elite_models)
        # model_weights: [num_elite_models, N_s*batch_size, 1]
        model_weights = self.model_weights.repeat(N_s, 1, 1).reshape(-1, num_elite_models).T.unsqueeze(-1)

        mean = torch.sum(mean*model_weights, dim=0) 
        logvar = torch.sum(logvar*model_weights, dim=0)

        # [batch_size, out_size]
        return mean, logvar

    def compute_likelihood(self, model_in:torch.Tensor, y: torch.Tensor):
        '''
        Compute the likelihood of the data given the model.
        
        Args:
            - model_in: [N_s, batch_size, in_size]
            - y: [N_s * batch_size, out_size]
        ---
        Returns:
            - likelihood: [N_s*batch_size, num_elite_models]
            - log_likelihood: [N_s, batch_size]
        '''
        N_s, batch_size, in_size = model_in.shape
        model_in = model_in.reshape(-1, in_size)

        # model_in: [N_s*batch_size, in_size] -> mean, logvar: [num_elite_models, N_s*batch_size, out_size]
        mean, logvar = self._default_forward(model_in.float(), only_elite=True)
        
        outsize = mean.shape[-1]
        # mean, logvar: [num_elite_models, N_s, batch_size, out_size]
        mean, logvar = mean.reshape(-1, N_s, batch_size, outsize), logvar.reshape(-1, N_s, batch_size, outsize)

        # likelihood: [num_elite_models, N_s, batch_size]

        likelihood, log_likelihood = self._compute_likehood(mean, logvar, y)
        # likelihood: [N_s*batch_size, num_elite_models]
        likelihood = likelihood.permute(1,2,0).reshape(-1, len(self.elite_models))
        return likelihood, log_likelihood
    
    def _compute_likehood(self, mean:torch.Tensor, logvar:torch.Tensor, y: torch.Tensor):
        '''
        Compute likelihood of the ensemble model and single model.

        Args:
            - mean: [num_elite_models, N_s, batch_size, out_size]
            - logvar: [num_elite_models, N_s, batch_size, out_size]
            - y: [N_s * batch_size, out_size]
        ---
        Returns:
            - likelihood: [num_elite_models, N_s, batch_size]
            - log_likelihood: [N_s, batch_size]
        '''

        num_elite_models, N_s, batch_size, out_size = mean.shape

        # compute log likelihood for ensemble model        
        log_likelihood = torch.zeros(N_s*batch_size).to(self.device)

        model_weights = self.model_weights.repeat(N_s, 1, 1).reshape(-1, num_elite_models).T.unsqueeze(-1)
        # [N_s*batch_size, out_size]
        ensemble_mean = torch.sum(mean.reshape(-1, N_s*batch_size, out_size)*model_weights, dim=0) 
        ensemble_var = torch.exp(torch.sum(logvar.reshape(-1, N_s*batch_size, out_size)*model_weights, dim=0))

        pdf = []
        for i in range(num_elite_models):
            dist = D.Normal(mean[i].reshape(N_s*batch_size, out_size), torch.sqrt(logvar[i].reshape(N_s*batch_size, out_size).exp()))
            pdf.append(torch.mean(torch.exp(dist.log_prob(y)),dim=-1))
        pdf = torch.stack(pdf,dim=0).reshape(-1, N_s, batch_size)
        likelihood = nn.Softmax(dim=0)(pdf)
        dist = D.Normal(ensemble_mean, torch.sqrt(ensemble_var))
        log_likelihood = torch.mean(dist.log_prob(y),dim=-1).reshape(N_s, batch_size)

        return likelihood, log_likelihood

if __name__ == "__main__":
    pass

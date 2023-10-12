#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   gaussian_mixture_mlp.py
@Time    :   2023/08/22 15:06:54
@Author  :   Hu Bin 
@Version :   1.0
@Desc    :   None
'''
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import hydra
import omegaconf
import torch
from torch import nn as nn
from torch.nn import functional as F
import torch.distributions as D
import mbrl.util.math
from einops import rearrange, repeat
from .gaussian_mlp import GaussianMLP
from .util import EnsembleLinearLayer, truncated_normal_init

class GaussianMixtureMLP(GaussianMLP):
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
        n_components: int = 3,
    ):
        super().__init__(in_size,out_size,device,num_layers,ensemble_size,hid_size,deterministic,propagation_method,learn_logvar_bounds,activation_fn_cfg)

        self.n_components = n_components

        def create_linear_layer(l_in, l_out):
            return EnsembleLinearLayer(ensemble_size, l_in, l_out)
        
        if not deterministic:
            self.mean_and_logvar = create_linear_layer(hid_size, 2 * out_size * self.n_components)
            self.weight = create_linear_layer(hid_size, self.n_components)
            self.min_logvar = nn.Parameter(
                -10 * torch.ones(1, self.n_components * out_size), requires_grad=learn_logvar_bounds
            )
            self.max_logvar = nn.Parameter(
                0.5 * torch.ones(1, self.n_components * out_size), requires_grad=learn_logvar_bounds
            )
        self.to(self.device)

    def _maybe_toggle_layers_use_only_elite(self, only_elite: bool):
        if self.elite_models is None:
            return
        if self.num_members > 1 and only_elite:
            for layer in self.hidden_layers:
                # each layer is (linear layer, activation_func)
                layer[0].set_elite(self.elite_models)
                layer[0].toggle_use_only_elite()
            self.mean_and_logvar.set_elite(self.elite_models)
            self.mean_and_logvar.toggle_use_only_elite()
            self.weight.set_elite(self.elite_models)
            self.weight.toggle_use_only_elite()

    def _default_forward(
        self, x: torch.Tensor, only_elite: bool = False, **_kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        self._maybe_toggle_layers_use_only_elite(only_elite)
        x = self.hidden_layers(x)
        mean_and_logvar = self.mean_and_logvar(x)
        weight = self.weight(x)
        self._maybe_toggle_layers_use_only_elite(only_elite)
        if self.deterministic:
            return mean_and_logvar, None
        else:
            means = mean_and_logvar[..., : self.out_size*self.n_components]
            logvars = mean_and_logvar[..., self.out_size*self.n_components :]
            weights = torch.nn.Softmax(dim=-1)(weight)
            logvars = self.max_logvar - F.softplus(self.max_logvar - logvars)
            logvars = self.min_logvar + F.softplus(logvars - self.min_logvar)
            means = rearrange(means, 'e b (d c) -> e b d c', c = self.n_components)
            logvars = rearrange(logvars, 'e b (d c) -> e b d c', c = self.n_components)
            weights = rearrange(weights, 'e b (1 c) -> e b 1 c', c = self.n_components)
            return means, logvars, weights
        
    def _forward_ensemble(        
        self,
        x: torch.Tensor,
        rng: Optional[torch.Generator] = None,
        propagation_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        model_len = (
            len(self.elite_models) if self.elite_models is not None else len(self)
        )       
        if x.shape[0] % model_len != 0:
            raise ValueError(
                f"GaussianMixtureMLP ensemble requires batch size to be a multiple of the "
                f"number of models. Current batch size is {x.shape[0]} for "
                f"{model_len} models."
            )
        x = x.unsqueeze(0)  
        copied_x = x.repeat(model_len, 1, 1)
        means, logvars, weights = self._default_forward(copied_x, only_elite=True)
        e, b, d, c = means.shape
        variances = logvars.exp()
        avg_mean = torch.sum(means*weights,dim=-1,keepdim=True)
        mean_bias = (means - avg_mean)**2
        # see https://export.arxiv.org/pdf/1709.02249.pdf
        uncertainty = torch.sum(weights * variances ,dim=(-1,-2)) + torch.sum(weights * mean_bias, dim=(-1,-2))  
        model_to_use = torch.argmin(uncertainty, dim=0)
        mask = repeat(model_to_use, 'b -> 1 b d c', d=d, c=c)
        means = means.gather(0, mask).squeeze()
        logvars = logvars.gather(0, mask).squeeze()
        mask = repeat(model_to_use, 'b -> 1 b 1 c', c=c)
        weights = weights.gather(0, mask).squeeze()
     
        return means, logvars, weights
    
    def sample_1d(
        self,
        model_input: torch.Tensor,
        model_state: Dict[str, torch.Tensor],
        deterministic: bool = False,
        rng: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
 
        if deterministic or self.deterministic:
            return (
                self.forward(
                    model_input,
                    rng=rng,
                    propagation_indices=model_state["propagation_indices"],
                )[0],
                model_state,
            )
        assert rng is not None
        means, logvars, weights = self.forward(
            model_input, rng=rng, propagation_indices=model_state["propagation_indices"]
        )
        variances = logvars.exp()
        stds = torch.sqrt(variances)
        mix = D.Categorical(weights)
        comp = D.Independent(D.Normal(means.transpose(-1,-2), stds.transpose(-1,-2)),1)
        gmm = D.MixtureSameFamily(mix, comp)

        return gmm.sample(), model_state
    

    def gmm_nll(
        self,
        pred_mean: torch.Tensor,
        pred_logvar: torch.Tensor,
        pred_weight: torch.Tensor,
        target: torch.Tensor,
        reduce: bool = True,
        EM_step: bool = False,
    ) -> torch.Tensor:
        """Negative log-likelihood for Gaussian Mixture Model by Expectation-Maximization

        Args:
            pred_mean (tensor): the predicted mean.
            pred_logvar (tensor): the predicted log variance.
            pred_weight (tensor): the predicted weight.
            target (tensor): the target value.
            reduce (bool): if ``False`` the loss is returned w/o reducing.
                Defaults to ``True``.

        Returns:
            (tensor): the negative log-likelihood.
        """
        target = repeat(target, 'e b d -> e b d c',c=self.n_components)
        dist = D.Normal(pred_mean, torch.sqrt(pred_logvar.exp()))  
        if EM_step:
            # E-step 
            pdf = torch.exp(dist.log_prob(target)) * pred_weight #pdf
            p = torch.nn.Softmax(dim=-1)(pdf) #conditional probability P(c|x)
            # M-step
            p_sum = torch.sum(p,dim=1,keepdim=True)
            mean = torch.sum(p * target,dim=1,keepdim=True) / p_sum
            var = torch.sum(p * (target-mean)**2, dim=1,keepdim=True) / p_sum
            weight = p_sum / target.shape[1]
            dist = D.Normal(mean, torch.sqrt(var))
        # negative log-likelihood
        else:
            weight = pred_weight
        pdf =  torch.exp(dist.log_prob(target))
        losses = - torch.log(torch.sum(weight*pdf, dim=-1) + 1e-8)

        if reduce:
            return losses.sum(dim=1).mean()
        return losses    
    
    def _nll_loss(self, model_in: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert model_in.ndim == target.ndim
        if model_in.ndim == 2:  # add ensemble dimension
            model_in = model_in.unsqueeze(0)
            target = target.unsqueeze(0)
        pred_mean, pred_logvar, pred_weight = self.forward(model_in, use_propagation=False)
        if target.shape[0] != self.num_members:
            target = target.repeat(self.num_members, 1, 1)
        nll = (
            self.gmm_nll(pred_mean, pred_logvar, pred_weight, target, reduce=False)
            .mean((1, 2))  # average over batch and target dimension
            .sum()
        )  # sum over ensemble dimension
        nll += 0.01 * (self.max_logvar.sum() - self.min_logvar.sum())
        return nll   
     
    def eval_score(  
        self, model_in: torch.Tensor, target: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
     
        assert model_in.ndim == 2 and target.ndim == 2
        with torch.no_grad():
            means, _ , weights= self.forward(model_in, use_propagation=False)
            pred_mean = torch.sum(weights*means, dim=-1)
            target = target.repeat((self.num_members, 1, 1))
            return F.mse_loss(pred_mean, target, reduction="none"), {}
if __name__ == '__main__':
    pass

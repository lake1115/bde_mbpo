from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np

import omegaconf
import torch
from torch import nn as nn
from torch.nn import functional as F
import torch.distributions as D
from .gaussian_mlp import GaussianMLP

class GaussianMeanMLP(GaussianMLP):
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

        # x: [batch_size, in_size] 
        # -> mean, logvar: [num_elite_models, batch_size, out_size] 
        #               or [num_elite_models, N_s*batch_size, out_size]
        mean, logvar = self._default_forward(x, only_elite=True)
        mean = torch.mean(mean, dim=0)
        logvar = torch.mean(logvar, dim=0)

        # [batch_size, out_size]
        return mean, logvar

if __name__ == "__main__":
    pass

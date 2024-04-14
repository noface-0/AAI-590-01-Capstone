import torch
import torch.nn as nn
from typing import List, Optional, Union



class RegularizedLinear(nn.Linear):
    """
    Extends the nn.Linear module to include L2 or L1 regularization.
    """
    def __init__(
        self, 
        in_features, 
        out_features, 
        bias=True, 
        regularization='l2', 
        reg_strength=0.01
    ):
        super().__init__(in_features, out_features, bias)
        self.regularization = regularization
        self.reg_strength = reg_strength

    def forward(self, input):
        output = super().forward(input)
        if self.regularization == 'l2':
            self.reg_loss = 0.5 * self.reg_strength * \
                            torch.sum(torch.pow(self.weight, 2))
        elif self.regularization == 'l1':
            self.reg_loss = self.reg_strength * \
                            torch.sum(torch.abs(self.weight))
        else:
            self.reg_loss = 0
        return output
    

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.linear = nn.Linear(dim, dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        return x + self.activation(self.linear(x))
    

def build_mlp(
    dims: List[int],
    activation: Optional[Union[nn.Module, str]] = nn.ReLU,
    batch_norm: bool = False,
    dropout: Optional[float] = None,
    residual: bool = False,
    layer_norm: bool = False,
    regularization: Optional[str] = None,  # 'l1' or 'l2'
    reg_strength: float = 0.01
) -> nn.Sequential:
    net_list = []
    for i in range(len(dims) - 1):
        if regularization in ['l1', 'l2']:
            net_list.append(
                RegularizedLinear(
                    dims[i], 
                    dims[i + 1], 
                    regularization=regularization, 
                    reg_strength=reg_strength
                )
            )
        else:
            net_list.append(nn.Linear(dims[i], dims[i + 1]))
        
        if batch_norm:
            net_list.append(nn.BatchNorm1d(dims[i + 1]))
        if layer_norm:
            net_list.append(nn.LayerNorm(dims[i + 1]))
        
        if isinstance(activation, str):
            if activation.lower() == 'relu':
                net_list.append(nn.ReLU())
            elif activation.lower() == 'leakyrelu':
                net_list.append(nn.LeakyReLU())
        elif activation:
            net_list.append(activation())
        
        if dropout is not None:
            net_list.append(nn.Dropout(dropout))
        
        if residual and i > 0 and dims[i] == dims[i + 1]:
            net_list.append(ResidualBlock(dims[i]))

    return nn.Sequential(*net_list)
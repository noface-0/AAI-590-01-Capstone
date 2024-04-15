import torch
from torch import Tensor, optim, nn
from typing import List, Optional, Type

from config.base import Config

# reference: https://github.com/AI4Finance-Foundation/FinRL


class AgentBase:
    def __init__(
        self, 
        net_dims: List[int], 
        state_dim: int, 
        action_dim: int, 
        gpu_id: int = 0, 
        args: Config = Config()
    ):
        self.initialize_parameters(state_dim, action_dim, args)
        self.device = self.determine_device(gpu_id)
        self.initialize_networks(net_dims, state_dim, action_dim)
        self.initialize_optimizers(args.learning_rate)

    def initialize_parameters(
        self, 
        state_dim: int, 
        action_dim: int, 
        args: Config
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = args.gamma
        self.batch_size = args.batch_size
        self.repeat_times = args.repeat_times
        self.reward_scale = args.reward_scale
        self.soft_update_tau = args.soft_update_tau
        self.states = None

    def determine_device(self, gpu_id: int) -> torch.device:
        return torch.device(
            f"cuda:{gpu_id}" if torch.cuda.is_available() \
                and gpu_id >= 0 
            else "cpu"
        )

    def initialize_networks(
        self, 
        net_dims: List[int], 
        state_dim: int, 
        action_dim: int
    ):
        act_class = getattr(self, "act_class", None)
        cri_class = getattr(self, "cri_class", None)

        self.act = self.act_target = self.create_network(
            act_class, net_dims, state_dim, action_dim
        )
        self.cri = self.cri_target = self.create_network(
            cri_class, net_dims, state_dim, action_dim, 
            default=self.act
        )

    def create_network(
        self, 
        net_class: Optional[Type[nn.Module]], 
        net_dims: List[int], 
        state_dim: int, 
        action_dim: int, 
        default: Optional[nn.Module] = None
    ) -> nn.Module:
        if net_class is not None:
            return net_class(
                net_dims, state_dim, action_dim
            ).to(self.device)
        return default

    def initialize_optimizers(self, learning_rate: float):
        self.act_optimizer = optim.Adam(
            self.act.parameters(), learning_rate
        )
        self.cri_optimizer = optim.Adam(
            self.cri.parameters(), learning_rate
        ) if hasattr(self, "cri_class") else self.act_optimizer

        self.criterion = nn.SmoothL1Loss()

    @staticmethod
    def optimizer_update(
        optimizer: optim.Optimizer, 
        objective: Tensor
    ):
        optimizer.zero_grad()
        objective.backward()
        optimizer.step()

    @staticmethod
    def soft_update(
        target_net: nn.Module, 
        current_net: nn.Module, 
        tau: float
    ):
        for target_param, current_param in zip(
            target_net.parameters(), current_net.parameters()
        ):
            target_param.data.copy_(
                tau * current_param.data + (1.0 - tau) 
                * target_param.data
            )

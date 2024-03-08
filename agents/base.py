import torch
from torch import Tensor
from typing import List

from config.base import Config


class AgentBase:
    def __init__(
            self, 
            net_dims: List[int], 
            state_dim: int, 
            action_dim: int, 
            gpu_id: int = 0, 
            args: Config = Config()
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.gamma = args.gamma
        self.batch_size = args.batch_size
        self.repeat_times = args.repeat_times
        self.reward_scale = args.reward_scale
        self.soft_update_tau = args.soft_update_tau

        self.states = None  # assert self.states == (1, state_dim)
        self.device = torch.device(
            f"cuda:{gpu_id}" if (torch.cuda.is_available() \
                                 and (gpu_id >= 0)) else "cpu"
        )

        act_class = getattr(self, "act_class", None)
        cri_class = getattr(self, "cri_class", None)
        self.act = self.act_target = act_class(
            net_dims, state_dim, action_dim
        ).to(self.device)
        self.cri = self.cri_target = cri_class(
            net_dims, state_dim, action_dim
        ).to(self.device) if cri_class else self.act

        self.act_optimizer = torch.optim.Adam(
            self.act.parameters(), args.learning_rate
        )
        self.cri_optimizer = torch.optim.Adam(
            self.cri.parameters(), args.learning_rate
        ) if cri_class else self.act_optimizer

        self.criterion = torch.nn.SmoothL1Loss()

    @staticmethod
    def optimizer_update(optimizer, objective: Tensor):
        optimizer.zero_grad()
        objective.backward()
        optimizer.step()

    @staticmethod
    def soft_update(
        target_net: torch.nn.Module, 
        current_net: torch.nn.Module, 
        tau: float
    ):
        for tar, cur in zip(
            target_net.parameters(), current_net.parameters()
        ):
            tar.data.copy_(cur.data * tau + tar.data * (1.0 - tau))
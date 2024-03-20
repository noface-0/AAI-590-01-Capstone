import torch
import torch.nn as nn
from typing import List, Tuple
from torch import Tensor
from torch.distributions.normal import Normal
from models.mlp import build_mlp
from agents.base import AgentBase
from config.base import Config


class ActorSAC(nn.Module):
    def __init__(self, dims: List[int], state_dim: int, action_dim: int):
        super().__init__()
        self.net = build_mlp(dims=[state_dim, *dims, action_dim * 2])  # Mean and log_std

    def forward(self, state: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.net(state)
        mean, log_std = torch.chunk(x, 2, dim=-1)
        log_std = torch.clamp(log_std, min=-20, max=2)
        return mean, log_std

    def get_action(self, state: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        mean, log_std = self(state)
        std = log_std.exp()
        dist = Normal(mean, std)
        z = dist.rsample()
        action = torch.tanh(z)
        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        return action, log_prob, z

    @staticmethod
    def convert_action_for_env(action: Tensor) -> Tensor:
        return action.tanh()


class CriticSAC(nn.Module):
    def __init__(self, dims: List[int], state_dim: int, action_dim: int):
        super().__init__()
        self.net1 = build_mlp(dims=[state_dim + action_dim, *dims, 1])
        self.net2 = build_mlp(dims=[state_dim + action_dim, *dims, 1])

    def forward(self, state: Tensor, action: Tensor) -> Tuple[Tensor, Tensor]:
        sa = torch.cat([state, action], dim=-1)
        return self.net1(sa), self.net2(sa)


class AgentSAC(AgentBase):
    def __init__(
        self,
        net_dims: List[int],
        state_dim: int,
        action_dim: int,
        gpu_id: int = 0,
        args: Config = Config()
    ):
        self.if_off_policy = True
        self.act_class = getattr(self, "act_class", ActorSAC)
        self.cri_class = getattr(self, "cri_class", CriticSAC)
        AgentBase.__init__(self, net_dims, state_dim, action_dim, gpu_id, args)

        self.alpha = getattr(args, "alpha", 0.2)  # Entropy coefficient
        self.target_entropy = getattr(args, "target_entropy", -action_dim)

    def explore_env(self, env, horizon_len: int) -> List[Tensor]:
        states = torch.zeros(
            (horizon_len, self.state_dim), dtype=torch.float32
        ).to(self.device)
        actions = torch.zeros(
            (horizon_len, self.action_dim), dtype=torch.float32
        ).to(self.device)
        rewards = torch.zeros(
            horizon_len, dtype=torch.float32
        ).to(self.device)
        dones = torch.zeros(
            horizon_len, dtype=torch.bool
        ).to(self.device)

        ary_state = self.states[0]
        select_action = self.select_action

        for i in range(horizon_len):
            state = torch.as_tensor(
                ary_state, dtype=torch.float32, device=self.device
            )
            action = select_action(state.unsqueeze(0)).squeeze(0)
            ary_action = action.detach().cpu().numpy()
            ary_state, reward, done, _, info = env.step(ary_action)
            if done:
                ary_state = env.reset()[0]

            states[i] = state
            actions[i] = action
            rewards[i] = reward
            dones[i] = done

        self.states[0] = ary_state
        rewards = (rewards * self.reward_scale).unsqueeze(1)
        undones = (1 - dones.type(torch.float32)).unsqueeze(1)

        return states, actions, rewards, undones
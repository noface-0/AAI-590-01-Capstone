import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from models.mlp import build_mlp
from agents.base import AgentBase
from config.base import Config
from typing import List, Tuple
from torch import Tensor

class ActorSAC(nn.Module):
    def __init__(
            self, 
            state_dim: int, 
            action_dim: int, 
            hidden_dims: List[int]
    ):
        super().__init__()
        self.net = build_mlp(
            [state_dim, *hidden_dims, action_dim * 2]
        )  # Mean and log_std

    def forward(self, state: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.net(state)
        mean, log_std = torch.chunk(x, 2, dim=-1)
        log_std = torch.clamp(log_std, min=-20, max=2)
        return mean, log_std

    def sample_action(
            self, state: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        mean, log_std = self(state)
        std = log_std.exp()
        dist = Normal(mean, std)
        z = dist.rsample()
        action = torch.tanh(z)
        log_prob = dist.log_prob(z) - \
            torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        return action, log_prob, z


class CriticSAC(nn.Module):
    def __init__(
            self, 
            state_dim: int, 
            action_dim: int, 
            hidden_dims: List[int]
    ):
        super().__init__()
        self.net1 = build_mlp([state_dim + action_dim, *hidden_dims, 1])
        self.net2 = build_mlp([state_dim + action_dim, *hidden_dims, 1])

    def forward(
            self, state: Tensor, action: Tensor
    ) -> Tuple[Tensor, Tensor]:
        sa = torch.cat([state, action], dim=-1)
        return self.net1(sa), self.net2(sa)


class AgentSAC(AgentBase):
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int, 
        hidden_dims: List[int],
        args: Config = Config(), 
        gpu_id: int = None
    ):
        # Ensure hidden_dims is a list or tuple
        if not isinstance(hidden_dims, (list, tuple)):
            hidden_dims = [hidden_dims]
        
        self.act_class = ActorSAC
        self.cri_class = CriticSAC
        super().__init__(
            hidden_dims, state_dim, action_dim, gpu_id, args
        )
        self.actor = ActorSAC(state_dim, action_dim, hidden_dims)
        self.critic1 = CriticSAC(state_dim, action_dim, hidden_dims)
        self.critic2 = CriticSAC(state_dim, action_dim, hidden_dims)
        self.target_critic1 = CriticSAC(
            state_dim, action_dim, hidden_dims
        )
        self.target_critic2 = CriticSAC(
            state_dim, action_dim, hidden_dims
        )

        self.target_critic1.load_state_dict(
            self.critic1.state_dict()
        )
        self.target_critic2.load_state_dict(
            self.critic2.state_dict()
        )

        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=args.lr_actor
        )
        self.critic1_optimizer = optim.Adam(
            self.critic1.parameters(), lr=args.lr_critic
        )
        self.critic2_optimizer = optim.Adam(
            self.critic2.parameters(), lr=args.lr_critic
        )

        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha  # Entropy coefficient

        # If gpu_id is provided and valid, set device accordingly
        if gpu_id is not None and torch.cuda.is_available():
            self.device = torch.device(f'cuda:{gpu_id}')
        else:
            self.device = torch.device('cpu')

        # Move models to the specified device
        self.actor.to(self.device)
        self.critic1.to(self.device)
        self.critic2.to(self.device)
        self.target_critic1.to(self.device)
        self.target_critic2.to(self.device)

    def update(self, state, action, reward, next_state, done):
        with torch.no_grad():
            next_action, next_log_prob, _ = \
                self.actor.sample_action(next_state)
            target_q1_next, target_q2_next = \
                self.target_critic1(next_state, next_action), \
                self.target_critic2(next_state, next_action)
            min_target_q_next = torch.min(target_q1_next, target_q2_next) \
                                - self.alpha * next_log_prob
            target_q = reward + (1 - done) * self.gamma * min_target_q_next

        # Update Critic Networks
        current_q1, current_q2 = self.critic1(state, action), \
                                 self.critic2(state, action)
        critic1_loss = torch.nn.functional.mse_loss(current_q1, target_q)
        critic2_loss = torch.nn.functional.mse_loss(current_q2, target_q)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # Update Actor Network
        new_action, log_prob, _ = self.actor.sample_action(state)
        q1_new, q2_new = self.critic1(state, new_action), \
            self.critic2(state, new_action)
        min_q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha * log_prob - min_q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft Update Target Networks
        self.soft_update(self.target_critic1, self.critic1, self.tau)
        self.soft_update(self.target_critic2, self.critic2, self.tau)

    def select_action(self, state: Tensor) -> Tensor:
        with torch.no_grad():
            action, _, _ = self.actor.sample_action(state)
        return action

    def soft_update(self, target, source, tau):
        for target_param, param in zip(
            target.parameters(), source.parameters()
        ):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + param.data * tau
            )
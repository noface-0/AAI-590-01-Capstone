import torch
import torch.nn as nn
from typing import List, Tuple
from torch import Tensor
from torch.distributions.normal import Normal
from models.mlp import build_mlp
from agents.base import AgentBase
from config.base import Config


class ActorSAC(nn.Module):
    def __init__(
            self, 
            dims: List[int], 
            state_dim: int, 
            action_dim: int
    ):
        super().__init__()
        self.net = build_mlp(
            [state_dim, *dims, action_dim * 2], 
            activation=Config().activation, 
            batch_norm=False
        )

    def forward(self, state: Tensor) -> Tensor:
        x = self.net(state)
        mean, log_std = torch.chunk(x, 2, dim=-1)
        log_std = torch.clamp(log_std, min=-20, max=2)
        std = log_std.exp()
        dist = Normal(mean, std)
        z = dist.rsample()
        action = torch.tanh(z)
        return action
    
    def get_action(self, state: Tensor) -> Tensor:
        return self(state)

    @staticmethod
    def convert_action_for_env(action: Tensor) -> Tensor:
        return action.tanh()


class CriticSAC(nn.Module):
    def __init__(
            self, 
            dims: List[int], 
            state_dim: int, 
            action_dim: int
    ):
        super().__init__()
        self.net1 = build_mlp(
            [state_dim + action_dim, *dims, 1], 
            activation=Config().activation, 
            batch_norm=False
        )
        self.net2 = build_mlp(
            [state_dim + action_dim, *dims, 1], 
            activation=Config().activation, 
            batch_norm=False
        )

    def forward(
            self, 
            state: Tensor, 
            action: Tensor
    ) -> Tuple[Tensor, Tensor]:
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
        AgentBase.__init__(
            self, net_dims, state_dim, action_dim, gpu_id, args
        )

        self.alpha = getattr(args, "ent_coef", 0.2)  # Entropy coefficient
        self.target_entropy = getattr(args, "target_entropy", -action_dim)
        self.tau = getattr(args, "soft_update_tau", 0.005)  # Soft update factor for target networks

    def select_action(self, state: Tensor) -> Tensor:
        with torch.no_grad():
            action = self.act.get_action(state)
        return action

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

    def update_net(
            self, buffer_items: List[Tensor]
    ) -> Tuple[float, ...]:
        states, actions, rewards, undones = buffer_items
        next_state = torch.from_numpy(self.states[0])
        next_state = next_state.to(states.device)
        next_states = torch.cat(
            (states[1:], next_state.unsqueeze(0)), dim=0
        )

        q1_loss, q2_loss = self.update_critic(
            states, actions, rewards, undones, next_states
        )

        actor_loss = self.update_actor(states)

        self.soft_update(self.cri_target, self.cri, self.tau)

        return q1_loss.item(), q2_loss.item(), actor_loss.item()

    def update_critic(
            self, states, actions, rewards, undones, next_states
    ):
        # Compute target Q-values
        with torch.no_grad():
            next_actions = self.act.get_action(next_states)
            next_q1, next_q2 = self.cri_target(next_states, next_actions)
            next_q = torch.min(next_q1, next_q2)
            target_q = rewards + undones * self.gamma * next_q

        # Compute current Q-values
        current_q1, current_q2 = self.cri(states, actions)
        # critic loss
        q1_loss = nn.MSELoss()(current_q1, target_q)
        q2_loss = nn.MSELoss()(current_q2, target_q)

        self.cri_optimizer.zero_grad()
        (q1_loss + q2_loss).backward()
        self.cri_optimizer.step()

        return q1_loss, q2_loss

    def update_actor(self, states):
        # actor loss
        actions = self.act.get_action(states)
        q1, q2 = self.cri(states, actions)
        q = torch.min(q1, q2)
        actor_loss = -q.mean()

        self.act_optimizer.zero_grad()
        actor_loss.backward()
        self.act_optimizer.step()

        return actor_loss

    def soft_update(self, target_net, current_net, tau):
        for target_param, current_param in zip(
            target_net.parameters(), current_net.parameters()
        ):
            target_param.data.copy_(
                tau * current_param.data + (1 - tau) * target_param.data
            )
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from typing import List
from copy import deepcopy
from torch.optim import Adam

from config.base import Config
from models.mlp import build_mlp
from agents.base import AgentBase


class ActorTD3(nn.Module):
    def __init__(self, state_dim, action_dim, net_dims):
        super().__init__()
        self.net = build_mlp(
            [state_dim, *net_dims, action_dim], 
            activation=Config().activation
        )
        
    def forward(self, state):
        return self.net(state).tanh()

class CriticTD3(nn.Module):
    def __init__(self, state_dim, action_dim, net_dims):
        super().__init__()
        self.net1 = build_mlp(
            [state_dim + action_dim, *net_dims, 1], 
            activation=Config().activation
        )
        self.net2 = build_mlp(
            [state_dim + action_dim, *net_dims, 1], 
            activation=Config().activation
        )
        
    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        return self.net1(sa), self.net2(sa)


class AgentTD3(AgentBase):
    def __init__(
        self,
        net_dims: List[int], 
        state_dim: int, 
        action_dim: int, 
        gpu_id: int = 0, 
        args: Config = Config()
    ):
        print(net_dims)
        self.if_off_policy = True
        self.act_class = getattr(self, "act_class", ActorTD3)
        self.cri_class = getattr(self, "cri_class", CriticTD3)
        AgentBase.__init__(self, net_dims, state_dim, action_dim, gpu_id, args)

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.net_dims = net_dims
        self.device= torch.device(f'cuda:{gpu_id}' if gpu_id >= 0 else 'cpu')
        
        self.gamma = args.gamma
        self.tau = args.soft_update_tau
        self.policy_noise = 0.2
        self.noise_clip = 0.5
        self.exploration_noise = 0.1
        self.update_actor_interval = 2  # Update actor every 2 steps as an example
        self.reward_scale = args.reward_scale
        
        self.actor = ActorTD3(self.state_dim, self.action_dim, self.net_dims).to(self.device)
        self.actor_target = deepcopy(self.actor)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=args.learning_rate)
        
        self.critic = CriticTD3(self.state_dim, self.action_dim, self.net_dims).to(self.device)
        self.critic_target = deepcopy(self.critic)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=args.learning_rate)

        self.total_it = 0

    def explore_env(self, env, horizon_len):
        states, actions, rewards, dones = [], [], [], []
        state = env.reset()

        for _ in range(horizon_len):
            action = self.select_action(state)
            next_state, reward, done, _ = env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)

            state = next_state if not done else env.reset()

        return np.array(states), np.array(actions), np.array(rewards), np.array(dones)

    def update_net(self, replay_buffer):
        obj_critics = 0.0
        obj_actors = 0.0

        for _ in range(int(replay_buffer.size / self.batch_size)):
            self.total_it += 1
            state, action, next_state, reward, done = replay_buffer.sample(self.batch_size)

            with torch.no_grad():
                # Adding noise to action for the target policy smoothing
                noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
                next_action = (self.actor_target(next_state) + noise).clamp(-1, 1)

                # Compute the target Q value
                target_Q1, target_Q2 = self.critic_target(next_state, next_action)
                target_Q = torch.min(target_Q1, target_Q2)
                target_Q = reward + (1 - done) * self.gamma * target_Q

            # Get current Q estimates
            current_Q1, current_Q2 = self.critic(state, action)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Delayed policy updates
            if self.total_it % self.update_actor_interval == 0:
                # Compute actor loss
                actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Soft update the target networks
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            obj_critics += critic_loss.item()
            obj_actors += actor_loss.item() if self.total_it % self.update_actor_interval == 0 else 0

        return obj_critics / self.total_it, obj_actors / (self.total_it // self.update_actor_interval)

    def select_action(self, state, noise=True):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state).cpu().data.numpy().flatten()
        self.actor.train()
        if noise:
            action += np.random.normal(0, self.exploration_noise, size=action.shape)
        return np.clip(action, -1, 1)


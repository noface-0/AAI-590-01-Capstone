import torch
import torch.nn as nn
from typing import List, Tuple
from torch import Tensor
from torch.distributions.normal import Normal

from models.mlp import build_mlp
from agents.base import AgentBase
from config.base import Config


class ActorPPO(nn.Module):
    def __init__(self, dims: List[int], state_dim: int, action_dim: int):
        super().__init__()
        self.net = build_mlp(dims=[state_dim, *dims, action_dim])
        self.action_std_log = nn.Parameter(
            torch.zeros((1, action_dim)), requires_grad=True
        )  # trainable parameter

    def forward(self, state: Tensor) -> Tensor:
        return self.net(state).tanh()  # action.tanh()

    def get_action(self, state: Tensor) -> Tuple[Tensor, Tensor]:  # for exploration
        action_avg = self.net(state)
        action_std = self.action_std_log.exp()

        dist = Normal(action_avg, action_std)
        action = dist.sample()
        logprob = dist.log_prob(action).sum(1)
        return action, logprob

    def get_logprob_entropy(
            self, state: Tensor, action: Tensor
    ) -> Tuple[Tensor, Tensor]:
        action_avg = self.net(state)
        action_std = self.action_std_log.exp()

        dist = Normal(action_avg, action_std)
        logprob = dist.log_prob(action).sum(1)
        entropy = dist.entropy().sum(1)
        return logprob, entropy

    @staticmethod
    def convert_action_for_env(action: Tensor) -> Tensor:
        return action.tanh()


class CriticPPO(nn.Module):
    def __init__(self, dims: List[int], state_dim: int, _action_dim: int):
        super().__init__()
        self.net = build_mlp(dims=[state_dim, *dims, 1])

    def forward(self, state: Tensor) -> Tensor:
        return self.net(state)  # advantage value
    

class AgentPPO(AgentBase):
    def __init__(
            self, 
            net_dims: List[int], 
            state_dim: int, 
            action_dim: int, 
            gpu_id: int = 0, 
            args: Config = Config()
    ):
        self.if_off_policy = False
        self.act_class = getattr(self, "act_class", ActorPPO)
        self.cri_class = getattr(self, "cri_class", CriticPPO)
        AgentBase.__init__(self, net_dims, state_dim, action_dim, gpu_id, args)

        self.ratio_clip = getattr(args, "ratio_clip", 0.25)  # `ratio.clamp(1 - clip, 1 + clip)`
        self.lambda_gae_adv = getattr(args, "lambda_gae_adv", 0.95)  # could be 0.80~0.99
        self.lambda_entropy = getattr(args, "lambda_entropy", 0.01)  # could be 0.00~0.10
        self.lambda_entropy = torch.tensor(
            self.lambda_entropy, dtype=torch.float32, device=self.device
        )

    def explore_env(self, env, horizon_len: int) -> List[Tensor]:
        states = torch.zeros(
            (horizon_len, self.state_dim), dtype=torch.float32
        ).to(self.device)
        actions = torch.zeros(
            (horizon_len, self.action_dim), dtype=torch.float32
        ).to(self.device)
        logprobs = torch.zeros(
            horizon_len, dtype=torch.float32
        ).to(self.device)
        rewards = torch.zeros(
            horizon_len, dtype=torch.float32
        ).to(self.device)
        dones = torch.zeros(
            horizon_len, dtype=torch.bool
        ).to(self.device)

        ary_state = self.states[0]

        get_action = self.act.get_action
        convert = self.act.convert_action_for_env
        for i in range(horizon_len):
            state = torch.as_tensor(
                ary_state, dtype=torch.float32, device=self.device
            )
            action, logprob = [
                t.squeeze(0) for t in get_action(state.unsqueeze(0))[:2]
            ]

            ary_action = convert(action).detach().cpu().numpy()
            ary_state, reward, done, _, info = env.step(ary_action)
            if done:
                ary_state = env.reset()[0]

            states[i] = state
            actions[i] = action
            logprobs[i] = logprob
            rewards[i] = reward
            dones[i] = done

        self.states[0] = ary_state
        rewards = (rewards * self.reward_scale).unsqueeze(1)
        undones = (1 - dones.type(torch.float32)).unsqueeze(1)
        return states, actions, logprobs, rewards, undones

    def update_net(self, buffer) -> List[float]:
        with torch.no_grad():
            states, actions, logprobs, rewards, undones = buffer
            buffer_size = states.shape[0]

            '''get advantages reward_sums'''
            bs = 2 ** 10  # set a smaller 'batch_size' when out of GPU memory.
            values = [
                self.cri(states[i:i + bs]) for i in range(0, buffer_size, bs)
            ]
            values = torch.cat(values, dim=0).squeeze(1)  # values.shape == (buffer_size, )

            advantages = self.get_advantages(rewards, undones, values)  # advantages.shape == (buffer_size, )
            reward_sums = advantages + values  # reward_sums.shape == (buffer_size, )
            del rewards, undones, values

            advantages = (
                advantages - advantages.mean()) / (advantages.std(dim=0) + 1e-5
            )
        assert logprobs.shape == advantages.shape \
            == reward_sums.shape == (buffer_size,)

        '''update network'''
        obj_critics = 0.0
        obj_actors = 0.0

        update_times = int(buffer_size * self.repeat_times / self.batch_size)
        assert update_times >= 1
        for _ in range(update_times):
            indices = torch.randint(
                buffer_size, size=(self.batch_size,), requires_grad=False
            )
            state = states[indices]
            action = actions[indices]
            logprob = logprobs[indices]
            advantage = advantages[indices]
            reward_sum = reward_sums[indices]

            value = self.cri(state).squeeze(1)  # critic network predicts the reward_sum (Q value) of state
            obj_critic = self.criterion(value, reward_sum)
            self.optimizer_update(self.cri_optimizer, obj_critic)

            new_logprob, obj_entropy = self.act.get_logprob_entropy(state, action)
            ratio = (new_logprob - logprob.detach()).exp()
            surrogate1 = advantage * ratio
            surrogate2 = advantage * ratio.clamp(
                1 - self.ratio_clip, 1 + self.ratio_clip
            )
            obj_surrogate = torch.min(surrogate1, surrogate2).mean()

            obj_actor = obj_surrogate + obj_entropy.mean() * self.lambda_entropy
            self.optimizer_update(self.act_optimizer, -obj_actor)

            obj_critics += obj_critic.item()
            obj_actors += obj_actor.item()
        a_std_log = getattr(self.act, 'a_std_log', torch.zeros(1)).mean()
        return (
            obj_critics / update_times, obj_actors / 
            update_times, a_std_log.item()
        )

    def get_advantages(
            self, 
            rewards: Tensor, 
            undones: Tensor, 
            values: Tensor
    ) -> Tensor:
        advantages = torch.empty_like(values)  # advantage value

        masks = undones * self.gamma
        horizon_len = rewards.shape[0]

        next_state = torch.tensor(
            self.states, dtype=torch.float32
        ).to(self.device)
        next_value = self.cri(next_state).detach()[0, 0]

        advantage = 0  # last_gae_lambda
        for t in range(horizon_len - 1, -1, -1):
            delta = rewards[t] + masks[t] * next_value - values[t]
            advantages[t] = advantage = delta + masks[t] \
                * self.lambda_gae_adv * advantage
            next_value = values[t]
        return advantages
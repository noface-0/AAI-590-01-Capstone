from __future__ import annotations
import os
import json
import torch
import numpy as np

from agents.ppo import AgentPPO
from agents.sac import AgentSAC
from agents.td3 import AgentTD3
from config.base import Config
from utils.training_utils import train_agent


MODELS = {"ppo": AgentPPO, "sac": AgentSAC, "td3": AgentTD3}
OFF_POLICY_MODELS = ["ddpg", "td3", "sac"]
ON_POLICY_MODELS = ["ppo"]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


# reference: https://github.com/AI4Finance-Foundation/FinRL


class DRLAgent:
    """Implementations of DRL algorithms
    Attributes
    ----------
        env: gym environment class
            user-defined class
    Methods
    -------
        get_model()
            setup DRL algorithms
        train_model()
            train DRL algorithms in a train dataset
            and output the trained model
        DRL_prediction()
            make a prediction in a test dataset and get results
    """

    def __init__(
            self, 
            env, 
            price_array, 
            tech_array, 
            turbulence_array,
            objective,
            agent
    ):
        self.env = env
        self.price_array = price_array
        self.tech_array = tech_array
        self.turbulence_array = turbulence_array
        self.objective = objective
        self.agent = agent

    def get_model(self, model_name, model_kwargs):
        env_config = {
            "price_array": self.price_array,
            "tech_array": self.tech_array,
            "turbulence_array": self.turbulence_array,
            "if_train": True,
            "objective": self.objective,
            "agent": self.agent
        }
        environment = self.env(config=env_config)

        env_args = {
            'config': env_config,
            'env_name': environment.env_name,
            'state_dim': environment.state_dim,
            'action_dim': environment.action_dim,
            'if_discrete': False
        }
        
        agent = MODELS[model_name]

        if model_name not in MODELS:
            raise NotImplementedError("NotImplementedError")
        
        model = Config(
            agent_class=agent, env_class=self.env, env_args=env_args
        )
        model.if_off_policy = model_name in OFF_POLICY_MODELS

        for key in ('seed', 'target_step', 'eval_gap'):
            if key in model_kwargs:
                setattr(model, key, model_kwargs[key])

        return model

    def train_model(self, model, cwd, total_timesteps=5000):
        model.cwd = cwd
        model.break_step = total_timesteps
        train_agent(model)

    @staticmethod
    def DRL_prediction(model_name, cwd, net_dimension, environment):
        if model_name not in MODELS:
            raise NotImplementedError(
                f"Model '{model_name}' not implemented."
            )

        agent_class = MODELS[model_name]
        environment.env_num = 1
        agent = agent_class(
            net_dimension, environment.state_dim, 
            environment.action_dim
        )
        actor = agent.act

        model_path = os.path.join(cwd, 'actor.pth')
        print(f"| load actor from: {model_path}")
        actor.load_state_dict(
            torch.load(model_path, map_location='cpu')
        )

        state, done = environment.reset()[0], False
        episode_returns, episode_total_assets = [], [
            environment.initial_total_asset
        ]

        with torch.no_grad():
            for _ in range(environment.max_step):
                if done:
                    break
                action = agent.act(
                    torch.tensor([state], device=agent.device)
                ).cpu().numpy()[0]
                state, _, done, _, _ = environment.step(action)
                current_asset = environment.calculate_total_asset()
                episode_total_assets.append(current_asset)
                episode_returns.append(
                    current_asset / environment.initial_total_asset
                )

        environment.save_portfolio_value_log()
        environment.save_trade_log()

        min_asset, max_asset = min(episode_total_assets), episode_total_assets[0]
        drawdown = (max_asset - min_asset) / max_asset
        returns_np = np.array(episode_returns)
        mean_return = returns_np.mean()
        return_std = returns_np.std()
        sharpe_ratio = (mean_return / return_std) if return_std else 0

        return episode_total_assets, episode_returns[-1], drawdown, sharpe_ratio
import gym
import numpy as np
from typing import Tuple

# reference: https://github.com/AI4Finance-Foundation/FinRL

class PendulumEnv(gym.Wrapper):
    def __init__(self):
        gym.logger.set_level(40)  # Block warning
        gym_env_name = "Pendulum-v0" if \
            gym.__version__ < '0.18.0' else "Pendulum-v1"
        super().__init__(env=gym.make(gym_env_name))

        '''the necessary env information when you design a custom env'''
        self.env_name = gym_env_name  # the name of this env.
        self.state_dim = self.observation_space.shape[0]  # feature number of state
        self.action_dim = self.action_space.shape[0]  # feature number of action
        self.if_discrete = False  # discrete action or continuous action

    def reset(self) -> np.ndarray:  # reset the agent in env
        return self.env.reset()

    def step(
            self, 
            action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, dict]:  # agent interacts in env
        # Suggest adjusting action space to (-1, +1) when designing a custom env.
        state, reward, done, info_dict = self.env.step(action * 2)
        return state.reshape(self.state_dim), float(reward), done, info_dict
import os
import numpy as np
import torch.nn as nn


# reference: https://github.com/AI4Finance-Foundation/FinRL

class Config:
    def __init__(self, agent_class=None, env_class=None, env_args=None):
        self.env_class = env_class  # env = env_class(**env_args)
        self.env_args = env_args  # env = env_class(**env_args)

        if env_args is None:  # dummy env_args
            env_args = {
                'env_name': None, 
                'state_dim': None, 
                'action_dim': None, 
                'if_discrete': None
            }
        self.env_name = env_args['env_name']  # the name of environment. Be used to set 'cwd'.
        self.state_dim = env_args['state_dim']  # vector dimension (feature number) of state
        self.action_dim = env_args['action_dim']  # vector dimension (feature number) of action
        self.if_discrete = env_args['if_discrete']  # discrete or continuous action space

        self.agent_class = agent_class  # agent = agent_class(...)

        '''Arguments for reward shaping'''
        self.gamma = 0.999  # discount factor of future rewards
        self.reward_scale = 2**-11 # adjust this based on reward signal. higher for realized gains obj
        self.reward_obj = 'portfolio_value' # see options below

        '''Arguments for training'''
        self.gpu_id = int(0)  # `int` means the ID of single GPU, -1 means CPU
        self.net_dims = [512, 512, 512]  # the middle layer dimension of MLP (MultiLayer Perceptron)
        self.learning_rate = 0.0003  # 2 ** -14 ~= 6e-5
        self.activation = nn.ReLU
        self.soft_update_tau = 5e-3  # 2 ** -8 ~= 5e-3
        self.batch_size = int(128)  # num of transitions sampled from replay buffer.
        self.horizon_len = int(4000)  # collect horizon_len step while exploring, then update network
        self.buffer_size = None  # ReplayBuffer size. Empty the ReplayBuffer for on-policy.
        self.repeat_times = 6.0  # repeatedly update network using ReplayBuffer to keep critic's loss small

        '''Arguments for evaluate'''
        self.cwd = None  # current working directory to save model. None means set automatically
        self.break_step = +np.inf  # break training if 'total_step > break_step'
        self.eval_times = int(32)  # number of times that get episodic cumulative return
        self.eval_per_step = int(2e4)  # evaluate the agent per training steps
    
    def to_dict(self):
        return self.__dict__

    def init_before_training(self):
        if self.cwd is None:  # set cwd (current working directory) for saving model
            self.cwd = f'./{self.env_name}_{self.agent_class.__name__[5:]}'
        os.makedirs(self.cwd, exist_ok=True)


# REWARD OBJECTIVE OPTIONS
# realized_gains - encourages selling for realization
# portfolio_value - main objective is to increase value / may hold a lot
# sharpe_ratio - maximizes risk adjusted return

# HELPFUL VALUES

# A2C_PARAMS
# "n_steps": 5, 
# "ent_coef": 0.01, 
# "learning_rate": 0.0007


# PPO_PARAMS
# "n_steps": 2048,
# "ent_coef": 0.01,
# "learning_rate": 0.00025,
# "batch_size": 64

# DDPG_PARAMS
# "batch_size": 128, 
# "buffer_size": 50000, 
# "learning_rate": 0.001

# TD3_PARAMS
# "batch_size": 100, 
# "buffer_size": 1000000, 
# "learning_rate": 0.001

# SAC_PARAMS
# "batch_size": 256,
# "buffer_size": 100000,
# "learning_rate": 0.0003,
# "learning_starts": 100,
# "ent_coef": "auto_0.1",
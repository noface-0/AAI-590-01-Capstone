from __future__ import annotations

import os
import gymnasium as gym
import numpy as np
from numpy import random as rd

from config.base import Config

# reference: https://github.com/AI4Finance-Foundation/FinRL

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.realpath(__file__))
)

class StockTradingEnv(gym.Env):
    def __init__(
        self,
        config,
        initial_account=1e7,
        gamma=0.99,
        turbulence_thresh=99,
        min_stock_rate=0.1,
        max_stock=3e2,
        initial_capital=1e6,
        buy_cost_pct=1e-3,
        sell_cost_pct=1e-3,
        reward_scaling=Config().reward_scale,
        reward_objective=Config().reward_obj,
        # reward_scaling=1,
        initial_stocks=None,
    ):
        price_ary = config["price_array"]
        tech_ary = config["tech_array"]
        turbulence_ary = config["turbulence_array"]
        if_train = config["if_train"]

        self.objective = config["objective"]
        self.agent = config['agent']
        self.if_train = if_train

        self.price_ary = price_ary.astype(np.float32)
        self.tech_ary = tech_ary.astype(np.float32)
        self.turbulence_ary = turbulence_ary

        self.tech_ary = self.tech_ary * 2**-7
        # self.tech_ary = (self.tech_ary - self.tech_ary.min()) / self.tech_ary.ptp()

        self.turbulence_bool = (
            turbulence_ary > turbulence_thresh
        ).astype(np.float32)
        self.turbulence_ary = (
            self.sigmoid_sign(turbulence_ary, turbulence_thresh) * 2**-5
        ).astype(np.float32)

        stock_dim = self.price_ary.shape[1]
        self.gamma = gamma
        self.max_stock = max_stock
        self.min_stock_rate = min_stock_rate
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.initial_capital = initial_capital
        self.initial_stocks = (
            np.zeros(stock_dim, dtype=np.float32)
            if initial_stocks is None
            else initial_stocks
        )

        # reset()
        self.day = None
        self.amount = None
        self.stocks = None
        self.total_asset = None
        self.gamma_reward = None
        self.initial_total_asset = None

        # environment information
        self.env_name = "StockEnv"
        # self.state_dim = 1 + 2 + 2 * stock_dim + self.tech_ary.shape[1]
        # # amount + (turbulence, turbulence_bool) + (price, stock) * stock_dim + tech_dim
        self.state_dim = 1 + 2 + 3 * stock_dim + self.tech_ary.shape[1]
        # amount + (turbulence, turbulence_bool) + (price, stock) * stock_dim + tech_dim
        self.stocks_cd = None
        self.action_dim = stock_dim
        self.max_step = self.price_ary.shape[0] - 1
        self.if_train = if_train
        self.if_discrete = False
        self.target_return = 10.0
        self.episode_return = 0.0

        self.observation_space = gym.spaces.Box(
            low=-3000, high=3000, shape=(self.state_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(self.action_dim,), dtype=np.float32
        )

        self.trade_log = []
        self.portfolio_value_log = []
        self.returns = []
        self.risk_free_rate = 0.0
        self.realized_gains = 0.0
        self.cost_basis = None

        # reward shaping
        self.reward_objective = reward_objective
        self.reward_scale_base = reward_scaling 
        self.reward_scale = self.reward_scale_base
        self.performance_window = 100
        self.performance_threshold = 0.1
        self.episode_rewards = []
        

    def reset(
        self,
        *,
        seed=None,
        options=None,
    ):
        self.day = 0
        price = self.price_ary[self.day]

        self.cost_basis = price.copy()

        if self.if_train:
            self.stocks = (
                self.initial_stocks \
                    + rd.randint(0, 64, size=self.initial_stocks.shape)
            ).astype(np.float32)
            self.stocks_cool_down = np.zeros_like(self.stocks)
            self.amount = (
                self.initial_capital * rd.uniform(0.95, 1.05)
                - (self.stocks * price).sum()
            )
        else:
            self.stocks = self.initial_stocks.astype(np.float32)
            self.stocks_cool_down = np.zeros_like(self.stocks)
            self.amount = self.initial_capital

        self.total_asset = self.amount + (self.stocks * price).sum()
        self.initial_total_asset = self.total_asset
        self.gamma_reward = 0.0
        self.returns = []
        return self.get_state(price), {}  # state
    
    def save_trade_log(self):
        filename = (
            f'{BASE_DIR}/models/runs/drl/{self.objective}/'
            f'{self.agent}/trade_log.txt'
        )

        with open(filename, "w") as file:
            file.write("action,num_shares,price,day\n")
            for trade in self.trade_log:
                file.write(
                    f"{trade[0]},{trade[1]},{trade[2]},{trade[3]}\n"
                )

        # reset the trade log for the next episode
        self.trade_log = []
        self.portfolio_value_log = []

    def save_portfolio_value_log(self):
        filename = (
            f'{BASE_DIR}/models/runs/drl/{self.objective}/'
            f'{self.agent}/portfolio_value_log.txt'
        )
        with open(filename, "w") as file:
            file.write("day,total_asset\n")
            for entry in self.portfolio_value_log:
                file.write(f"{entry[0]},{entry[1]}\n")

        self.portfolio_value_log = []

    def adjust_reward_scaling(self):
        # turned off
        return self.reward_scale_base
    
        if len(self.episode_rewards) >= self.performance_window:
            recent_performance = np.mean(
                self.episode_rewards[-self.performance_window:]
            )
            adjustment_factor_up = (
                self.reward_scale_base * 1.05 - self.reward_scale_base
            )
            adjustment_factor_down = (
                self.reward_scale_base * 0.95 - self.reward_scale_base
            )
            
            if recent_performance > self.performance_threshold:
                self.reward_scale += adjustment_factor_up
            else:
                self.reward_scale += adjustment_factor_down
            
            self.episode_rewards = []

    def step(self, actions):
        actions = (actions * self.max_stock).astype(int)

        self.day += 1
        price = self.price_ary[self.day]
        self.stocks_cool_down += 1

        self.adjust_reward_scaling()

        if self.turbulence_bool[self.day] == 0:
            min_action = int(self.max_stock * self.min_stock_rate)  # stock_cd

            for index in np.where(actions < -min_action)[0]:  # sell_index:
                if price[index] > 0:  # Sell only if current asset is > 0
                    sell_num_shares = min(self.stocks[index], -actions[index])
                    if sell_num_shares > 0:
                        self.trade_log.append(
                            ["sell", -sell_num_shares, price[index], self.day]
                        )

                        realized_gain = (
                            price[index] - self.cost_basis[index]
                        ) * sell_num_shares
                        self.realized_gains += realized_gain

                    self.stocks[index] -= sell_num_shares
                    self.amount += (
                        price[index] * sell_num_shares * (1 - self.sell_cost_pct)
                    )
                    self.stocks_cool_down[index] = 0
            for index in np.where(actions > min_action)[0]:  # buy_index:
                if price[index] > 0:
                    buy_num_shares = min(self.amount // price[index], actions[index])
                    if buy_num_shares > 0:
                        self.trade_log.append(
                            ["buy", buy_num_shares, price[index], self.day]
                        )

                        # Update cost_basis using weighted average
                        total_shares = self.stocks[index] + buy_num_shares
                        if total_shares > 0 and \
                            np.isfinite(self.cost_basis[index]) \
                                and np.isfinite(price[index]):
                            self.cost_basis[index] = (
                                (self.cost_basis[index] * self.stocks[index]) 
                                + (price[index] * buy_num_shares)
                            ) / total_shares
                        else:
                            self.cost_basis[index] = price[index]

                    self.stocks[index] += buy_num_shares
                    self.amount -= price[index] * buy_num_shares \
                        * (1 + self.buy_cost_pct)
                    self.stocks_cool_down[index] = 0
            for index in range(len(actions)):
                if -min_action <= actions[index] <= min_action:
                    self.trade_log.append(["hold", 0, price[index], self.day])

        else:  # sell all when turbulence
            for index in range(len(self.stocks)):
                if self.stocks[index] > 0:
                    sell_num_shares = self.stocks[index]
                    self.trade_log.append(
                        ["sell", -sell_num_shares, price[index], self.day]
                    )

            self.amount += (
                self.stocks * price).sum() * (1 - self.sell_cost_pct)
            self.stocks[:] = 0
            self.stocks_cool_down[:] = 0
            

        state = self.get_state(price)
        total_asset = self.amount + (self.stocks * price).sum()
        asset_change = total_asset - self.total_asset

        # calculate unrealized gains
        unrealized_gains = (price - self.cost_basis) * self.stocks
        unrealized_gains = unrealized_gains.sum()

        # maximizing return
        if self.reward_objective == 'portfolio_value':
            reward = asset_change * self.reward_scale
        elif self.reward_objective == 'realized_gains':
            reward = (
                self.realized_gains + unrealized_gains
            ) * self.reward_scale
            self.realized_gains = 0
        elif self.reward_objective == 'sharpe_ratio':
            # maximizing sharpe ratio
            reward = self.calculate_sharpe_ratio(asset_change)
        else:
            raise ValueError("REWARD should be in ['portfolio_value', "
                             "'sharpe_ratio', 'realized_gains']")

        self.total_asset = total_asset
        self.gamma_reward = self.gamma_reward * self.gamma + reward
        done = self.day == self.max_step
        if done:
            reward = self.gamma_reward
            self.episode_return = total_asset / self.initial_total_asset

        return state, reward, done, False, dict()

    def get_state(self, price):
        amount = np.array(self.amount * (2**-12), dtype=np.float32)
        scale = np.array(2**-6, dtype=np.float32)
        return np.hstack(
            (
                amount,
                self.turbulence_ary[self.day],
                self.turbulence_bool[self.day],
                price * scale,
                self.stocks * scale,
                self.stocks_cool_down,
                self.tech_ary[self.day],
            )
        )  # state.astype(np.float32)

    @staticmethod
    def sigmoid_sign(ary, thresh):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x * np.e)) - 0.5

        return sigmoid(ary / thresh) * thresh
    
    def calculate_sharpe_ratio(self, asset_change):
        current_return = asset_change / self.total_asset
        self.returns.append(current_return)
        if len(self.returns) > 1:
            excess_returns = np.array(self.returns) - self.risk_free_rate
            sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns, ddof=1)
        else:
            sharpe_ratio = 0
        return sharpe_ratio * self.reward_scale
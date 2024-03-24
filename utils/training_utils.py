import os
import torch
import time
import numpy as np
from typing import Tuple, List

from config.base import Config
from utils.env_utils import build_env

# reference: https://github.com/AI4Finance-Foundation/FinRL


def get_rewards_and_steps(env, actor, if_render=False):
    if not hasattr(actor, 'get_logprob_entropy'):  # SAC agent
        device = next(actor.parameters()).device  # net.parameters() is a Python generator.
        state = env.reset()
        cumulative_returns = 0
        steps = 0

        while True:
            if isinstance(state, tuple):
                state = state[0]
            if len(state.shape) == 1:
                state = np.expand_dims(state, axis=0)

            tensor_state = torch.as_tensor(
                state, dtype=torch.float32, device=device
            )
            with torch.no_grad():
                tensor_action = actor.get_action(tensor_state)

            if tensor_action.ndim == 0:
                action = tensor_action.item()
            else:
                action = tensor_action.detach().cpu().numpy()

            state, reward, done, _, info = env.step(action)
            cumulative_returns += reward

            if if_render:
                env.render()

            steps += 1

            if done:
                break

        return cumulative_returns, steps

    else: # PPO agent
        device = next(actor.parameters()).device

        state = env.reset()[0]
        episode_steps = 0
        cumulative_returns = 0.0
        for episode_steps in range(12345):
            tensor_state = torch.as_tensor(
                state, dtype=torch.float32, device=device
            ).unsqueeze(0)
            tensor_action = actor(tensor_state)
            action = tensor_action.detach().cpu().numpy()[0]
            state, reward, done, _, info = env.step(action)
            cumulative_returns += reward

            if if_render:
                env.render()
            if done:
                break
        return cumulative_returns, episode_steps + 1


def train_agent(args: Config):
    args.init_before_training()

    env = build_env(args.env_class, args.env_args)
    agent = args.agent_class(
        args.net_dims, 
        args.state_dim, 
        args.action_dim, 
        gpu_id=args.gpu_id, 
        args=args
    )
    agent.states = env.reset()[0][np.newaxis, :]

    evaluator = Evaluator(eval_env=build_env(args.env_class, args.env_args),
                          eval_per_step=args.eval_per_step,
                          eval_times=args.eval_times,
                          cwd=args.cwd)
    torch.set_grad_enabled(False)
    while True: # start training
        buffer_items = agent.explore_env(env, args.horizon_len)

        torch.set_grad_enabled(True)
        logging_tuple = agent.update_net(buffer_items)
        torch.set_grad_enabled(False)

        evaluator.evaluate_and_save(
            agent.act, args.horizon_len, logging_tuple
        )
        if (evaluator.total_step > args.break_step) \
            or os.path.exists(f"{args.cwd}/stop"):
            torch.save(agent.act.state_dict(), args.cwd + '/actor.pth')
            break  # stop training when reach `break_step` or `mkdir cwd/stop`


def render_agent(
        env_class, 
        env_args: dict, 
        net_dims: List[int], 
        agent_class, 
        actor_path: str, 
        render_times: int = 8
):
    env = build_env(env_class, env_args)

    state_dim = env_args['state_dim']
    action_dim = env_args['action_dim']
    agent = agent_class(net_dims, state_dim, action_dim, gpu_id=-1)
    actor = agent.act

    print(f"| render and load actor from: {actor_path}")
    actor.load_state_dict(
        torch.load(actor_path, map_location=lambda storage, loc: storage)
    )
    for i in range(render_times):
        cumulative_reward, episode_step = get_rewards_and_steps(
            env, actor, if_render=True
        )
        print(
            f"{i:4}  cumulative_reward {cumulative_reward:9.3f}  "
            f"episode_step {episode_step:5.0f}"
        )


class Evaluator:
    def __init__(
            self, 
            eval_env, 
            eval_per_step: int = 1e4, 
            eval_times: int = 8, 
            cwd: str = '.'
    ):
        self.cwd = cwd
        self.env_eval = eval_env
        self.eval_step = 0
        self.total_step = 0
        self.start_time = time.time()
        self.eval_times = eval_times  # number of times that get episodic cumulative return
        self.eval_per_step = eval_per_step  # evaluate the agent per training steps

        self.recorder = []
        self.returns_history = []
        self.drawdown_history = []
        print(
            "\n| `step`: Number of samples, total training steps,"
            " or `env.step()` runs."
            "\n| `time`: Time from start of training to now."
            "\n| `avgR`: Avg. cumulative rewards per episode."
            "\n| `stdR`: Std dev of cumulative rewards."
            "\n| `avgS`: Avg. steps per episode."
            "\n| `objC`: Critic network loss function."
            "\n| `objA`: Actor network avg Q value."
            "\n| `returns`: Avg. returns per episode."
            "\n| `drawdown`: Max drawdown per episode."
            f"\n| {'step':>8} | {'time':>8} | {'avgR':>8} |"
            f" {'stdR':>8} | {'avgS':>8} | {'objC':>8} |"
            f" {'objA':>8} | {'returns':>8} | {'drawdown':>8}"
        )
            
    def evaluate_and_save(
            self, 
            actor, 
            horizon_len: int, 
            logging_tuple: tuple
    ):
        self.total_step += horizon_len
        if self.eval_step + self.eval_per_step > self.total_step:
            return
        self.eval_step = self.total_step

        rewards_steps_ary = [
            get_rewards_and_steps(self.env_eval, actor) \
                for _ in range(self.eval_times)
        ]
        rewards_steps_ary = np.array(rewards_steps_ary, dtype=np.float32)
        avg_r = rewards_steps_ary[:, 0].mean()  # average of cumulative rewards
        std_r = rewards_steps_ary[:, 0].std()  # std of cumulative rewards
        avg_s = rewards_steps_ary[:, 1].mean()  # average of steps in an episode

        episode_returns = rewards_steps_ary[:, 0]
        returns = episode_returns / self.env_eval.initial_total_asset

        cumulative_returns = np.cumsum(episode_returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = np.zeros_like(cumulative_returns)
        drawdown[1:] = ((running_max[:-1] - cumulative_returns[1:]) / 
                        np.maximum(running_max[:-1], 1e-8))
        drawdown = np.maximum.accumulate(drawdown)

        # Record returns and drawdown history
        self.returns_history.append(returns.mean())
        self.drawdown_history.append(drawdown[-1])

        used_time = time.time() - self.start_time
        self.recorder.append((self.total_step, used_time, avg_r))

        print(f"| {self.total_step:8.2e} | {used_time:8.0f} | "
              f"{avg_r:8.4f} | {std_r:8.4f} | {avg_s:8.0f} | "
              f"{logging_tuple[0]:8.4f} | {logging_tuple[1]:8.4f} | "
              f"{returns.mean():8.6f} | {drawdown[-1]:8.6f} |")

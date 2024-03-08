import os
import torch
import time
import numpy as np
from typing import Tuple, List

from config.base import Config
from utils.env_utils import build_env

# reference: https://github.com/AI4Finance-Foundation/FinRL

def get_rewards_and_steps(
    env, actor, if_render: bool = False
) -> Tuple[float, int]:  # cumulative_rewards and episode_steps
    device = next(actor.parameters()).device  # net.parameters() is a Python generator.

    state = env.reset()[0]
    episode_steps = 0
    cumulative_returns = 0.0  # sum of rewards in an episode
    for episode_steps in range(12345):
        tensor_state = torch.as_tensor(
            state, dtype=torch.float32, device=device
        ).unsqueeze(0)
        tensor_action = actor(tensor_state)
        action = tensor_action.detach().cpu().numpy()[0]  # not need detach(), because using torch.no_grad() outside
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
        print(
            "\n| `step`: Number of samples, total training steps, or "
            "running times of `env.step()`."
            "\n| `time`: Time from the start of training to now."
            "\n| `avgR`: Avg. cumulative rewards, sum of rewards per episode."
            "\n| `stdR`: Std dev of cumulative rewards per episode."
            "\n| `avgS`: Avg. steps per episode."
            "\n| `objC`: Critic network objective or loss function."
            "\n| `objA`: Actor network objective, avg Q value of critic."
            f"\n| {'step':>8} | {'time':>8} | "
            f"{'avgR':>8} | {'stdR':>6} | "
            f"{'avgS':>6} | {'objC':>8} | "
            f"{'objA':>8}"
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

        used_time = time.time() - self.start_time
        self.recorder.append((self.total_step, used_time, avg_r))
        
        print(
            f"| `step`:{self.total_step:8.2e} | `time`:{used_time:8.0f} | "
            f"`avgR`:{avg_r:8.2f} | `stdR`:{std_r:6.2f} | `avgS`:{avg_s:6.0f} | "
            f"`objC`:{logging_tuple[0]:8.2f} | `objA`:{logging_tuple[1]:8.2f} |"
        )
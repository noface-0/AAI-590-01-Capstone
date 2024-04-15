import gym
import inspect

# reference: https://github.com/AI4Finance-Foundation/FinRL

def build_env(env_class=None, env_args=None):
    def kwargs_filter(function, kwargs: dict) -> dict:
        sign = inspect.signature(function).parameters.values()
        sign = {val.name for val in sign}
        return {key: kwargs[key] for key in sign.intersection(kwargs.keys())}

    if env_class.__module__ == 'gym.envs.registration':  # special rule
        env = env_class(id=env_args['env_name'])
    else:
        filtered_args = kwargs_filter(env_class.__init__, env_args.copy())
        env = env_class(**filtered_args)
    
    for attr_str in ('env_name', 'state_dim', 'action_dim', 'if_discrete'):
        setattr(env, attr_str, env_args[attr_str])
    
    return env
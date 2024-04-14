from __future__ import annotations

import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split

from processing.transform import AlpacaProcessor
from agents.dlr import DRLAgent
from config.training import TRAIN_SPLIT
from utils.plots import plot_drl_learning


BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.realpath(__file__))
)


def train(
    start_date,
    end_date,
    ticker_list,
    time_interval,
    technical_indicator_list,
    drl_lib,
    env,
    model_name,
    if_vix=True,
    data=pd.DataFrame(),
    split=True,
    api_key=None,
    api_secret=None,
    api_url=None,
    objective: str='max_returns',
    process_data: bool=True,
    **kwargs,
):
    dp = AlpacaProcessor(
        api_key, 
        api_secret, 
        api_url, 
        save_scaler=True,
        time_interval=time_interval
    )

    if data.empty:
        # download data
        data = dp.download_data(
            ticker_list, 
            start_date, 
            end_date, 
            time_interval
        )
        process_data = True
    if split:
        data = data.sample(frac=1).reset_index(drop=True)
        
        train_data, test_data = train_test_split(
            data, test_size=1-TRAIN_SPLIT, stratify=data['tic']
        )
        data = train_data

    if process_data:
        data = dp.clean_data(data)

        if if_vix:
            data = dp.add_vix(data)
            # add vix to indicators here
        else:
            # add turbulence to indicators here
            data = dp.add_turbulence(data)

        data = dp.add_technical_indicators(data)
        data = dp.preprocess_data(data)

    price_array, tech_array, turbulence_array = dp.df_to_array(
        data, 
        tech_indicator_list=technical_indicator_list, 
        if_vix=if_vix
    )
    env_config = {
        "price_array": price_array,
        "tech_array": tech_array,
        "turbulence_array": turbulence_array,
        "if_train": True,
        "objective": objective,
        "agent": model_name
    }
    env_instance = env(config=env_config)

    # read parameters
    cwd = kwargs.get("cwd", "./" + str(model_name))

    DRLAgent_erl = DRLAgent
    break_step = kwargs.get("break_step", 1e6)
    erl_params = kwargs.get("erl_params")
    agent = DRLAgent_erl(
        env=env,
        price_array=price_array,
        tech_array=tech_array,
        turbulence_array=turbulence_array,
        objective=objective,
        agent=model_name

    )
    model = agent.get_model(model_name, model_kwargs=erl_params)
    trained_model = agent.train_model(
        model=model, cwd=cwd, total_timesteps=break_step
    )


def test(
    start_date,
    end_date,
    ticker_list,
    time_interval,
    technical_indicator_list,
    drl_lib,
    env,
    model_name,
    if_vix=True,
    data=pd.DataFrame(),
    split=True,
    api_key=None,
    api_secret=None,
    api_url=None,
    objective: str='max_returns',
    process_data: bool=True,
    **kwargs,
):
    dp = AlpacaProcessor(
        api_key, 
        api_secret, 
        api_url, 
        save_scaler=False,
        time_interval=time_interval
    )

    if data.empty:
        # download data
        data = dp.download_data(
            ticker_list, 
            start_date, 
            end_date, 
            time_interval
        )
        process_data = True
    else:
        if split:
            data = data.sample(frac=1).reset_index(drop=True)
            
            train_data, test_data = train_test_split(
                data, test_size=1-TRAIN_SPLIT, stratify=data['tic']
            )
            data = test_data

    if process_data:
        data = dp.clean_data(data)

        if if_vix:
            data = dp.add_vix(data)
        else:
            data = dp.add_turbulence(data)

        data = dp.add_technical_indicators(data)
        data = dp.preprocess_data(data)

    price_array, tech_array, turbulence_array = dp.df_to_array(
        data, 
        tech_indicator_list=technical_indicator_list, 
        if_vix=if_vix
    )

    env_config = {
        "price_array": price_array,
        "tech_array": tech_array,
        "turbulence_array": turbulence_array,
        "if_train": False,
        "objective": objective,
        "agent": model_name
    }
    env_instance = env(config=env_config)

    # load elegantrl needs state dim, action dim and net dim
    net_dimension = kwargs.get("net_dimension", 2**7)
    cwd = kwargs.get("cwd", "./" + str(model_name))
    print("price_array: ", len(price_array))

    DRLAgent_erl = DRLAgent
    episode_metrics = DRLAgent_erl.DRL_prediction(
        model_name=model_name,
        cwd=cwd,
        net_dimension=net_dimension,
        environment=env_instance,
    )
    episode_total_assets = episode_metrics[0]
    episode_return = episode_metrics[1]
    drawdown = episode_metrics[2]
    sharpe_ratio = episode_metrics[3]

    eval_file_path = (
        f'{BASE_DIR}/models/runs/drl/{objective}/{model_name}/evalutation.json'
    )
    os.makedirs(os.path.dirname(eval_file_path), exist_ok=True)

    eval_dict = {
        "final_episode_return": episode_return,
        "max_drawdown": drawdown,
        "sharpe_ratio": sharpe_ratio
    }

    with open(eval_file_path, 'w') as f:
        json.dump(eval_dict, f)

    learning_metrics_path = (
        f'models/runs/drl/{objective}/{model_name}/'
        'papertrading_erl/learning_metrics.txt'
    )
    plot_drl_learning(
        path=learning_metrics_path,
        eval_path=eval_file_path,
        agent=model_name
    )
    return episode_total_assets, episode_return, drawdown, sharpe_ratio
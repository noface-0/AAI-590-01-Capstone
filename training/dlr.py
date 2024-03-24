from __future__ import annotations

import pandas as pd
from processing.transform import AlpacaProcessor
from agents.dlr import DRLAgent
from config.training import TRAIN_SPLIT


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
    else:
        if split:
            train_size = int(len(data) * TRAIN_SPLIT)
            train_data = data[:train_size]
            test_data = data[train_size:]
            data = train_data

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
        "if_train": True,
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
    **kwargs,
):
    dp = AlpacaProcessor(
        api_key, 
        api_secret, 
        api_url,
        time_interval=time_interval
    )
    # download data
    if data.empty:
        data = dp.download_data(
            ticker_list, 
            start_date, 
            end_date, 
            time_interval
        )
    else:
        if split:
            train_size = int(len(data) * TRAIN_SPLIT)
            train_data = data[:train_size]
            test_data = data[train_size:]
            data = test_data

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
    }
    env_instance = env(config=env_config)

    # load elegantrl needs state dim, action dim and net dim
    net_dimension = kwargs.get("net_dimension", 2**7)
    cwd = kwargs.get("cwd", "./" + str(model_name))
    print("price_array: ", len(price_array))

    DRLAgent_erl = DRLAgent
    episode_total_assets, episode_return, drawdown = DRLAgent_erl.DRL_prediction(
        model_name=model_name,
        cwd=cwd,
        net_dimension=net_dimension,
        environment=env_instance,
    )
    return episode_total_assets, episode_return, drawdown
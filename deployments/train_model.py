import os
import io
import json
import shutil
import argparse
import pandas as pd
import boto3
import logging
import torch

from processing.extract import download_data
from environments.base import StockTradingEnv
from training.dlr import train as drl_train, test as drl_test
from training.fnn import train as fnn_train, test as fnn_test
from training.ga import evolve_portfolio
from config.indicators import INDICATORS
from config.tickers import DOW_30_TICKER, SP_500_TICKER
from config.models import (
    ERL_PARAMS, 
    SAC_PARAMS,
    PPO_PARAMS, 
    TD3_PARAMS,
    GA_PARAMS
)
from config.base import Config
from config.training import (
    TIME_INTERVAL,
    TRAIN_START_DATE,
    TRAIN_END_DATE,
    TEST_START_DATE,
    TEST_END_DATE,
    AGENT,
    FNN_EPOCHS,
    FNN_TRIALS,
    OBJECTIVE
)
from utils.utils import get_var
from deployments.s3_utils import (
    load_data_from_s3,
    load_model_from_local_path,
    save_model_to_s3,
    save_file_to_s3,
    get_secret
)


API_KEY = get_var("ALPACA_API_KEY")
API_SECRET = get_var("ALPACA_API_SECRET")
API_BASE_URL = get_var("ALPACA_API_BASE_URL")

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.realpath(__file__))
)
time_interval = int(TIME_INTERVAL.strip("Min"))

exploration_tickers = DOW_30_TICKER #+ SP_500_TICKER


def train_model(
        train_data=pd.DataFrame(), 
        validation_data=pd.DataFrame(),
        api_key=API_KEY,
        api_secret=API_SECRET,
        api_url=API_BASE_URL,
        agent: str=AGENT,
        process_data=True
):
    # Initialize environment
    env = StockTradingEnv

    agent_configs = {
        "ppo": PPO_PARAMS,
        "sac": SAC_PARAMS,
        "td3": TD3_PARAMS,
    }
    params = agent_configs.get(agent)

    if train_data.empty or validation_data.empty:
        logging.info("No training data provided. Auto-downloading.")
        train_data = download_data(
            ticker_list=exploration_tickers,
            start_date=TRAIN_START_DATE,
            end_date=TEST_END_DATE,
            time_interval=TIME_INTERVAL,
            api_key=API_KEY,
            api_secret=API_SECRET,
            api_base_url=API_BASE_URL
        )
        validation_data = train_data
        split = True
    else:
        split = False

    print("Starting training phase...")
    # fnn will split data if auto-downloading
    fnn_train(
        data=train_data,
        start_date=TRAIN_START_DATE,
        end_date=TEST_END_DATE,
        ticker_list=exploration_tickers,
        time_interval=TIME_INTERVAL,
        if_vix=True,
        api_key=API_KEY,
        api_secret=API_SECRET,
        api_url=API_BASE_URL,
        forecast_steps=60, # this can be set to time_interval
        n_trials=FNN_TRIALS,
        num_epochs=FNN_EPOCHS,
        process_data=process_data
    )

    optimized_portfolio = evolve_portfolio(
        objective=OBJECTIVE,
        num_generations=GA_PARAMS['num_generations'],
        mutation_rate=GA_PARAMS['mutation_rate'],
        start_date=TRAIN_START_DATE,
        end_date=TEST_END_DATE,
        ticker_list=exploration_tickers,
        time_interval=TIME_INTERVAL,
        data=train_data
    )

    drl_train(
        data=train_data,
        start_date=TRAIN_START_DATE,
        end_date=TRAIN_END_DATE,
        ticker_list=exploration_tickers,
        time_interval=TIME_INTERVAL, 
        technical_indicator_list=INDICATORS,
        drl_lib='elegantrl',
        env=env,
        model_name=agent,
        if_vix=True,
        erl_params=params,
        cwd=f'{BASE_DIR}/models/runs/drl/{OBJECTIVE}/{AGENT}/papertrading_erl',
        break_step=1000000,
        split=split,
        api_key=api_key,
        api_secret=api_secret,
        api_url=api_url,
        objective=OBJECTIVE,
        process_data=process_data
    )
    
    # Testing phase
    print("Starting validation phase...")
    # fnn will split data if auto-downloading
    fnn_test(
        data=train_data,
        start_date=TRAIN_START_DATE,
        end_date=TEST_END_DATE,
        ticker_list=exploration_tickers,
        time_interval=TIME_INTERVAL,
        if_vix=True,
        api_key=API_KEY,
        api_secret=API_SECRET,
        api_url=API_BASE_URL,
        forecast_steps=100, # this can be set to time_interval
        process_data=process_data
    )
    account_value_erl = drl_test(
        data=validation_data,
        start_date=TEST_START_DATE,
        end_date=TEST_END_DATE,
        ticker_list=exploration_tickers,
        time_interval=TIME_INTERVAL,
        technical_indicator_list=INDICATORS,
        drl_lib='elegantrl',
        env=env,
        model_name=agent,
        if_vix=True,
        cwd=f'{BASE_DIR}/models/runs/drl/{OBJECTIVE}/{AGENT}/papertrading_erl',
        net_dimension=Config().net_dims,
        split=split,
        api_key=api_key,
        api_secret=api_secret,
        api_url=api_url,
        objective=OBJECTIVE,
        process_data=process_data
    )
    print(
        "DRL Validation phase completed. Final account value:", 
        account_value_erl[0]
    )

    if not train_data.empty:
        if not validation_data.empty:
            full_data_df = pd.concat(
                [train_data, validation_data], 
                ignore_index=True
            )
        else:
            full_data_df = train_data.copy()
    else:
        full_data_df = pd.DataFrame()

    print("Starting full data training phase...")
    drl_train(
        data=full_data_df,
        start_date=TRAIN_START_DATE,
        end_date=TEST_END_DATE,
        ticker_list=exploration_tickers,
        time_interval=TIME_INTERVAL, 
        technical_indicator_list=INDICATORS,
        drl_lib='elegantrl',
        env=env,
        model_name=agent,
        if_vix=True,
        erl_params=params,
        cwd=f'{BASE_DIR}/models/runs/drl/{OBJECTIVE}/{AGENT}/papertrading_erl_retrain',
        break_step=1000000,
        split=False,
        api_key=api_key,
        api_secret=api_secret,
        api_url=api_url,
        objective=OBJECTIVE,
        process_data=process_data
    )


def copy_file(local_source_path: str, destination_path: str):
    """
    Copy a file from a local source path to another specified path.

    :param local_source_path: Path to the source file.
    :param destination_path: Path where the file should be saved.
    """
    destination_dir = os.path.dirname(destination_path)
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    shutil.copy2(local_source_path, destination_path)

    print(f"File successfully copied to {destination_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process input data for training."
    )
    parser.add_argument(
        '--training', 
        type=str, 
        default=os.environ.get('S3_TRAINING')
    )
    parser.add_argument(
        '--validation', 
        type=str, 
        default=os.environ.get('S3_VALIDATION')
    )
    args = parser.parse_args()

    train_input = os.environ.get('S3_TRAINING')
    val_input = os.environ.get('S3_VALIDATION')

    # train_data = load_data_from_s3(train_input)
    # val_data = load_data_from_s3(val_input)

    # train_data_df = pd.read_parquet(io.BytesIO(train_data))
    # validation_data_df = pd.read_parquet(io.BytesIO(val_data))

    api_key = get_secret("ALPACA_API_KEY")
    api_secret = get_secret("ALPACA_API_SECRET")
    api_url = get_secret("ALPACA_API_BASE_URL")

    train_model(
        # train_data=train_data_df, 
        # validation_data=validation_data_df,
        api_key=api_key,
        api_secret=api_secret,
        api_url=api_url
    )

    bucket_name = 'rl-trading-v1-runs'
    local_path = f'{BASE_DIR}/models/runs/drl/{OBJECTIVE}/{AGENT}/papertrading_erl_retrain/actor.pth'
    local_filename = os.path.basename(local_path)
    local_eval_path = f'{BASE_DIR}/models/runs/drl/{OBJECTIVE}/{AGENT}/evaluation.json'
    destination_path = os.path.join("/opt/ml/model", local_filename)

    eval_s3_path = f"runs/evaluation/drl/{OBJECTIVE}/{AGENT}/evaluation.json"
    model_s3_path = 'runs/models/actor.pth'

    model = load_model_from_local_path(local_path)

    copy_file(local_path, destination_path)
    save_file_to_s3(local_eval_path, bucket_name, eval_s3_path)
    save_model_to_s3(model, bucket_name, model_s3_path)

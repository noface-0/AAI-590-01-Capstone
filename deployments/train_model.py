import os
import io
import shutil
import argparse
import pandas as pd
import boto3
import logging
import torch

from environments.base import StockTradingEnv
from training.train_test import train, test
from config.indicators import INDICATORS
from config.tickers import DOW_30_TICKER
from config.models import ERL_PARAMS, SAC_PARAMS
from config.training import (
    TIME_INTERVAL,
    TRAIN_START_DATE,
    TRAIN_END_DATE,
    TEST_START_DATE,
    TEST_END_DATE,
    AGENT
)
from deployments.s3_utils import (
    load_data_from_s3,
    load_model_from_local_path,
    save_model_to_s3,
    save_file_to_s3,
    get_secret
)


def train_model(
        train_data=pd.DataFrame(), 
        validation_data=pd.DataFrame(),
        api_key=None,
        api_secret=None,
        api_url=None,
):
    # Initialize environment
    env = StockTradingEnv

    agent_configs = {
        "ppo": ERL_PARAMS,
        "sac": ERL_PARAMS
    }
    params = agent_configs.get(AGENT)

    if train_data.empty or validation_data.empty:
        logging.info("No training data provided. Auto-downloading.")
        split = True
    else:
        split = False

    # Training phase
    print("Starting training phase...")
    train(
        data=train_data,
        start_date=TRAIN_START_DATE,
        end_date=TRAIN_END_DATE,
        ticker_list=DOW_30_TICKER,
        time_interval=TIME_INTERVAL, 
        technical_indicator_list=INDICATORS,
        drl_lib='elegantrl',
        env=env,
        model_name=AGENT,
        if_vix=True,
        erl_params=params,
        cwd=f'{BASE_DIR}/models/runs/papertrading_erl',
        break_step=1e6,
        split=split,
        api_key=api_key,
        api_secret=api_secret,
        api_url=api_url
    )
    
    # Testing phase
    print("Starting validation phase...")
    account_value_erl = test(
        data=validation_data,
        start_date=TEST_START_DATE,
        end_date=TEST_END_DATE,
        ticker_list=DOW_30_TICKER,
        time_interval=TIME_INTERVAL,
        technical_indicator_list=INDICATORS,
        drl_lib='elegantrl',
        env=env,
        model_name=AGENT,
        if_vix=True,
        cwd=f'{BASE_DIR}/models/runs/papertrading_erl',
        net_dimension=params['net_dimension'],
        split=split,
        api_key=api_key,
        api_secret=api_secret,
        api_url=api_url
    )
    print(
        "Validation phase completed. Final account value:", 
        account_value_erl
    )

    full_data_df = pd.concat(
        [train_data, validation_data], ignore_index=True
    ) if not train_data.empty else pd.DataFrame()

    print("Starting full data training phase...")
    train(
        data=full_data_df,
        start_date=TRAIN_START_DATE,
        end_date=TEST_END_DATE,
        ticker_list=DOW_30_TICKER,
        time_interval=TIME_INTERVAL, 
        technical_indicator_list=INDICATORS,
        drl_lib='elegantrl',
        env=env,
        model_name=AGENT,
        if_vix=True,
        erl_params=params,
        cwd=f'{BASE_DIR}/models/runs/papertrading_erl_retrain',
        break_step=1e6,
        split=False,
        api_key=api_key,
        api_secret=api_secret,
        api_url=api_url
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
    import os
    BASE_DIR = os.path.dirname(
        os.path.dirname(os.path.realpath(__file__))
    )

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
    local_path = f'{BASE_DIR}/models/runs/papertrading_erl_retrain/actor.pth'
    local_filename = os.path.basename(local_path)
    local_eval_path = f'{BASE_DIR}/models/runs/eval/evaluation.json'
    destination_path = os.path.join("/opt/ml/model", local_filename)

    eval_s3_path = "runs/evaluation/evaluation.json"
    model_s3_path = 'runs/models/actor.pth'

    model = load_model_from_local_path(local_path)

    copy_file(local_path, destination_path)
    save_file_to_s3(local_eval_path, bucket_name, eval_s3_path)
    save_model_to_s3(model, bucket_name, model_s3_path)

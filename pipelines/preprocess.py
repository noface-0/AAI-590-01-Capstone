import os
import copy
import argparse
import pandas as pd
import numpy as np
import boto3
import logging
import pathlib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


BASE_DIR = os.path.dirname(os.path.realpath(__file__))


def process_data(df):
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )
    
    numerical_cols = [
        col for col in 
        df.select_dtypes(include=['float64', 'int64']).columns 
        if col != "timestamp"
    ]
    
    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_cols)
        ],
        remainder='passthrough'
    )
    
    df_transformed = preprocess.fit_transform(df)
    
    transformed_columns = preprocess.transformers_[0][2] + \
        [col for col in df.columns if col 
         not in numerical_cols or col == "timestamp"]
    
    df_reconstructed = pd.DataFrame(
        df_transformed, columns=transformed_columns
    )
    
    df_shuffled = df_reconstructed.sample(frac=1, random_state=42)
    train, temp = train_test_split(
        df_shuffled, test_size=0.3, random_state=42
    )
    validation, test = train_test_split(
        temp, test_size=(0.5), random_state=42
    )
    
    opt_dir = pathlib.Path("/opt/ml/processing")
    train_dir = opt_dir / 'training'
    validation_dir = opt_dir / 'validation'
    test_dir = opt_dir / 'testing'

    for directory in [train_dir, validation_dir, test_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    train.to_parquet(train_dir / "training.parquet", index=False)
    validation.to_parquet(validation_dir / "validation.parquet", index=False)
    test.to_parquet(test_dir / "testing.parquet", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process input data for training."
    )
    parser.add_argument(
        "--input-data", 
        type=str, 
        required=True,
        help="The input data."
    )
    args = parser.parse_args()

    base_dir = "/opt/ml/processing"
    pathlib.Path(f"{base_dir}/data").mkdir(parents=True, exist_ok=True)
    input_data = args.input_data
    bucket = input_data.split("/")[2]
    key = "/".join(input_data.split("/")[3:])

    logging.info("Downloading data from bucket: %s, key: %s", bucket, key)
    fn = f"{base_dir}/data/extracted_stocks.parquet"
    s3 = boto3.resource("s3")
    s3.Bucket(bucket).download_file(key, fn)

    input_df = pd.read_parquet(fn)

    process_data(input_df)
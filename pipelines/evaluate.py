import json
import shutil
import os
import argparse

from deployments.s3_utils import (
    get_secret,
    load_data_from_s3
)

# evaluation already performed. Just extracting for Sagemaker condition step
if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.realpath(__file__))

    parser = argparse.ArgumentParser(
        description="Process input data for testing."
    )
    parser.add_argument(
        '--testing', 
        type=str, 
        default=os.environ.get('S3_TESTING')
    )
    args = parser.parse_args()

    # eval has already taken place in the training
    bucket_name = 'rl-trading-v1-runs'
    eval_s3_path = f"s3://{bucket_name}/runs/evaluation/evaluation.json"

    evaluation_data = load_data_from_s3(eval_s3_path)
    evaluation_data_json = json.loads(evaluation_data.decode('utf-8'))

    evaluation_output_dir = "/opt/ml/processing/evaluation"
    evaluation_output_file = "evaluation.json"
    evaluation_output_path = os.path.join(
        evaluation_output_dir, evaluation_output_file
    )

    try:
        if os.path.exists(evaluation_output_dir):
            print(f"Directory {evaluation_output_dir} exists.")
        else:
            os.makedirs(evaluation_output_dir, exist_ok=True)
            print(f"Directory {evaluation_output_dir} created.")
    except Exception as e:
        print(f"Failed to prepare the directory {evaluation_output_dir}. "
              f"Reason: {e}")

    with open(evaluation_output_path, "w") as f:
        f.write(json.dumps(evaluation_data_json))
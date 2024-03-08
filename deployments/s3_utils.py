import boto3
import torch
import io


def load_model_from_s3(bucket_name, s3_path):
    """
    Load a PyTorch model directly from an S3 bucket.
    """
    s3 = boto3.client('s3')
    obj = s3.get_object(Bucket=bucket_name, Key=s3_path)
    bytestream = io.BytesIO(obj['Body'].read())
    model = torch.load(bytestream)
    return model


def save_model_from_s3(
        bucket_name, 
        s3_path, 
        local_file_path
):
    s3 = boto3.client('s3')
    obj = s3.get_object(Bucket=bucket_name, Key=s3_path)
    bytestream = io.BytesIO(obj['Body'].read())
    
    with open(local_file_path, 'wb') as local_file:
        local_file.write(bytestream.getvalue())


def save_model_to_s3(model, bucket_name, s3_path):
    """
    Save a PyTorch model to an S3 bucket.
    """
    byte_stream = io.BytesIO()
    torch.save(model, byte_stream)
    byte_stream.seek(0)
    
    s3 = boto3.client('s3')
    try:
        s3.upload_fileobj(byte_stream, bucket_name, s3_path)
        print(
            f"Model successfully uploaded to s3://{bucket_name}/{s3_path}"
        )
    except Exception as e:
        print(
            f"Failed to upload the model to s3://{bucket_name}/{s3_path}. "
            f"Error: {e}"
        )


def save_file_to_s3(local_file_path, bucket_name, s3_path):
    """
    Save a file to an S3 bucket from a local file path.

    Parameters:
    - local_file_path (str): The local file path of the file to upload.
    - bucket_name (str): The name of the S3 bucket.
    - s3_path (str): The S3 path where the file will be saved.
    """
    s3 = boto3.client('s3')
    try:
        with open(local_file_path, 'rb') as file:
            s3.upload_fileobj(file, bucket_name, s3_path)
        print(
            f"File successfully uploaded to s3://{bucket_name}/{s3_path}"
        )
    except Exception as e:
        print(
            f"Failed to upload the file to s3://{bucket_name}/{s3_path}. "
            f"Error: {e}"
        )


def load_data_from_s3(s3_path: str):
    """
    Load data from an S3 path.

    Parameters:
    - s3_path (str): The S3 path to the data, in the format s3://bucket/key

    Returns:
    - data (bytes): The data read from the S3 object.
    """
    path_parts = s3_path.split("/")
    bucket = path_parts[2]
    key = "/".join(path_parts[3:])
    
    s3 = boto3.client('s3')

    obj = s3.get_object(Bucket=bucket, Key=key)

    data = obj['Body'].read()
    
    return data


def load_model_from_local_path(local_path):
    """
    Load a PyTorch model from a local file path.

    Parameters:
    - local_path (str): The local file path to the .pth file.

    Returns:
    - model: The loaded PyTorch model.
    """
    model = torch.load(local_path)
    return model


def create_secret(
        key_name: str, 
        key_value: str,
        description: str=None
):
    client = boto3.client('secretsmanager')
    response = client.create_secret(
        Name=key_name,
        Description=description,
        SecretString=key_value
    )


def get_secret(
        secret_name: str, 
        region_name: str = "us-east-1"
) -> str:
    """
    Retrieve a secret from AWS Secrets Manager.

    Parameters:
    - secret_name (str): The name of the secret.
    - region_name (str): The AWS region where the secret is stored. 
        Default is 'us-east-1'.

    Returns:
    - secret (str): The secret value.
    """
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )

    try:
        get_secret_value_response = client.get_secret_value(
            SecretId=secret_name
        )
    except Exception as e:
        print(f"Error retrieving secret {secret_name}: {e}")
        return None

    if 'SecretString' in get_secret_value_response:
        secret = get_secret_value_response['SecretString']
    else:
        secret = get_secret_value_response['SecretBinary']

    return secret
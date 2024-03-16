# Only to be used when requested the full historical dataset 
# 
# https://firstratedata.com/api/data_file?type=stock&period=full&ticker_range=C&timeframe=1day&adjustment=adj_split&userid=PvghgBS6m0mmDJDfhI678w

# To be used to request the most recent day's data
# 
# https://firstratedata.com/api/data_file?type=stock&period=full&ticker_range=A&adjustment=adj_split&timeframe=1hour&userid=PvghgBS6m0mmDJDfhI678w

from dask.distributed import Client
import os
import boto3
from botocore.exceptions import ClientError
import requests
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
import dask.bag as db
from io import BytesIO
from zipfile import ZipFile
import time
import timeit

# AWS credentials, fetched from environment variables for security
ACCESS_KEY = os.getenv('AWS_ACCESS_KEY_ID')
SECRET_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')

# S3 Bucket configuration
BUCKET_NAME = "capstone-data90210"

# PATHS
RAW_DATA_PATH = "rawdata/"

S3_CLIENT = boto3.client('s3', aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY)

# Checks if the specified S3 bucket exists.
def bucket_exists(bucket_name):
    try:
        S3_CLIENT.head_bucket(Bucket=bucket_name)
        return True
    except ClientError as error:
        if error.response['Error']['Code'] == '404':
            return False
        else:
            raise

# Ensures that the specified path exists within the S3 bucket. Creates the path if it does not exist.
def ensure_s3_path(bucket_name, path):
    if not path.endswith('/'):
        path += '/'

    s3 = boto3.resource('s3', aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY)
    bucket = s3.Bucket(bucket_name)
    
    objs = list(bucket.objects.filter(Prefix=path))

    if not objs or all(o.key != path for o in objs):
        try:
            bucket.put_object(Key=path)
            print(f"Created path '{path}' in bucket '{bucket_name}'.")
        except ClientError as error:
            print(f"Failed to create path '{path}' in bucket '{bucket_name}'. Reason: {error}")


def remove_from_s3(bucket_name, path):
    try:
        response = S3_CLIENT.delete_object(Bucket=bucket_name, Key=path)
        print(f"Deleted '{path}' from bucket '{bucket_name}'.")
    except ClientError as error:
        print(f"Failed to delete '{path}' from bucket '{bucket_name}'. Reason: {error}")


def download_and_upload_to_s3(bucket_name, s3_path, url, file_name):
    try:
        print("Downloading " + file_name + " from URL...")
        # Download the file from the URL
        response = requests.get(url)
        response.raise_for_status()

        # Upload the content to S3 with the specified path and filename
        s3_key = f"{s3_path}{file_name}" if s3_path else file_name
        
        S3_CLIENT.put_object(Body=response.content, Bucket=bucket_name, Key=s3_key)
        print(f"Uploaded '{file_name}' to '{s3_key}' in bucket '{bucket_name}'.")
    except requests.RequestException as e:
        print(f"Error downloading the file from the URL: {e}")
    except ClientError as error:
        print(f"Failed to upload '{file_name}' to bucket '{bucket_name}'. Reason: {error}")


def is_s3_folder_empty(bucket_name, folder_path):
    try:
        response = S3_CLIENT.list_objects(Bucket=bucket_name, Prefix=folder_path)

        # If the folder is empty, 'Contents' key will not be present in the response
        print(response)
        return True
    except NoCredentialsError:
        print("Credentials not available.")
        return False
    except PartialCredentialsError:
        print("Partial credentials provided.")
        return False



def pull_whole_history():
    print("Pulling whole history")
    
    url_base = 'https://firstratedata.com/api/data_file?type=stock&period=full&ticker_range='
    timeframe = '&timeframe=1min&adjustment=adj_split&userid=PvghgBS6m0mmDJDfhI678w'
    character_range = list(map(chr, range(ord('A'), ord('Z')+1)))
    urls = [f"{url_base}{char}{timeframe}" for char in character_range]
    
    runtime = time.time()

    for url in urls:
        char = url.split('=')[3].split('&')[0]
        filename = f"{char}.zip"
        print(filename)

        # Use timeit for more accurate timing
        start_time = timeit.default_timer()
        
        download_and_upload_to_s3(BUCKET_NAME, RAW_DATA_PATH, url, filename)
        
        end_time = timeit.default_timer()
        elapsed_time = end_time - start_time
        print(f"Time taken: {elapsed_time} seconds")

    endtime = time.time()
    total_elapsed_time = endtime - runtime
    print(f"Total time taken: {total_elapsed_time} seconds")
    # unzip_files(RAW_DATA_PATH)

def main():
    # Ensure the S3 bucket exists
    if not bucket_exists(BUCKET_NAME):
        print(f"The bucket '{BUCKET_NAME}' does not exist. Exiting.")
        exit()

    print(f"The bucket '{BUCKET_NAME}' exists.")

    # Ensure the specified path exists in the S3 bucket
    ensure_s3_path(BUCKET_NAME, RAW_DATA_PATH)
    
    # Remove Universe File
    remove_from_s3(BUCKET_NAME, "universe.csv")
    
    # Download Universe File
    download_and_upload_to_s3(BUCKET_NAME, "","https://firstratedata.com/api/ticker_listing?type=stock&userid=PvghgBS6m0mmDJDfhI678w", "universe.csv")
    
    if is_s3_folder_empty(BUCKET_NAME, RAW_DATA_PATH):
        # If the folder is empty, execute the function to pull the whole history
        pull_whole_history()
    else:
        print(f"The folder '{RAW_DATA_PATH}' is not empty. Skipping pull_whole_history.")
    
    


if __name__ == '__main__':
    main()

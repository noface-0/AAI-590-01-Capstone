import boto3
import sagemaker
import alpaca_trade_api as tradeapi
import pandas as pd
from sagemaker.session import Session
from sagemaker.feature_store.feature_group import FeatureGroup
from concurrent.futures import ThreadPoolExecutor

from utils.utils import get_var


API_KEY = get_var("ALPACA_API_KEY")
API_SECRET = get_var("ALPACA_API_SECRET")
API_BASE_URL = get_var("ALPACA_API_BASE_URL")


def extract_stock_data(limit: int=None, upload: bool=True):
    region = boto3.Session().region_name

    boto_session = boto3.Session(region_name=region)
    sagemaker_session = sagemaker.session.Session()

    bucket_name = "stockdata90210"
    default_bucket = sagemaker_session.default_bucket()

    sagemaker_client = boto_session.client(
        service_name="sagemaker", region_name=region
    )
    featurestore_runtime = boto_session.client(
        service_name="sagemaker-featurestore-runtime", 
        region_name=region
    )
    feature_store_session = Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_featurestore_runtime_client=featurestore_runtime,
    )
    response = sagemaker_client.list_feature_groups(
        SortBy='CreationTime', SortOrder='Descending'
    )
    if not response['FeatureGroupSummaries']:
        raise Exception("No feature groups found.")
    latest_feature_group_name = response['FeatureGroupSummaries'] \
        [0]['FeatureGroupName']
    description = sagemaker_client.describe_feature_group(
        FeatureGroupName=latest_feature_group_name
    )
    stock_table = description['OfflineStoreConfig'] \
        ['DataCatalogConfig']['TableName']
    print(f"Using latest feature group: {latest_feature_group_name}")

    stock_feature_group = FeatureGroup(
        name=latest_feature_group_name,
        sagemaker_session=feature_store_session
    )
    stock_query = stock_feature_group.athena_query()

    if limit:
        query_string = (
            f'SELECT * FROM "{stock_table}" LIMIT {limit}'
        )
    else:
        query_string = (
            f'SELECT * FROM "{stock_table}"'
        )
    stock_query.run(
        query_string=query_string,
        output_location=f"s3://{bucket_name}/query_results/",
    )
    stock_query.wait()
    dataset = stock_query.as_dataframe()

    parquet_path = (
        f"s3://{default_bucket}/stock_data/extracted_stock_data.parquet"
    )
    if upload:
        dataset.to_parquet(parquet_path, engine='pyarrow', index=False)

    return dataset


def download_data(
    ticker_list, 
    start_date, 
    end_date, 
    time_interval, 
    api_key=API_KEY, 
    api_secret=API_SECRET, 
    api_base_url=API_BASE_URL
) -> pd.DataFrame:
    """
    Downloads data using Alpaca's tradeapi.REST method.

    Parameters:
    - ticker_list : list of strings, each string is a ticker
    - start_date : string in the format 'YYYY-MM-DD'
    - end_date : string in the format 'YYYY-MM-DD'
    - time_interval: string representing the interval ('1D', '1Min', etc.)
    - api_key: Alpaca API key
    - api_secret: Alpaca API secret
    - api_base_url: Alpaca API base URL

    Returns:
    - pd.DataFrame with the requested data
    """
    api = tradeapi.REST(api_key, api_secret, api_base_url, "v2")

    def _fetch_data_for_ticker(ticker, start_date, end_date, time_interval):
        bars = api.get_bars(
            ticker,
            time_interval,
            start=start_date.isoformat(),
            end=end_date.isoformat(),
        ).df
        bars["symbol"] = ticker
        return bars

    NY = "America/New_York"
    start_date = pd.Timestamp(start_date + " 09:30:00", tz=NY)
    end_date = pd.Timestamp(end_date + " 15:59:00", tz=NY)

    # Use ThreadPoolExecutor to fetch data for
    # multiple tickers concurrently
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [
            executor.submit(
                _fetch_data_for_ticker,
                ticker,
                start_date,
                end_date,
                time_interval,
            )
            for ticker in ticker_list
        ]
        data_list = [future.result() for future in futures]
    # Combine the data
    data_df = pd.concat(data_list, axis=0)
    # Convert the timezone
    data_df = data_df.tz_convert(NY)
    # If time_interval is less than a day,
    # filter out the times outside of NYSE trading hours
    if pd.Timedelta(time_interval) < pd.Timedelta(days=1):
        data_df = data_df.between_time("09:30", "15:59")
    # Reset the index and rename the columns for consistency
    data_df = data_df.reset_index().rename(
        columns={
            "index": "timestamp",
            "symbol": "tic",
        }
    )
    # Sort the data by both timestamp and tic for consistent ordering
    data_df = data_df.sort_values(by=["tic", "timestamp"])
    # Reset the index and drop the old index column
    data_df = data_df.reset_index(drop=True)

    return data_df
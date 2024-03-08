import boto3
import sagemaker
from sagemaker.session import Session
from sagemaker.feature_store.feature_group import FeatureGroup


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
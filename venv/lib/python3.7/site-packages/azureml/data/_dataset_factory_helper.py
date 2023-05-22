# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Contains helper methods for dataset factory."""


def parse_target(target, add_managed_dataset_prefix=False):
    from azureml.data.azure_storage_datastore import AbstractAzureStorageDatastore
    from azureml.data.azure_data_lake_datastore import AbstractADLSDatastore
    from azureml.data.hdfs_datastore import HDFSDatastore
    from azureml.data.datapath import DataPath
    from azureml.data.constants import MANAGED_DATASET

    datastore = None
    relative_path = None

    if isinstance(target, AbstractAzureStorageDatastore) or isinstance(target, AbstractADLSDatastore):
        datastore = target
        relative_path = MANAGED_DATASET if add_managed_dataset_prefix else '/'
    elif isinstance(target, DataPath):
        datastore = target._datastore
        relative_path = (MANAGED_DATASET if add_managed_dataset_prefix else '/') \
            if target.path_on_datastore is None else target.path_on_datastore
    elif isinstance(target, tuple) and len(target) == 2:
        datastore = target[0]
        relative_path = target[1]
    if (not isinstance(datastore, AbstractAzureStorageDatastore)
            and not isinstance(datastore, AbstractADLSDatastore)
            and not isinstance(datastore, HDFSDatastore)):
        raise ValueError("The target type is not supported, target: {}".format(target))

    return datastore, relative_path


def _set_spark_config(datastore):
    from pyspark.sql import SparkSession
    from azureml.data.constants import AZURE_BLOB, AZURE_DATA_LAKE_GEN2, AZURE_DATA_LAKE
    spark = SparkSession.builder.getOrCreate()

    if datastore.datastore_type == AZURE_BLOB:
        account_name = datastore.account_name
        account_key = datastore.account_key
        endpoint = datastore.endpoint
        spark.conf.set('fs.azure.account.key.{}.blob.{}'.format(account_name, endpoint), account_key)
    elif datastore.datastore_type == AZURE_DATA_LAKE_GEN2:
        account_name = datastore.account_name
        client_id = datastore.client_id
        client_secret = datastore.client_secret
        endpoint = datastore.endpoint
        tenant_id = datastore.tenant_id
        authority_url = datastore.authority_url
        prefix = "fs.azure.account"
        provider = "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider"
        storage_account = "{}.dfs.{}".format(account_name, endpoint)
        spark.conf.set("{}.auth.type.{}".format(prefix, storage_account), "OAuth")
        spark.conf.set("{}.oauth.provider.type.{}".format(prefix, storage_account), provider)
        spark.conf.set("{}.oauth2.client.id.{}".format(prefix, storage_account), client_id)
        spark.conf.set("{}.oauth2.client.secret.{}".format(prefix, storage_account), client_secret)
        spark.conf.set("{}.oauth2.client.endpoint.{}".format(prefix, storage_account),
                       "{}/{}/oauth2/token".format(authority_url, tenant_id))
    elif datastore.datastore_type == AZURE_DATA_LAKE:
        client_id = datastore.client_id
        client_secret = datastore.client_secret
        tenant_id = datastore.tenant_id
        authority_url = datastore.authority_url
        prefix = "fs.adl"  # dfs.adls deprecated
        spark.conf.set("{}.oauth2.access.token.provider.type".format(prefix), "ClientCredential")
        spark.conf.set("{}.oauth2.client.id".format(prefix), client_id)
        spark.conf.set("{}.oauth2.credential".format(prefix), client_secret)
        spark.conf.set("{}.oauth2.refresh.url".format(prefix),
                       "{}/{}/oauth2/token".format(authority_url, tenant_id))
    else:
        raise ValueError(
            "The datastore type {} is not supported.".format(datastore.datastore_type))


def _get_output_uri(datastore, path):
    from azureml.data.constants import AZURE_BLOB, AZURE_DATA_LAKE_GEN2, AZURE_DATA_LAKE

    if datastore.datastore_type == AZURE_BLOB:
        output_uri = 'wasbs://{}@{}.blob.{}/{}'.format(datastore.container_name, datastore.account_name,
                                                       datastore.endpoint, path)
    elif datastore.datastore_type == AZURE_DATA_LAKE_GEN2:
        output_uri = 'abfss://{}@{}.dfs.{}/{}'.format(datastore.container_name, datastore.account_name,
                                                      datastore.endpoint, path)
    elif datastore.datastore_type == AZURE_DATA_LAKE:
        output_uri = 'adl://{}.azuredatalakestore.net/{}'.format(datastore.store_name, path)
    else:
        raise ValueError(
            "The datastore type {} is not supported.".format(datastore.datastore_type))

    return output_uri


def write_spark_dataframe(spark_dataframe, datastore, relative_path_with_guid, show_progress):
    console = get_progress_logger(show_progress)
    _set_spark_config(datastore)
    output_uri = _get_output_uri(datastore, relative_path_with_guid)

    console("Writing spark dataframe to {}".format(relative_path_with_guid))
    spark_dataframe.write.mode("overwrite").option("header", "true").format("parquet").save(output_uri)


def get_progress_logger(show_progress):
    import sys
    console = sys.stdout

    def log(message):
        if show_progress:
            console.write("{}\n".format(message))

    return log

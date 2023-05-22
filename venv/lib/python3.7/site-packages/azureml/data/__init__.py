# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""Contains modules supporting data representation for Datastore and Dataset in Azure Machine Learning.

This package contains core functionality supporting :class:`azureml.core.datastore.Datastore` and
:class:`azureml.core.dataset.Dataset` classes in the :mod:`azureml.core` package.
Datastore objects contain connection information to Azure storage services that can be easily referred to
by name without the need to work directly with or hard code connection information in scripts. Datastore
supports a number of different services represented by classes in this package, including
:class:`azureml.data.azure_storage_datastore.AzureBlobDatastore`,
:class:`azureml.data.azure_storage_datastore.AzureFileDatastore`, and
:class:`azureml.data.azure_data_lake_datastore.AzureDataLakeDatastore`. For a full
list of supported storage services, see the :class:`azureml.core.datastore.Datastore` class.

While a Datastore acts as a container for your data files, you can think of a Dataset as a reference or
pointer to specific data that's in your datastore. The following Datasets types are supported:

* :class:`azureml.data.TabularDataset` represents data in a tabular format created by parsing the provided
  file or list of files.

* :class:`azureml.data.FileDataset` references single or multiple files in your datastores or public URLs.

For more information, see the article [Add & register
datasets](https://docs.microsoft.com/azure/machine-learning/how-to-create-register-datasets).
To get started working with a datasets, see https://aka.ms/tabulardataset-samplenotebook and
https://aka.ms/filedataset-samplenotebook.
"""

from azureml._base_sdk_common import __version__ as VERSION
from .tabular_dataset import TabularDataset
from .file_dataset import FileDataset
from .dataset_factory import DataType
from .output_dataset_config import OutputFileDatasetConfig
from .output_dataset_config import HDFSOutputDatasetConfig
from .output_dataset_config import LinkFileOutputDatasetConfig, LinkTabularOutputDatasetConfig
from .datacache import DatacacheStore

__version__ = VERSION

__all__ = [
    'DataType',
    'FileDataset',
    'TabularDataset',
    'OutputFileDatasetConfig',
    'HDFSOutputDatasetConfig',
    'LinkFileOutputDatasetConfig',
    'LinkTabularOutputDatasetConfig',
    'DatacacheStore'
]

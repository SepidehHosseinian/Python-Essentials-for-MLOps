# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""This package contains classes used to manage compute targets in Azure Machine Learning.

For more information about choosing compute targets for training and deployment, see [What are compute targets in
Azure Machine Learning?](https://docs.microsoft.com/azure/machine-learning/concept-compute-target)
"""

from azureml._base_sdk_common import __version__ as VERSION
from .compute import ComputeTarget
from .aks import AksCompute
from .amlcompute import AmlCompute
from .batch import BatchCompute
from .computeinstance import ComputeInstance
from .dsvm import DsvmCompute
from .datafactory import DataFactoryCompute
from .adla import AdlaCompute
from .databricks import DatabricksCompute
from .hdinsight import HDInsightCompute
from .remote import RemoteCompute
from .kubernetescompute import KubernetesCompute
from .kusto import KustoCompute
from .synapse import SynapseCompute

__version__ = VERSION

__all__ = [
    'AdlaCompute',
    'AksCompute',
    'AmlCompute',
    'BatchCompute',
    'ComputeInstance',
    'ComputeTarget',
    'DatabricksCompute',
    'DataFactoryCompute',
    'DsvmCompute',
    'HDInsightCompute',
    'RemoteCompute',
    'KubernetesCompute',
    'KustoCompute',
    'SynapseCompute'
]

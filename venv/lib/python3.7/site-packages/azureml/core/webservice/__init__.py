# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""Contains functionality for deploying machine learning models as web service endpoints in Azure Machine Learning.

Deploying an Azure Machine Learning model as a web service creates an endpoint and a REST API. You can send
data to this API and receive the prediction returned by the model.

You create a web service when you deploy a :class:`azureml.core.model.Model` or :class:`azureml.core.image.Image`
to Azure Container Instances (:mod:`azureml.core.webservice.aci` module), Azure Kubernetes Service
(:mod:`azureml.core.webservice.aks` module) and Azure Kubernetes Endpoint (AksEndpoint), or field-programmable
gate arrays (FPGA). Deployment using a model is recommended for most use cases, while deployment using an image
is recommended for advanced use cases. Both types of deployment are supported in the classes in this module.
"""

from azureml._base_sdk_common import __version__ as VERSION
from .webservice import Webservice
from .aci import AciWebservice
from .aks import AksWebservice
from .aks import AksEndpoint
from .local import LocalWebservice
from .unknown_webservice import UnknownWebservice


__version__ = VERSION

__all__ = [
    'AciWebservice',
    'AksEndpoint',
    'AksWebservice',
    'LocalWebservice',
    'UnknownWebservice',
    'Webservice'
]

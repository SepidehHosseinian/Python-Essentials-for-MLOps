# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""Contains functionality for managing images that are deployed as web service endpoints in Azure Machine Learning.

This class is DEPRECATED. Use the :class:`azureml.core.Environment` class instead.

An Image is used to deploy a :class:`azureml.core.model.Model`, script, and associated files as a web service
endpoint or IoT Edge device. The endpoint handles incoming scoring requests and return predictions. This package's
key classes are the :class:`azureml.core.image.Image` class, the parent class of Azure Machine Learning images,
and the derived :class:`azureml.core.image.ContainerImage` class for Docker Images as well as preview Images like
FPGA.

Unless you have a workflow that specifically requires using images, you should instead use the
:class:`azureml.core.Environment` class to define your image. Then you can use the Environment object with
the :class:`azureml.core.model.Model` ``deploy()`` method to deploy the model as a web service.
You can also use the Model ``package()`` method to create an image that can be downloaded to your local
Docker install as an image or as a Dockerfile.

For information on using the Model class, see [Deploy models with Azure Machine
Learning](https://docs.microsoft.com/azure/machine-learning/how-to-deploy-and-where).

For information on using custom images, see [Deploy a model using a custom Docker base
image](https://docs.microsoft.com/azure/machine-learning/how-to-deploy-custom-docker-image).
"""

from azureml._base_sdk_common import __version__ as VERSION
from .image import Image
from .container import ContainerImage
from .unknown_image import UnknownImage


__version__ = VERSION

__all__ = [
    'ContainerImage',
    'Image',
    'UnknownImage'
]

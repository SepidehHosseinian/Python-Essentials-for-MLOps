# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Initialize _restclient."""

from .rest_client import RestClient
from .run_client import RunClient
from .experiment_client import ExperimentClient
from .jasmine_client import JasmineClient
from .models_client import ModelsClient
from .assets_client import AssetsClient
from .run_artifacts_client import RunArtifactsClient
from azureml._base_sdk_common import __version__ as VERSION
__version__ = VERSION

__all__ = ['RestClient', 'RunClient', 'ExperimentClient', 'JasmineClient', 'RunArtifactsClient', 'ModelsClient',
           'AssetsClient']

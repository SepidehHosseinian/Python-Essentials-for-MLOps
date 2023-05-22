# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Contains functionality for creating and managing reproducible environments in Azure Machine Learning.

Environments provide a way to manage software dependency so that controlled environments are reproducible with
minimal manual configuration as you move between local and distributed cloud development environments. An environment
encapsulates Python packages, environment variables, software settings for training and scoring scripts,
and run times on either Python, Spark, or Docker. For more information about using environments for training and
deployment with Azure Machine Learning, see [Create and manage reusable
environments](https://docs.microsoft.com/azure/machine-learning/how-to-use-environments).
"""
from __future__ import print_function
import collections
import logging
import json
import base64
import os
import subprocess
import zipfile
import tempfile
import sys
import time
import requests
import urllib
import hashlib

from collections import OrderedDict
from enum import Enum
from azure.core.exceptions import ResourceExistsError
from azureml._base_sdk_common.abstract_run_config_element import _AbstractRunConfigElement
from azureml._base_sdk_common.field_info import _FieldInfo
from azureml.core.conda_dependencies import CondaDependencies, PYTHON_DEFAULT_VERSION
from azureml.core.container_registry import ContainerRegistry
from azureml.core.databricks import DatabricksSection
from azureml.exceptions import UserErrorException, AzureMLException

from azureml._restclient.environment_client import EnvironmentClient
from azureml.core._serialization_utils import _serialize_to_dict, _deserialize_and_add_to_object
from azureml._project.project_manager import _get_tagged_image

try:
    # import SDK version
    from azureml._base_sdk_common._version import ver as SDKVERSION
except ImportError:
    # default to v0
    SDKVERSION = "0.0"

module_logger = logging.getLogger(__name__)

DEFAULT_CPU_IMAGE = _get_tagged_image("mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04")
DEFAULT_GPU_IMAGE = _get_tagged_image("mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.3-cudnn8-ubuntu20.04")
_DEFAULT_SHM_SIZE = "2g"

_CONDA_DEPENDENCIES_FILE_NAME = "conda_dependencies.yml"
_BASE_DOCKERFILE_FILE_NAME = "BaseDockerfile"
_DEFINITION_FILE_NAME = "azureml_environment.json"
_PRIVATE_PKG_CONTAINER_NAME = "azureml-private-packages"
_EMS_ORIGIN_NAME = "Environment"
_DOCKER_BUILD_CONTEXT_CONTAINER = "azureml"
_DOCKER_BUILD_CONTEXT_PATH_PREFIX = "DockerBuildContext"
_DOCKER_BUILD_CONTEXT_MAX_FILES = 100
_DOCKER_BUILD_CONTEXT_MAX_BYTES = 1024 * 1024
_DIR_HASH_BUFFER_SIZE = 65536


class PythonSection(_AbstractRunConfigElement):
    """Defines the Python environment and interpreter to use on a target compute for a run.

    This class is used in the :class:`azureml.core.environment.Environment` class.

    :var user_managed_dependencies: Indicates whether Azure Machine Learning reuses an existing Python
        environment.

        If set to True, you are responsible for ensuring that all the necessary packages are available in the
        Python environment you choose to run the script in. If False (the default), Azure will create a Python
        environment for you based on the conda dependencies specification. The environment is built once, and
        is be reused in subsequent executions as long as the conda dependencies remain unchanged.
    :vartype user_managed_dependencies: bool

    :var interpreter_path: The Python interpreter path. This is only used when user_managed_dependencies=True.
        The default is "python".
    :vartype interpreter_path: str

    :var conda_dependencies: Conda dependencies.
    :vartype conda_dependencies: azureml.core.conda_dependencies.CondaDependencies

    """

    # This is used to deserialize.
    # This is also the order for serialization into a file.
    _field_to_info_dict = collections.OrderedDict([
        ("user_managed_dependencies", _FieldInfo(bool, "user_managed_dependencies=True indicates that the environment"
                                                       "will be user managed. False indicates that AzureML will"
                                                       "manage the user environment.")),
        ("interpreter_path", _FieldInfo(str, "The python interpreter path")),
        ("_conda_dependencies_file", _FieldInfo(
            str, "Path to the conda dependencies file to use for this run. If a project\n"
                 "contains multiple programs with different sets of dependencies, it may be\n"
                 "convenient to manage those environments with separate files.",
            serialized_name="conda_dependencies_file")),
        ("_base_conda_environment", _FieldInfo(
            str, "The base conda environment used for incremental environment creation.",
            serialized_name="base_conda_environment")),
    ])

    def __init__(self, **kwargs):
        """Class PythonSection constructor."""
        super(PythonSection, self).__init__()

        self.interpreter_path = "python"
        self.user_managed_dependencies = False
        self._conda_dependencies_file = None
        self.conda_dependencies = None
        self._base_conda_environment = None

        options = {"_skip_defaults": False}
        options.update(kwargs)
        if not options["_skip_defaults"]:
            self.conda_dependencies = CondaDependencies()

        self._initialized = True

    @property
    def conda_dependencies_file(self):
        """DEPRECATED. The path to the conda dependencies file to use for this run.

        Use :meth:`from_existing_conda_environment`.
        """
        return self._conda_dependencies_file

    @conda_dependencies_file.setter
    def conda_dependencies_file(self, value):
        module_logger.warning("Property conda_dependencies_file is deprecated. "
                              "Use Environment.from_conda_specification")
        if value:
            self._conda_dependencies_file = value
        else:
            self._conda_dependencies_file = None


class PythonEnvironment(PythonSection):
    """DEPRECATED. Use the :class:`azureml.core.environment.PythonSection` class.

    .. remarks::

        This class is deprecated. Use the :class:`azureml.core.environment.PythonSection` class.
    """

    def __init__(self):
        """Class PythonEnvironment constructor."""
        super(PythonEnvironment, self).__init__()
        module_logger.warning(
            "'PythonEnvironment' will be deprecated soon. Please use PythonSection from 'azureml.core.environment'.")


class DockerImagePlatform(_AbstractRunConfigElement):
    """Defines a connection to an Azure Container Registry.

    :var os: Operating system. One of `Linux`, `Windows` or `OSX`. Default is `Linux`
    :vartype os: str

    :var architecture: CPU Architecture. Default is `amd64`
    :vartype architecture: str
    """

    # This is used to deserialize.
    # This is also the order for serialization into a file.
    _field_to_info_dict = collections.OrderedDict([
        ("os", _FieldInfo(str, "Operating System")),
        ("architecture", _FieldInfo(str, "Architecture"))])

    def __init__(self):
        """Class DockerImagePlatform constructor."""
        super(DockerImagePlatform).__init__()
        self.os = "Linux"
        self.architecture = "amd64"


class DockerBuildContext(_AbstractRunConfigElement):
    """Defines a Docker build context.

    :var azureml.core.environment.DockerBuildContext.location_type: Type of Docker build context location
    :vartype location_type: azureml.core.environment.DockerBuildContext.LocationType

    :var azureml.core.environment.DockerBuildContext.location: Docker build context location.
    :vartype location: str

    :var dockerfile_path: Path of the Dockerfile relative to the root of the build context, defaults to Dockerfile.
    :vartype dockerfile_path: str
    """

    class LocationType(Enum):
        """Type of Docker build context location.

        :param Git: Docker build context stored in a Git repository.
        :param StorageAccount: Docker build context stored in a storage account.
        """

        Git = "git"
        StorageAccount = "storageAccount"

    # This is used to deserialize.
    # This is also the order for serialization into a file.
    _field_to_info_dict = collections.OrderedDict([
        ("location_type", _FieldInfo(LocationType, "Type of Docker build context location")),
        ("location", _FieldInfo(str, "Docker build context location")),
        ("dockerfile_path", _FieldInfo(str, "Dockerfile path"))])

    def __init__(self, location_type=None, location=None, dockerfile_path="Dockerfile"):
        """Class DockerBuildContext constructor."""
        super(DockerBuildContext).__init__()
        self.location_type = location_type
        self.location = location
        self.dockerfile_path = dockerfile_path or "Dockerfile"
        self._initialized = True

    @staticmethod
    def from_local_directory(workspace, path, dockerfile_path="Dockerfile"):
        """Create DockerBuildContext object from a local directory containing a Docker build context.

        :param workspace: The workspace in which the environment will be created.
        :type workspace: azureml.core.workspace.Workspace

        :param path: Path to the directory containing the Docker build context.
        :type path: str

        :param dockerfile_path: Dockerfile path relative to the root of the Docker build context,
            defaults to Dockerfile.
        :type dockerfile_path: str

        :return: The DockerBuildContext object.
        :rtype: azureml.core.environment.DockerBuildContext
        """
        # get the files to upload, and warn if the directory may be too large
        files = Environment._get_files_to_upload(path,
                                                 _DOCKER_BUILD_CONTEXT_MAX_FILES,
                                                 _DOCKER_BUILD_CONTEXT_MAX_BYTES)
        if len(files) == 0:
            raise UserErrorException("No files found in {}.".format(path))

        # compute hash for all files and generate path prefix
        hash = Environment._hash_files(files.values())
        path_prefix = "{}/{}".format(_DOCKER_BUILD_CONTEXT_PATH_PREFIX, hash)

        # retrieve storage account name and key from workspace
        storage_account_full = workspace.get_details().get("storageAccount")
        if storage_account_full is None:
            raise AzureMLException("Workspace has no storage account.")
        storage_account = storage_account_full[storage_account_full.rfind("/") + 1:]
        storage_endpoint = workspace._auth_object._cloud_type.suffixes.storage_endpoint
        storage_account_key = workspace.list_keys().get("userStorageKey")
        if storage_account_key is None:
            raise AzureMLException("Unable to retrieve storage account key.")

        # upload files
        from azureml._vendor.azure_storage.blob import BlobServiceClient
        account_url = "https://{account_name}.blob.{endpoint}".format(
            account_name=storage_account, endpoint=storage_endpoint
        )
        blob_service_client = BlobServiceClient(
            account_url=account_url,
            credential={
                "account_name": storage_account,
                "account_key": storage_account_key
            }
        )
        try:
            blob_service_client.create_container(_DOCKER_BUILD_CONTEXT_CONTAINER)
        except ResourceExistsError:
            module_logger.info("Container {} already exists".format(_DOCKER_BUILD_CONTEXT_CONTAINER))
        logging.info("Uploading {} file{} from {} to workspace storage \
            account {}.".format(len(files), "s" if len(files) > 1 else "", path, storage_account))
        for relative_path, source_path in files.items():
            module_logger.debug("Uploading {}".format(relative_path))
            relative_path_normalized = relative_path.replace(os.sep, '/')
            target_path = "{}/{}".format(path_prefix, relative_path_normalized)
            with open(source_path, 'rb') as data:
                blob_service_client.get_blob_client(
                    container=_DOCKER_BUILD_CONTEXT_CONTAINER,
                    blob=target_path
                ).upload_blob(data, overwrite=True)

        location = blob_service_client.get_blob_client(
            container=_DOCKER_BUILD_CONTEXT_CONTAINER,
            blob=path_prefix + "/"
        ).url
        context = DockerBuildContext(DockerBuildContext.LocationType.StorageAccount,
                                     location,
                                     dockerfile_path)
        return context


class DockerSection(_AbstractRunConfigElement):
    """Defines settings to customize the Docker image built to the environment's specifications.

    The DockerSection class is used in the :class:`azureml.core.environment.Environment` class to
    customize and control the final resulting Docker image that contains the specified environment.

    .. remarks::

        The following example shows how to load docker steps as a string.

        .. code-block:: python

            from azureml.core import Environment
            myenv = Environment(name="myenv")
            # Specify docker steps as a string.
            dockerfile = r'''
            FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04
            RUN echo "Hello from custom container!"
            '''

            # Alternatively, load from a file.
            #with open("dockerfiles/Dockerfile", "r") as f:
            #    dockerfile=f.read()

            myenv.docker.base_dockerfile = dockerfile

        For more information about using Docker in environments, see the article `Enable
        Docker <https://docs.microsoft.com/azure/machine-learning/how-to-use-environments#enable-docker>`_.

    :var enabled: Indicates whether to perform this run inside a Docker container. Default is False.
         DEPRECATED: Use the azureml.core.runconfig.DockerConfiguration class.
    :vartype enabled: bool

    :var base_image: The base image used for Docker-based runs. Mutually exclusive with "base_dockerfile"
        and "build_context" variables. Example value: "ubuntu:latest".
    :vartype base_image: str

    :var base_dockerfile: The base Dockerfile used for Docker-based runs. Mutually exclusive with "base_image"
        and "build_context" variables. Example: line 1 "FROM ubuntu:latest" followed by line 2
        "RUN echo 'Hello world!'".
        The default is None.
    :vartype base_dockerfile: str

    :var build_context: The Docker build context to use to create the environment. Mutually
        exclusive with "base_image" and "base_dockerfile" variables.
        The default is None.
    :vartype build_context: azureml.core.environment.DockerBuildContext

    :var base_image_registry: Image registry that contains the base image.
    :vartype base_image_registry: azureml.core.container_registry.ContainerRegistry

    :var platform: Operating System and CPU architecture the image of the docker image.
    :vartype platform: azureml.core.environment.DockerImagePlatform

    :var enabled: Indicates whether to perform this run inside a Docker container. Default is False.
         DEPRECATED: Use the azureml.core.runconfig.DockerConfiguration class.
    :vartype enabled: bool

    :var shared_volumes: Indicates whether to use shared volumes. Set to False if necessary to work around
        shared volume bugs on Windows. The default is True.
        DEPRECATED: Use the azureml.core.runconfig.DockerConfiguration class.
    :vartype shared_volumes: bool

    :var gpu_support: DEPRECATED. Azure Machine Learning now automatically detects and uses
        NVIDIA Docker extension when available.
    :vartype gpu_support: bool

    :var arguments: Extra arguments to pass to the Docker run command. The default is None.
        DEPRECATED: Use the azureml.core.runconfig.DockerConfiguration class.
    :vartype arguments: builtin.list

    """

    # This is used to deserialize.
    # This is also the order for serialization into a file.
    _field_to_info_dict = collections.OrderedDict([
        ("_enabled", _FieldInfo(
            bool, "Set True to perform this run inside a Docker container.",
            serialized_name="enabled")),
        ("_base_image", _FieldInfo(
            str, "Base image used for Docker-based runs. Mutually exclusive with base_dockerfile and build_context.",
            serialized_name="base_image")),
        ("_base_dockerfile", _FieldInfo(
            str, "Base Dockerfile used for Docker-based runs. Mutually exclusive with base_image and build_context.",
            serialized_name="base_dockerfile")),
        ("_build_context", _FieldInfo(
            DockerBuildContext, "Build context used for Docker-based runs. "
            "Mutually exclusive with base_image and base_dockerfile.",
            serialized_name="build_context")),
        ("_shared_volumes", _FieldInfo(
            bool, "Set False if necessary to work around shared volume bugs.",
            serialized_name="shared_volumes")),
        ("_shm_size", _FieldInfo(
            str, "Shared memory size for Docker container. Default is {}.".format(_DEFAULT_SHM_SIZE),
            serialized_name="shm_size")),
        ("_arguments", _FieldInfo(
            list, "Extra arguments to the Docker run command.", list_element_type=str,
            serialized_name="arguments")),
        ("base_image_registry", _FieldInfo(ContainerRegistry,
                                           "Image registry that contains the base image.")),
        ("platform", _FieldInfo(DockerImagePlatform,
                                "Docker image platform"))])

    def __init__(self, **kwargs):
        """Class DockerSection constructor."""
        super(DockerSection, self).__init__()
        self._enabled = False
        self._shm_size = None
        self._shared_volumes = True
        self._arguments = list()
        self._base_image = None
        self.base_image_registry = ContainerRegistry()
        self._base_dockerfile = None
        self._build_context = None
        self.platform = DockerImagePlatform()

        options = {"_skip_defaults": False}
        options.update(kwargs)
        if not options["_skip_defaults"]:
            self._shm_size = _DEFAULT_SHM_SIZE
            self._base_image = DEFAULT_CPU_IMAGE

        self._initialized = True

    @property
    def base_image(self):
        """Get or set base image used for Docker-based runs."""
        return self._base_image

    @base_image.setter
    def base_image(self, value):
        if value:
            self._base_image = value
            if self._base_dockerfile:
                self._base_dockerfile = None
                module_logger.warning("Property base_dockerfile is mutually exclusive with base_image. "
                                      "Reset base_dockerfile to None")
            if self._build_context:
                self._build_context = None
                module_logger.warning("Property build_context is mutually exclusive with base_image. "
                                      "Reset build_context to None")
        else:
            self._base_image = None

    @property
    def base_dockerfile(self):
        """Get or set base dockerfile used for Docker-based runs."""
        return self._base_dockerfile

    @base_dockerfile.setter
    def base_dockerfile(self, value):
        if value:
            if hasattr(value, "read") and callable(getattr(value, "read")):
                self._base_dockerfile = value.read()
            elif os.path.isfile(value):
                # should nicely fail if not a string or valid path
                with open(value) as f:
                    self._base_dockerfile = f.read()
            else:
                self._base_dockerfile = value
            if self._base_image:
                self._base_image = None
                module_logger.warning("Property base_image is mutually exclusive with base_dockerfile. "
                                      "Reset base_image to None")
            if self._build_context:
                self._build_context = None
                module_logger.warning("Property build_context is mutually exclusive with base_dockerfile. "
                                      "Reset build_context to None")
        else:
            self._base_dockerfile = None

    @property
    def build_context(self):
        """Get or set Docker build context used for Docker-based runs."""
        return self._build_context

    @build_context.setter
    def build_context(self, value):
        if value:
            self._build_context = value
            if self._base_dockerfile:
                self._base_dockerfile = None
                module_logger.warning("Property base_dockerfile is mutually exclusive with base_image. "
                                      "Reset base_dockerfile to None")
            if self._base_image:
                self._base_image = None
                module_logger.warning("Property base_image is mutually exclusive with build_context. "
                                      "Reset base_image to None")
        else:
            self._build_context = None

    @property
    def gpu_support(self):
        """DEPRECATED. Azure automatically detects and uses the NVIDIA Docker extension when it is available."""
        module_logger.warning("'gpu_support' is no longer necessary; AzureML now automatically detects and uses "
                              "nvidia docker extension when it is available. It will be removed in a future release.")
        return True

    @gpu_support.setter
    def gpu_support(self, _):
        """DEPRECATED. Azure automatically detects and uses the NVIDIA Docker extension when it is available."""
        module_logger.warning("'gpu_support' is no longer necessary; AzureML now automatically detects and uses "
                              "nvidia docker extension when it is available. It will be removed in a future release.")

    @property
    def enabled(self):
        """DEPRECATED. Use the azureml.core.runconfig.DockerConfiguration class."""
        return self._enabled

    @enabled.setter
    def enabled(self, value=False):
        """DEPRECATED. Use the azureml.core.runconfig.DockerConfiguration class."""
        module_logger.warning("'enabled' is deprecated. Please use the azureml.core.runconfig.DockerConfiguration "
                              "object with the 'use_docker' param instead.")
        self._enabled = value

    @property
    def shared_volumes(self):
        """DEPRECATED. Use the azureml.core.runconfig.DockerConfiguration class."""
        return self._shared_volumes

    @shared_volumes.setter
    def shared_volumes(self, value=True):
        """DEPRECATED. Use the azureml.core.runconfig.DockerConfiguration class."""
        module_logger.warning("'shared_volumes' is deprecated. Please use the "
                              "azureml.core.runconfig.DockerConfiguration object instead.")
        self._shared_volumes = value

    @property
    def arguments(self):
        """DEPRECATED. Use the azureml.core.runconfig.DockerConfiguration class."""
        return self._arguments

    @arguments.setter
    def arguments(self, value=[]):
        """DEPRECATED. Use the azureml.core.runconfig.DockerConfiguration class."""
        module_logger.warning("'arguments' is deprecated. Please use the azureml.core.runconfig.DockerConfiguration "
                              "object instead.")
        self._arguments = value

    @property
    def shm_size(self):
        """DEPRECATED. Use the azureml.core.runconfig.DockerConfiguration class."""
        return self._shm_size

    @shm_size.setter
    def shm_size(self, value=None):
        """DEPRECATED. Use the azureml.core.runconfig.DockerConfiguration class."""
        module_logger.warning("'shm_size' is deprecated. Please use the azureml.core.runconfig.DockerConfiguration "
                              "object instead.")
        self._shm_size = value


class DockerEnvironment(DockerSection):
    """DEPRECATED. Use the :class:`azureml.core.runconfig.DockerConfiguration` class.

    .. remarks::

        This class is deprecated. Use the :class:`azureml.core.runconfig.DockerConfiguration` class.
    """

    def __init__(self):
        """Class DockerEnvironment constructor."""
        super(DockerEnvironment, self).__init__()
        module_logger.warning(
            "'DockerEnvironment' will be deprecated soon. Please use DockerConfiguration from "
            "'azureml.core.runconfig'.")


class SparkPackage(_AbstractRunConfigElement):
    """Defines a Spark dependency (package).

    :param group: The package group ID.
    :type group: str
    :param artifact: The package artifact ID.
    :type artifact: str
    :param version: The package version.
    :type version: str
    """

    # This is used to deserialize.
    # This is also the order for serialization into a file.
    _field_to_info_dict = collections.OrderedDict([
        ("group", _FieldInfo(str, "")),
        ("artifact", _FieldInfo(str, "")),
        ("version", _FieldInfo(str, ""))
    ])

    def __init__(self, group=None, artifact=None, version=None):
        """Class SparkPackage constructor."""
        super(SparkPackage, self).__init__()
        self.group = group
        self.artifact = artifact
        self.version = version
        self._initialized = True


class SparkSection(_AbstractRunConfigElement):
    """Defines Spark settings to use for the PySpark framework in the environment.

    This SparkSection class is used in the :class:`azureml.core.environment.Environment` class.

    :var repositories: A list of Spark repositories.
    :vartype repositories: builtin.list

    :var packages: The packages to use.
    :vartype packages: builtin.list

    :var precache_packages: Indicates Whether to precache the packages.
    :vartype precache_packages: bool
    """

    # This is used to deserialize.
    # This is also the order for serialization into a file.
    _field_to_info_dict = collections.OrderedDict([
        ("repositories", _FieldInfo(
            list, "List of spark repositories.", list_element_type=str)),
        ("packages", _FieldInfo(list, "The packages to use.",
                                list_element_type=SparkPackage)),
        ("precache_packages", _FieldInfo(bool, "Whether to precache the packages."))
    ])

    def __init__(self, **kwargs):
        """Class SparkSection constructor."""
        super(SparkSection, self).__init__()
        self.repositories = []
        self.packages = []
        self.precache_packages = True
        self._initialized = True


class SparkEnvironment(SparkSection):
    """DEPRECATED. Use the :class:`azureml.core.environment.SparkSection` class.

    .. remarks::

        This class is deprecated. Use the :class:`azureml.core.environment.SparkSection` class.
    """

    def __init__(self):
        """Class SparkEnvironment constructor."""
        super(SparkEnvironment, self).__init__()
        module_logger.warning(
            "'SparkEnvironment' will be deprecated soon. Please use SparkSection from 'azureml.core.environment'.")


class ImageBuildDetails(object):
    """Environment image build class.

    ImageBuildDetails class provides details about environment image build status.
    """

    def __init__(self, environment_client, location):
        """Class ImageBuildDetails constructor."""
        self.environment_client = environment_client
        self.log_url = None
        self._location = location

    @property
    def status(self):
        """Return image build status."""
        return self.environment_client._get_lro_response(self._location)["status"]

    def get_log(self):
        """Return image build log."""
        if not self.log_url:
            lro_response = self.environment_client._get_lro_response(self._location)
            self.log_url = lro_response.get("logUrl")

        if self.log_url:
            return requests.get(self.log_url, stream=True).text

    def wait_for_completion(self, show_output=True):
        """
        Wait for the completion of this cloud environment build.

        Returns the image build object after the wait.

        :param show_output: show_output=True shows the build status on sys.stdout.
        :type show_output: bool

        :return: The Image build object (self).
        :rtype: azureml.core.environment.ImageBuildDetails
        """
        def _incremental_print(log, printed, fileout):
            count = 0
            if log:
                for line in log.splitlines():
                    if count >= printed:
                        print(line, file=fileout)
                        printed += 1
                    count += 1

            return printed

        timeout_seconds = sys.maxsize
        file_handle = sys.stdout

        status = self.status
        last_status = None
        separator = ''
        time_run = 0
        sleep_period = 5
        printed = 0
        while status in ('Running', "GettingOrProvisioningACR", "Queued"):
            if time_run + sleep_period > timeout_seconds:
                if show_output:
                    print('Timed out of waiting, %sStatus: %s.' %
                          (separator, status), flush=True)
                break
            time_run += sleep_period
            time.sleep(sleep_period)
            status = self.status
            if last_status != status:
                if show_output:
                    print('Image Build Status: {0}\n'.format(status))
                last_status = status
                separator = ''
            else:
                if show_output:
                    content = self.get_log()
                    printed = _incremental_print(content, printed, file_handle)
                else:
                    print('.', end='', flush=True)
                separator = '\n'

        return self


class DockerImageDetails(dict):
    """AzureML docker image details class."""

    def __init__(self, *args, **kwargs):
        """Class constructor."""
        super().__init__(*args, **kwargs)

    @property
    def image_exist(self):
        """Return True if image exist."""
        return bool(self.get("imageExistsInRegistry"))

    @property
    def repository(self):
        """Repository name."""
        return self.get("dockerImage").get("name")

    @property
    def registry(self):
        """Container registry address."""
        return self.get("dockerImage").get("registry").get("address")

    @property
    def image(self):
        """Fully qualified image name."""
        return self.repository if self.registry is None else "/".join([self.registry, self.repository])

    @property
    def interpreter_path(self):
        """Interpreter path."""
        return self.get("pythonEnvironment").get("interpreterPath")

    @property
    def environment_path(self):
        """Path to the conda environment."""
        return self.get("pythonEnvironment").get("condaEnvironmentPath")

    @property
    def dockerfile(self):
        """Dockerfile content."""
        return self.get("ingredients").get("dockerfile") if self.get("ingredients", None) else None

    @property
    def _safe_dict(self):
        safe_copy = dict(self)
        username = safe_copy.get("dockerImage", {}).get("registry", {}).get("username", {})
        password = safe_copy.get("dockerImage", {}).get("registry", {}).get("password", {})
        if (username):
            safe_copy["dockerImage"]["registry"]["username"] = username[0:3] + "***"
        if (password):
            safe_copy["dockerImage"]["registry"]["password"] = "***"
        return safe_copy

    def __repr__(self):
        """Return str(self)."""
        return self._safe_dict.__repr__()

    def __str__(self):
        """Return repr(self)."""
        return self._safe_dict.__str__()


class RCranPackage(_AbstractRunConfigElement):
    """Defines the CRAN packages to be installed.

    :param name: Name of the package.
    :type name: str

    :param version: Version of the package.
    :type version: str

    :param repository: The repository name from where the package would be installed.
    :type repository: str
    """

    # This is used to deserialize.
    # This is also the order for serialization into a file.
    _field_to_info_dict = collections.OrderedDict([
        ("name", _FieldInfo(str, "Package name")),
        ("version", _FieldInfo(str, "Package version")),
        ("repository", _FieldInfo(str, "Package repository"))
    ])

    def __init__(self):
        """Class RCranPackage constructor."""
        super(RCranPackage, self).__init__()
        self.name = None
        self.version = None
        self.repository = None
        self._initialized = True


class RGitHubPackage(_AbstractRunConfigElement):
    """Defines the Github packages to be installed.

    :param repository: Repository address in the format username/repo[/subdir][@ref|#pull].
    :type repository: str

    :param auth_token: Personal access token to install from a private repo.
    :type auth_token: str
    """

    # This is used to deserialize.
    # This is also the order for serialization into a file.
    _field_to_info_dict = collections.OrderedDict([
        ("repository", _FieldInfo(str, "Repository address in the format username/repo[/subdir][@ref|#pull]")),
        ("auth_token", _FieldInfo(str, "Personal access token to install from a private repo"))
    ])

    def __init__(self):
        """Class RGitHubPackage constructor."""
        super(RGitHubPackage, self).__init__()
        self.repository = None
        self.auth_token = None
        self._initialized = True


class RSection(_AbstractRunConfigElement):
    """Defines the R environment to use on a target compute for a run.

    This class is used in the :class :`azureml.core.Environment` class .

    :var r_version: The version of R to be installed.
    :vartype r_version: str

    :var user_managed: Indicates whether the environment is managed by user or by AzureML.
    :vartype user_managed: bool

    :var rscript_path: The Rscript path to use if an environment build is not required.
        The path specified gets used to call the user script.
    :vartype rscript_path: str

    :var snapshot_date: Date of MRAN snapshot to use in YYYY-MM-DD format, e.g. "2019-04-17".
    :vartype snapshot_date: str

    :var cran_packages: The CRAN packages to use.
    :vartype cran_packages: builtin.list

    :var github_packages: The packages directly from GitHub.
    :vartype github_packages: builtin.list

    :var custom_url_packages: The packages from custom urls.
    :vartype custom_url_packages: builtin.list

    :var bioconductor_packages: The packages from Bioconductor.
    :vartype bioconductor_packages: builtin.list
    """

    # This is used to deserialize.
    # This is also the order for serialization into a file.
    _field_to_info_dict = collections.OrderedDict([
        ("r_version", _FieldInfo(str, "The version of R to be installed")),
        ("user_managed", _FieldInfo(bool, "Indicates whether the environment is managed by user or by AzureML.")),
        ("rscript_path", _FieldInfo(str, "The Rscript path to use if an environment build is not required")),
        ("snapshot_date", _FieldInfo(str, "Date of MRAN snapshot to use")),
        ("cran_packages", _FieldInfo(list, "The CRAN packages to use", list_element_type=RCranPackage)),
        ("github_packages", _FieldInfo(list, "The packages directly from GitHub", list_element_type=RGitHubPackage)),
        ("custom_url_packages", _FieldInfo(list, "The packages from custom urls", list_element_type=str)),
        ("bioconductor_packages", _FieldInfo(list, "The packages from Bioconductor", list_element_type=str)),
    ])

    def __init__(self):
        """Class RSection constructor."""
        super(RSection, self).__init__()
        self.r_version = None
        self.user_managed = False
        self.rscript_path = None
        self.snapshot_date = None
        self.cran_packages = None
        self.github_packages = None
        self.custom_url_packages = None
        self.bioconductor_packages = None
        self._initialized = True


class Environment(_AbstractRunConfigElement):
    r"""Configures a reproducible Python environment for machine learning experiments.

    An Environment defines Python packages, environment variables, and Docker settings that are used in
    machine learning experiments, including in data preparation, training, and deployment to a web service.
    An Environment is managed and versioned in an Azure Machine Learning :class:`azureml.core.workspace.Workspace`.
    You can update an existing environment and retrieve a version to reuse. Environments are exclusive to the
    workspace they are created in and can't be used across different workspaces.

    For more information about environments, see `Create and manage reusable
    environments  <https://docs.microsoft.com/azure/machine-learning/how-to-use-environments>`_.

    .. remarks::

        Azure Machine Learning provides curated environments, which are predefined environments that offer
        good starting points for building your own environments. Curated environments are backed by cached Docker
        images, providing a reduced run preparation cost. For more information about curated environments, see
        `Create and manage reusable
        environments <https://docs.microsoft.com/azure/machine-learning/how-to-use-environments>`_.

        There are a number of ways environment are created in the Azure Machine Learning, including when
        you:

        * Initialize a new Environment object.

        * Use one of the Environment class methods: :meth:`from_conda_specification`,
          :meth:`from_pip_requirements`, or :meth:`from_existing_conda_environment`.

        * Use the :meth:`azureml.core.experiment.Experiment.submit` method of the Experiment class to
          submit an experiment run without specifying an environment, including with an
          :class:`azureml.train.estimator.Estimator` object.

        The following example shows how to instantiate a new environment.

        .. code-block:: python

            from azureml.core import Environment
            myenv = Environment(name="myenv")

        You can manage an environment by registering it. Doing so allows you to track the environment's
        versions, and reuse them in future runs.

        .. code-block:: python

            myenv.register(workspace=ws)

        For more samples of working with environments, see the Jupyter Notebook `Using
        environments <https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/
        training/using-environments/using-environments.ipynb>`_.

    :param name: The name of the environment.

        .. note::

            Do not start your environment name with "Microsoft" or "AzureML". \
            The prefixes "Microsoft" and "AzureML" are reserved for curated environments. \
            For more information about curated environments, see `Create and manage reusable \
            environments <https://docs.microsoft.com/azure/machine-learning/how-to-use-environments>`__.

    :type name: string

    :var Environment.databricks: The section configures azureml.core.databricks.DatabricksSection library dependencies.
    :vartype databricks: azureml.core.databricks.DatabricksSection

    :var docker: This section configures settings related to the final Docker image built to the specifications
        of the environment and whether to use Docker containers to build the environment.
    :vartype docker: azureml.core.environment.DockerSection

    :var inferencing_stack_version: This section specifies the inferencing stack version added to the image.
        To avoid adding an inferencing stack, do not set this value. Valid value: "latest".
    :vartype inferencing_stack_version: string

    :var python: This section specifies which Python environment and interpreter to use on the target compute.
    :vartype python: azureml.core.environment.PythonSection

    :var spark: The section configures Spark settings. It is only used when framework is set to PySpark.
    :vartype spark: azureml.core.environment.SparkSection

    :var r: This section specifies which R environment to use on the target compute.
    :vartype r: azureml.core.environment.RSection

    :var version: The version of the environment.
    :vartype version: string

    :var asset_id: Asset ID. Populates when an environment is registered.
    :vartype asset_id: string
    """

    # This is used to deserialize.
    # This is also the order for serialization into a file.
    _field_to_info_dict = collections.OrderedDict([
        # In dict, values are assumed to be str
        ("name", _FieldInfo(str, "Environment name")),
        ("version", _FieldInfo(str, "Environment version")),
        ("_environment_variables", _FieldInfo(
            dict, "Environment variables set for the run.", serialized_name="environment_variables", user_keys=True)),
        ("python", _FieldInfo(PythonSection, "Python details")),
        ("docker", _FieldInfo(DockerSection, "Docker details")),
        ("spark", _FieldInfo(SparkSection, "Spark details")),
        ("databricks", _FieldInfo(DatabricksSection, "Databricks details")),
        ("r", _FieldInfo(RSection, "R details")),
        ("inferencing_stack_version", _FieldInfo(str, "Inferencing stack version")),
        ("asset_id", _FieldInfo(str, "Asset ID"))
    ])

    def __init__(self, name, **kwargs):
        """Class Environment constructor."""
        super(Environment, self).__init__()

        options = {"_skip_defaults": False}
        options.update(kwargs)
        _skip_defaults = options["_skip_defaults"]

        # Add Name/version validation for env management
        if name is not None and not isinstance(name, str):
            raise TypeError('Environment name expected str, {} found'.format(type(name)))
        self.name = name
        self.version = None
        self.python = PythonSection(_skip_defaults=_skip_defaults)
        self.docker = DockerSection(_skip_defaults=_skip_defaults)
        self.spark = SparkSection(_skip_defaults=_skip_defaults)
        self.databricks = DatabricksSection(_skip_defaults=_skip_defaults)
        self._environment_variables = dict()
        self.inferencing_stack_version = None
        self.r = None
        self.asset_id = None

        if not _skip_defaults:
            self._environment_variables = {"EXAMPLE_ENV_VAR": "EXAMPLE_VALUE"}

        self._initialized = True

    @property
    def environment_variables(self):
        """DEPRECATED: Use azureml.core.RunConfiguration object to set runtime variables."""
        return self._environment_variables

    @environment_variables.setter
    def environment_variables(self, value):
        # prevent devs from using deprecated properties
        if SDKVERSION.startswith("0."):
            raise UserErrorException("Runtime variables should be set in RunConfiguration")

        module_logger.warning("Property environment_variables is deprecated. "
                              "Use RunConfiguration.environment_variables to set runtime variables.")
        self._environment_variables = value

    def _get_base_info_dict(self):
        """Return base info dictionary.

        :return:
        :rtype: OrderedDict
        """
        return OrderedDict([
            ('Name', self.name),
            ('Version', self.version)
        ])

    def __str__(self):
        """Format Environment data into a string.

        :return:
        :rtype: str
        """
        info = self._get_base_info_dict()
        formatted_info = ',\n'.join(
            ["{}: {}".format(k, v) for k, v in info.items()])
        return "Environment({0})".format(formatted_info)

    def __repr__(self):
        """Representation of the object.

        :return: Return the string form of the Environment object
        :rtype: str
        """
        environment_dict = Environment._serialize_to_dict(self)
        return json.dumps(environment_dict, indent=4, sort_keys=True)

    def register(self, workspace):
        """Register the environment object in your workspace.

        :param workspace: The workspace
        :type workspace: azureml.core.workspace.Workspace
        :param name:
        :type name: str
        :return: Returns the environment object
        :rtype: azureml.core.environment.Environment
        """
        if self.version is not None:
            module_logger.warning("Environment version is set. Attempting to register desired version. "
                                  "To auto-version, reset version to None.")

        environment_client = EnvironmentClient(workspace.service_context)
        environment_dict = Environment._serialize_to_dict(self)
        response = environment_client._register_environment_definition(environment_dict)
        env = Environment._deserialize_and_add_to_object(response)

        return env

    @staticmethod
    def label(workspace, name, version, labels):
        """Label environment object in your workspace with the specified values.

        :param workspace: The workspace
        :type workspace: azureml.core.workspace.Workspace
        :param name: Environment name
        :type name: str
        :param version: Environment version
        :type version: str
        :param labels: Values to label Environment with
        :type labels: builtin.list[str]
        """
        environment_client = EnvironmentClient(workspace.service_context)
        environment_client._set_envionment_definition_labels(name, version, labels)

    def clone(self, new_name):
        """Clone the environment object.

        Returns a new instance of environment object with a new name.

        :param new_name: New environment name
        :type new_name: str
        :return: New environment object
        :rtype: azureml.core.environment.Environment
        """
        if not new_name:
            raise UserErrorException("Name is required to instantiate an Environment object")
        env = Environment._deserialize_and_add_to_object(Environment._serialize_to_dict(self))
        env.name = new_name
        env.version = None
        env.asset_id = None

        return env

    @staticmethod
    def get(workspace, name, version=None, label=None):
        """Return the environment object.

        If label is specified, the object previously labeled with the value will be returned.
        Only one of version or label parameters can be specified. If both are missed,
        the latest version of the Environment object will be returned.

        :param workspace: The workspace that contains the environment.
        :type workspace: azureml.core.workspace.Workspace
        :param name: The name of the environment to return.
        :type name: str
        :param version: The version of the environment to return.
        :type version: str
        :param label: Environment label value.
        :type label: str
        :return: The environment object.
        :rtype: azureml.core.environment.Environment
        """
        if label and version:
            raise UserErrorException("Only one of version or label can be specified")
        environment_client = EnvironmentClient(workspace.service_context)
        if label:
            environment_dict = environment_client._get_environment_definition_by_label(name=name, label=label)
        else:
            environment_dict = environment_client._get_environment_definition(name=name, version=version)
        env = Environment._deserialize_and_add_to_object(environment_dict)

        return env

    @staticmethod
    def list(workspace):
        """Return a dictionary containing environments in the workspace.

        :param workspace: The workspace from which to list environments.
        :type workspace: azureml.core.workspace.Workspace
        :return: A dictionary of environment objects.
        :rtype: builtin.dict[str, azureml.core.Environment]
        """
        environment_client = EnvironmentClient(workspace.service_context)
        environment_list = environment_client._list_definitions()

        result = {environment["name"]: Environment._deserialize_and_add_to_object(environment)
                  for environment in environment_list}

        return result

    @staticmethod
    def _from_dependencies(name, conda_specification=None, pip_requirements=None):
        """Create environment object from python dependenies.

        :param name: The environment name.
        :type name: str
        :param conda_specification: conda specification file.
        :type conda_specification: str
        :param pip_requirements: pip requirements file.
        :type pip_requirements: str
        :return: The environment object.
        :rtype: azureml.core.environment.Environment
        """
        if conda_specification and pip_requirements:
            raise UserErrorException("Only one of conda_specification or pip_requirements can be specified \
            as python dependencies")

        if conda_specification:
            return Environment.from_conda_specification(name, conda_specification)
        elif pip_requirements:
            return Environment.from_pip_requirements(name, pip_requirements)
        else:
            env = Environment(name)
            env.python.user_managed_dependencies = True
            return env

    @staticmethod
    def from_dockerfile(name, dockerfile, conda_specification=None, pip_requirements=None):
        """Create environment object from a Dockerfile with optional python dependenies.

        Python layer will be added to the environment if conda_specification or pip_requirements is specified.
        conda_specification and pip_requirements are mutually exclusive.

        :param name: The environment name.
        :type name: str
        :param dockerfile: Dockerfile content or path to the file.
        :type dockerfile: str
        :param conda_specification: conda specification file.
        :type conda_specification: str
        :param pip_requirements: pip requirements file.
        :type pip_requirements: str
        :return: The environment object.
        :rtype: azureml.core.environment.Environment
        """
        env = Environment._from_dependencies(name, conda_specification, pip_requirements)

        # set exclusive properties to None to suppress warning
        env.docker.base_image = None
        env.docker.build_context = None
        env.docker.base_dockerfile = dockerfile

        return env

    @staticmethod
    def from_docker_image(name, image, container_registry=None, conda_specification=None, pip_requirements=None):
        """Create environment object from a base docker image with optional python dependenies.

        Python layer will be added to the environment if conda_specification or pip_requirements is specified.
        conda_specification and pip_requirements are mutually exclusive.

        .. remarks::

            If base image is from private repository that requires authorization, and authorization is not set on the
            AzureML workspace level, `container_registry` is required

        :param name: The environment name.
        :type name: str
        :param image: fully qualified image name.
        :type image: str
        :param conda_specification: conda specification file.
        :type conda_specification: str
        :param container_registry: private container repository details.
        :type container_registry: azureml.core.container_registry.ContainerRegistry
        :param pip_requirements: pip requirements file.
        :type pip_requirements: str
        :return: The environment object.
        :rtype: azureml.core.environment.Environment
        """
        env = Environment._from_dependencies(name, conda_specification, pip_requirements)

        # set exclusive properties to None to suppress warnings
        env.docker.base_dockerfile = None
        env.docker.build_context = None

        env.docker.base_image = image
        if container_registry:
            env.docker.base_image_registry = container_registry
        return env

    @staticmethod
    def from_docker_build_context(name, docker_build_context):
        """Create environment object from a Docker build context.

        :param name: The environment name.
        :type name: str

        :param docker_build_context: The DockerBuildContext object.
        :type docker_build_context: azureml.core.environment.DockerBuildContext

        :return: The environment object.
        :rtype: azureml.core.environment.Environment
        """
        env = Environment(name, _skip_defaults=True)
        env.docker.build_context = docker_build_context

        return env

    @staticmethod
    def from_conda_specification(name, file_path):
        """Create environment object from an environment specification YAML file.

        To get an environment specification YAML file, see `Managing environments
        <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#sharing-an-environment>`_
        in the conda user guide.

        :param name: The environment name.
        :type name: str
        :param file_path: The conda environment specification YAML file path.
        :type file_path: str
        :return: The environment object.
        :rtype: azureml.core.environment.Environment
        """  # noqa: E501
        conda_dependencies = CondaDependencies(conda_dependencies_file_path=file_path)
        if not conda_dependencies._python_version:
            module_logger.warning('No Python version provided, defaulting to "{}"'.format(PYTHON_DEFAULT_VERSION))
            conda_dependencies.set_python_version(PYTHON_DEFAULT_VERSION)
        env = Environment(name=name)
        env.python.conda_dependencies = conda_dependencies

        return env

    @staticmethod
    def from_existing_conda_environment(name, conda_environment_name):
        """Create an environment object created from a locally existing conda environment.

        To get a list of existing conda environments, run ``conda env list``. For more information, see
        `Managing environments
        <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`__ in
        the conda user guide.

        :param name: The environment name.
        :type name: str
        :param conda_environment_name: The name of a locally existing conda environment.
        :type conda_environment_name: str
        :return: The environment object or None if exporting the conda specification file fails.
        :rtype: azureml.core.environment.Environment
        """  # noqa: E501
        env = None
        if Environment._check_conda_installation() and Environment._check_conda_env_existance(conda_environment_name):
            try:
                print("Exporting conda specifications for "
                      + "existing conda environment: {}".format(conda_environment_name))
                export_existing_env_cmd = ["conda", "env", "export", "--no-builds", "--name", conda_environment_name]
                conda_specifications = subprocess.check_output(export_existing_env_cmd)

                tmp_conda_spec_file = tempfile.NamedTemporaryFile(suffix=".yml", delete=False)
                tmp_conda_spec_file.write(conda_specifications)
                tmp_conda_spec_file.seek(0)
                tmp_conda_spec_file_name = tmp_conda_spec_file.name

                env = Environment.from_conda_specification(name, tmp_conda_spec_file_name)

                tmp_conda_spec_file.close()
            except subprocess.CalledProcessError as ex:
                print("Exporting conda specifications failed with exit code: {}".format(ex.returncode))
            finally:
                if os.path.isfile(tmp_conda_spec_file_name):
                    os.remove(tmp_conda_spec_file_name)

        return env

    @staticmethod
    def from_pip_requirements(name, file_path, pip_version=None):
        """Create an environment object created from a pip requirements file.

            .. remarks::

                Unpinned pip dependency will be added if `pip_version` is not specified.

        :param name: The environment name.
        :type name: str
        :param file_path: The pip requirements file path.
        :type file_path: str
        :param pip_version: Pip version for conda environment.
        :type pip_version: str
        :return: The environment object.
        :rtype: azureml.core.environment.Environment
        """
        env = Environment(name=name)
        with open(file_path) as req_file:
            env.python.conda_dependencies.set_pip_requirements(req_file.read().splitlines())

        if pip_version:
            env.python.conda_dependencies.add_conda_package("pip={}".format(pip_version))
        else:
            env.python.conda_dependencies.add_conda_package("pip")

        return env

    @staticmethod
    def add_private_pip_wheel(workspace, file_path, exist_ok=False):
        """Upload the private pip wheel file on disk to the Azure storage blob attached to the workspace.

        Throws an exception if a private pip wheel with the same name already exists in the workspace storage blob.

        :param workspace: The workspace object to use to register the private pip wheel.
        :type workspace: azureml.core.workspace.Workspace
        :param file_path: Path to the local pip wheel on disk, including the file extension.
        :type file_path: str
        :param exist_ok: Indicates whether to throw an exception if the wheel already exists.
        :type exist_ok: bool
        :return: Returns the full URI to the uploaded pip wheel on Azure blob storage to use in conda dependencies.
        :rtype: str
        """
        if not os.path.isfile(file_path):
            raise UserErrorException("Please make sure the wheel file exists at: {}".format(file_path))

        # since we're guaranteeing that the file_path points to a file, os.path.basename should return the file name.
        wheel_name = os.path.basename(file_path)

        try:
            from azureml._restclient.artifacts_client import ArtifactsClient
            artifacts_client = ArtifactsClient(workspace.service_context)

            batch_artifact_info = artifacts_client.create_empty_artifacts(
                origin=_EMS_ORIGIN_NAME,
                container=_PRIVATE_PKG_CONTAINER_NAME,
                paths=[wheel_name])
            wheel_artifact = batch_artifact_info.artifacts[wheel_name]
            wheel_artifact_content_info = batch_artifact_info.artifact_content_information[wheel_name]

            with open(file_path, "rb") as stream:
                artifacts_client.upload_stream_to_existing_artifact(
                    stream=stream,
                    artifact=wheel_artifact,
                    content_information=wheel_artifact_content_info)

            content_uri = wheel_artifact_content_info.content_uri
        except AzureMLException as error:
            if "Resource Conflict: ArtifactId" in repr(error):
                if exist_ok:
                    # wheel already exists so we'll just fetch the existing artifact to get the URI
                    content_uri = artifacts_client.get_file_uri(
                        origin=_EMS_ORIGIN_NAME,
                        container=_PRIVATE_PKG_CONTAINER_NAME,
                        path=wheel_name)
                else:
                    error_msg = """
A wheel with the name {} already exists.
Please make sure {} is the private wheel you would like to add to the workspace.
If you would like to have it be a no-op instead of throwing an exception for duplicate entries,
please set exist_ok=True""".format(wheel_name, wheel_name)
                    raise UserErrorException(error_msg)
            else:
                raise error

        parsed = urllib.parse.urlparse(content_uri)
        normalized_uri = parsed.scheme + "://" + parsed.netloc + parsed.path
        return normalized_uri

    def get_image_details(self, workspace):
        """Return the Image details.

        :param workspace: The workspace.
        :type workspace: azureml.core.workspace.Workspace
        :return: Returns the image details as dict
        :rtype: azureml.core.environment.DockerImageDetails
        """
        environment_client = EnvironmentClient(workspace.service_context)
        image_details_dict = environment_client._get_image_details(
            name=self.name, version=self.version)

        return DockerImageDetails(image_details_dict)

    def build(self, workspace, image_build_compute=None):
        """Build a Docker image for this environment in the cloud.

        :param workspace: The workspace and its associated Azure Container Registry where the image is stored.
        :type workspace: azureml.core.workspace.Workspace
        :param image_build_compute: The compute name where the image build will take place
        :type image_build_compute: str
        :return: Returns the image build details object.
        :rtype: azureml.core.environment.ImageBuildDetails
        """
        if self.version is None:
            module_logger.warning("Building a non-registered environment is not supported. "
                                  "Registering environment.")

        # Register environment. If instance matches - no-op, otherwise conflict should occur
        registered_environment = self.register(workspace)

        environment_client = EnvironmentClient(workspace.service_context)
        try:
            lro = environment_client._start_cloud_image_build(
                name=registered_environment.name,
                version=registered_environment.version,
                image_build_compute=image_build_compute)
        except Exception as error:
            # Temporary solution to filter attempts to build unregistered environments out
            if "Code: 404" in repr(error):
                raise UserErrorException("Environment '{}' is not registered".format(self.name))
            else:
                raise error

        location = lro["location"]
        return ImageBuildDetails(environment_client, location)

    def build_local(self, workspace, platform=None, **kwargs):
        """Build the local Docker or conda environment.

        .. remarks::

            The following examples show how to build a local environment. Please make sure `workspace`
            is instantiated as a valid azureml.core.workspace.Workspace object

            Build local conda environment

            .. code-block:: python

                from azureml.core import Environment
                myenv = Environment(name="myenv")
                registered_env = myenv.register(workspace)
                registered_env.build_local(workspace)

            Build local docker environment

            .. code-block:: python

                from azureml.core import Environment
                myenv = Environment(name="myenv")
                registered_env = myenv.register(workspace)
                registered_env.build_local(workspace, useDocker=True)

            Build docker image locally and optionally push it to the container registry associated with
            the workspace

            .. code-block:: python

                from azureml.core import Environment
                myenv = Environment(name="myenv")
                registered_env = myenv.register(workspace)
                registered_env.build_local(workspace, useDocker=True, pushImageToWorkspaceAcr=True)

        :param workspace: The workspace.
        :type workspace: azureml.core.workspace.Workspace
        :param platform: Platform. One of `Linux`, `Windows` or `OSX`. Current platform will be used by default.
        :type platform: str
        :param kwargs: Advanced keyword arguments
        :type kwargs: dict
        :return: Streams the on-going Docker or conda built output to the console.
        :rtype: str
        """
        environment_client = EnvironmentClient(workspace.service_context)

        if platform is None:
            if os.name == 'nt':
                platform = "Windows"
            else:
                platform = "Linux"

        # overwrite : add additional kwargs
        payload = dict({"platform": platform, "useDocker": self.docker.enabled}, **kwargs)

        try:
            recipe = environment_client._get_recipe_for_build(name=self.name, version=self.version, **payload)

        except Exception as error:
            # Temporary solution to filter attempts to build unregistered environments out
            if "Code: 404" in repr(error):
                raise UserErrorException("Environment '{}' is not registered".format(self.name))
            else:
                raise error

        setup_content = base64.b64decode(recipe["setupContentZip"])
        build_script = recipe["buildEnvironmentScriptName"]
        check_script = recipe["checkEnvironmentScriptName"]
        environment_variables = recipe["environmentVariables"]

        with tempfile.TemporaryDirectory() as setup_temp_dir:
            try:
                setup_content_zip_file = os.path.join(setup_temp_dir, "Setupcontent.zip")
                print("Saving setup content into ", setup_temp_dir)
                with open(setup_content_zip_file, "wb") as file:
                    file.write(setup_content)

                with zipfile.ZipFile(setup_content_zip_file, 'r') as zip_ref:
                    zip_ref.extractall(setup_temp_dir)

                self._build_environment(setup_temp_dir, check_script, build_script, environment_variables)
            except subprocess.CalledProcessError as ex:
                raise Exception("Building environment failed with exit code: {}".format(ex.returncode))

    def save_to_directory(self, path, overwrite=False):
        """Save an environment definition to a directory in an easily editable format.

        :param path: Path to the destination directory.
        :type path: str
        :param overwrite: If an existing directory should be overwritten. Defaults false.
        :type overwrite: bool
        """
        os.makedirs(path, exist_ok=overwrite)
        if self.python and self.python.conda_dependencies:
            self.python.conda_dependencies.save_to_file(path, _CONDA_DEPENDENCIES_FILE_NAME)

        if self.docker and self.docker.base_dockerfile:
            with open(os.path.join(path, _BASE_DOCKERFILE_FILE_NAME), "w") as base_dockerfile:
                base_dockerfile.write(self.docker.base_dockerfile)

        with open(os.path.join(path, _DEFINITION_FILE_NAME), "w") as definition:
            environment_dict = Environment._serialize_to_dict(self)

            # Don't serialize properties that got saved to separate files.
            if environment_dict.get("python") and environment_dict["python"].get("condaDependencies"):
                del environment_dict["python"]["condaDependencies"]
            if environment_dict.get("docker") and environment_dict["docker"].get("baseDockerfile"):
                del environment_dict["docker"]["baseDockerfile"]

            json.dump(environment_dict, definition, indent=4)

    @staticmethod
    def load_from_directory(path):
        """Load an environment definition from the files in a directory.

        :param path: Path to the source directory.
        :type path: str
        """
        definition_path = os.path.join(path, _DEFINITION_FILE_NAME)
        if not os.path.isfile(definition_path):
            raise FileNotFoundError(definition_path)

        with open(definition_path, "r") as definition:
            environment_dict = json.load(definition)
        env = Environment._deserialize_and_add_to_object(environment_dict)

        conda_file_path = os.path.join(path, _CONDA_DEPENDENCIES_FILE_NAME)
        if os.path.isfile(conda_file_path):
            env.python = env.python or PythonSection(_skip_defaults=True)
            env.python.conda_dependencies = CondaDependencies(conda_file_path)

        base_dockerfile_path = os.path.join(path, _BASE_DOCKERFILE_FILE_NAME)
        if os.path.isfile(base_dockerfile_path):
            with open(base_dockerfile_path, "r") as base_dockerfile:
                env.docker = env.docker or DockerSection(_skip_defaults=True)
                env.docker.base_dockerfile = base_dockerfile.read()

        return env

    @staticmethod
    def _check_conda_installation():
        is_conda_installed = True
        try:
            conda_version_cmd = ["conda", "--version"]
            conda_version = subprocess.check_output(conda_version_cmd).decode("UTF-8").replace("conda", "").strip()

            from distutils.version import LooseVersion
            installed_version = LooseVersion(conda_version)
            required_version = LooseVersion("4.4.0")
            if installed_version < required_version:
                print("AzureML requires Conda version {} or later.".format(required_version))
                print("You can update your installed conda version {} by running:".format(installed_version))
                print("    conda update conda")
                is_conda_installed = False
        except subprocess.CalledProcessError:
            print("Unable to run Conda package manager. Please make sure Conda (>= 4.4.0) is installed.")
            is_conda_installed = False

        return is_conda_installed

    @staticmethod
    def _check_conda_env_existance(conda_env_name):
        # Helper method to check if the conda environment exists as conda env export works for envs that don't exist.
        env_exists = False
        try:
            get_conda_env_list_cmd = ["conda", "env", "list", "--json"]
            conda_env_list_json = subprocess.check_output(get_conda_env_list_cmd)
            conda_env_list = json.loads(conda_env_list_json.decode("utf-8")).get("envs", {})
            # NOTE: conda env list --json returns the full path for the env
            if any(conda_env_name == os.path.basename(os.path.normpath(conda_env)) for conda_env in conda_env_list):
                env_exists = True
            else:
                print("Could not find existing conda environment named: {}".format(conda_env_name))
                print("Please make sure the conda environment {} exists.".format(conda_env_name))
        except subprocess.CalledProcessError as ex:
            print("Getting the list of existing conda environments failed with exit code: {}".format(ex.returncode))

        return env_exists

    @staticmethod
    def _discriminate_variations(raw_object):
        keys = [x[0] for x in raw_object.items()]

        reference_keys = ["name", "version"]
        if all(x in reference_keys for x in keys):
            return EnvironmentReference

        return Environment

    @staticmethod
    def _serialize_to_dict(environment, use_commented_map=False):
        environment_dict = _serialize_to_dict(environment, use_commented_map)

        # _serialization_utils._serialize_to_dict does not serialize condadependencies correctly.
        # Hence the work around to copy this in to the env object

        if environment is not None and not isinstance(environment, EnvironmentReference):
            if environment.python.conda_dependencies is not None:
                inline = environment.python.conda_dependencies._conda_dependencies
                environment_dict["python"]["condaDependencies"] = inline

        return environment_dict

    @staticmethod
    def _deserialize_and_add_to_object(serialized_dict):
        environment_name = serialized_dict.get("name")
        environment_version = serialized_dict.get("version")

        if Environment._discriminate_variations(serialized_dict) == EnvironmentReference:
            return EnvironmentReference(environment_name, environment_version)

        # _serialization_utils._deserialize_and_add_to_object does deserialize condaDependencies correctly.
        # Hence the work around to inject it to env object
        environment_object = Environment(environment_name, _skip_defaults=True)
        environment_object.version = environment_version

        env = _deserialize_and_add_to_object(Environment, serialized_dict, environment_object)

        inline_conda_dependencies = serialized_dict.get("python", {}).get("condaDependencies", None)
        if inline_conda_dependencies is not None:
            conda_dependencies = CondaDependencies(_underlying_structure=inline_conda_dependencies)
            env.python.conda_dependencies = conda_dependencies

        return env

    @staticmethod
    def _get_files_to_upload(path, max_files, max_bytes):
        # ensure directory exists
        if not os.path.isdir(path):
            raise UserErrorException("{} is not a valid directory.".format(path))

        # walk directory tree, counting files and summing their sizes
        results = {}
        files_counter = bytes_counter = 0
        for pathl, _, files in os.walk(path):
            for _file in files:
                fpath = os.path.join(pathl, _file)
                files_counter += 1
                bytes_counter += os.stat(fpath).st_size
                results[_file] = fpath
        if files_counter > max_files or bytes_counter > max_bytes:
            module_logger.warning("The directory {} may be too large to use as a Docker build "
                                  "context, which supports a maximum of {} files or {} bytes."
                                  .format(path, max_files, max_bytes))

        return results

    @staticmethod
    def _hash_files(file_names):
        # See https://stackoverflow.com/questions/22058048/hashing-a-file-in-python
        hash = hashlib.sha256()
        for file_name in file_names:
            with open(file_name, 'rb') as f:
                while True:
                    data = f.read(_DIR_HASH_BUFFER_SIZE)
                    if not data:
                        break
                    hash.update(data)

        return hash.hexdigest()[:32]

    def _build_environment(self, setup_temp_dir, check_script, build_script, environment_variables=None):
        check_env_command = self._get_command(setup_temp_dir, check_script)
        build_env_command = self._get_command(setup_temp_dir, build_script)

        env = os.environ
        if environment_variables:
            env.update(environment_variables)

        try:
            # Check if environment exists
            subprocess.check_call(check_env_command, stdout=sys.stdout,
                                  stderr=sys.stderr, cwd=setup_temp_dir, env=env)
        except subprocess.CalledProcessError:
            # else build Environment
            subprocess.check_call(build_env_command, stdout=sys.stdout,
                                  stderr=sys.stderr, cwd=setup_temp_dir, env=env)

    def _get_command(self, setup_temp_dir, script):
        script_path = os.path.join(setup_temp_dir, script)

        if os.name == "nt":
            command = ["cmd.exe", "/c", script_path]
        else:
            subprocess.call(["chmod", "+x", script_path], cwd=setup_temp_dir)
            command = ["/bin/bash", "-c", script_path]

        return command


class EnvironmentReference(_AbstractRunConfigElement):
    """References an existing environment definition stored in the workspace.

    An EnvironmentReference can be used in place of an Environment object.

    :param name: The name of the environment.
    :type name: string

    :param version: The version of the environment.
    :type version: string
    """

    _field_to_info_dict = collections.OrderedDict([
        ("name", _FieldInfo(str, "Environment name")),
        ("version", _FieldInfo(str, "Environment version"))
    ])

    def __init__(self, name, version=None):
        """Class EnvironmentReference constructor."""
        super(EnvironmentReference, self).__init__()

        self.name = name
        self.version = version
        self._initialized = True

    def __repr__(self):
        """Representation of the object.

        :return: Return the string form of the EnvironmentReference object
        :rtype: str
        """
        environment_dict = _serialize_to_dict(self)
        return json.dumps(environment_dict, indent=4)

    def get_environment(self, workspace):
        """Return the Environment object pointed at by this reference.

        :param workspace: The workspace that contains the persisted environment
        :type workspace: azureml.core.workspace.Workspace
        :return: The referenced Environment object
        :rtype: azureml.core.environment.Environment
        """
        return Environment.get(workspace, self.name, self.version)

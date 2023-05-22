# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

from __future__ import print_function

import docker
import io
import json
import logging
import os
import posixpath
import re
import requests
import subprocess
import tarfile
import tempfile
import time
import traceback
import uuid

from azureml._base_sdk_common import _ClientSessionId
from azureml._base_sdk_common.user_agent import get_user_agent
from azureml._base_sdk_common.utils import working_directory_context
from azureml._model_management._constants import ACI_WEBSERVICE_TYPE
from azureml._model_management._constants import AKS_WEBSERVICE_TYPE
from azureml._model_management._constants import AKS_ENDPOINT_TYPE
from azureml._model_management._constants import COMPUTE_TYPE_KEY
from azureml._model_management._constants import DOCKER_IMAGE_HTTP_PORT
from azureml._model_management._constants import HEALTH_CHECK_INTERVAL_SECONDS
from azureml._model_management._constants import MAX_HEALTH_CHECK_TRIES
from azureml._model_management._constants import MMS_SYNC_TIMEOUT_SECONDS
from azureml._model_management._constants import MMS_ROUTE_API_VERSION
from azureml._model_management._constants import WEBSERVICE_SCORE_PATH
from azureml._model_management._constants import WORKSPACE_RP_API_VERSION
from azureml._restclient.clientbase import ClientBase
from azureml.exceptions import UserErrorException
from azureml.exceptions import WebserviceException
from base64 import b64decode
from keyword import iskeyword
from pkg_resources import resource_string

try:
    # python 3
    from urllib.parse import urlparse
except ImportError:
    # python 2
    from urlparse import urlparse


CUDA_9_BASE_IMAGE = 'base-gpu:intelmpi2018.3-cuda9.0-cudnn7-ubuntu16.04'

model_payload_template = json.loads(resource_string(__name__, 'data/mms_model_payload_template.json').decode('ascii'))
image_payload_template = json.loads(resource_string(__name__,
                                                    'data/mms_workspace_image_payload_template.json').decode('ascii'))
profile_payload_template = json.loads(
    resource_string(__name__, "data/mms_profile_payload_template.json").decode("ascii")
)
aks_service_payload_template = json.loads(resource_string(__name__,
                                                          'data/aks_service_payload_template.json').decode('ascii'))
aci_service_payload_template = json.loads(resource_string(__name__,
                                                          'data/aci_service_payload_template.json').decode('ascii'))

# Templates for MMS creation requests involving environments
environment_image_payload_template = json.loads(
    resource_string(__name__, 'data/environment_image_payload_template.json').decode('ascii'))
base_service_create_payload_template = json.loads(
    resource_string(__name__, 'data/base_service_create_payload_template.json').decode('ascii'))
aci_specific_service_create_payload_template = json.loads(
    resource_string(__name__, 'data/aci_specific_service_create_payload_template.json').decode('ascii'))
aks_specific_service_create_payload_template = json.loads(
    resource_string(__name__, 'data/aks_specific_service_create_payload_template.json').decode('ascii'))
aks_specific_endpoint_create_payload_template = json.loads(
    resource_string(__name__, 'data/aks_specific_endpoint_create_payload_template.json').decode('ascii'))

module_logger = logging.getLogger(__name__)


def add_sdk_to_requirements():
    """
    Inject the SDK as a requirement for created images, since we use the SDK for schema and exception handling.
    :return: Path to the created pip-requirements file.
    :rtype: str
    """
    # always pass sdk as dependency to mms/ICE.
    generated_requirements = tempfile.mkstemp(suffix='.txt', prefix='requirements')[1]
    sdk_requirements_string = 'azureml-model-management-sdk==1.0.1b6.post1'
    with open(generated_requirements, 'w') as generated_requirements_file:
        generated_requirements_file.write(sdk_requirements_string)

    return generated_requirements


def create_tar_in_memory(files):
    """
    Construct a tar file in-memory.
    :param files: Sequence of (path, data) tuples.
    :type files: list
    :return: Tar file.
    :rtype: File-like object.
    """
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode='w') as tarf:
        def add_entry(name, content):
            tarinfo = tarfile.TarInfo(name)
            tarinfo.mtime = time.time()
            tarinfo.size = len(content)
            tarf.addfile(tarinfo, io.BytesIO(content))

        for path, data in files:
            if isinstance(data, bytes):
                add_entry(path, data)
            elif isinstance(data, str):
                # Assume data is a local path.
                if os.path.isdir(data):
                    for dirpath, _, filenames in os.walk(data):
                        for filename in filenames:
                            fullpath = os.path.join(dirpath, filename)
                            relpath = os.path.relpath(fullpath, start=data)

                            with open(fullpath, 'rb') as f:
                                add_entry(posixpath.join(path, *relpath.split(os.sep)), f.read())
                else:
                    # Assume data is a path to a regular file.
                    with open(data, 'rb') as f:
                        add_entry(path, f.read())
            else:
                # Assume data is file-like.
                add_entry(path, data.read())

    # Rewind so read() starts from the beginning
    output.seek(0)

    return output


def upload_dependency(workspace, dependency, create_tar=False, arcname=None, show_output=False):
    """
    :param workspace: AzureML workspace
    :type workspace: workspace: azureml.core.workspace.Workspace
    :param dependency: path (local, http[s], or wasb[s]) to dependency
    :type dependency: str
    :param create_tar: tar creation flag. Defaults to False
    :type create_tar: bool
    :param arcname: arcname to use for tar
    :type arcname: str
    :param show_output: Indicates whether to display the progress of service deployment.
    :type show_output: bool
    :return: (str, str): uploaded_location, dependency_name
    """
    from azureml._restclient.artifacts_client import ArtifactsClient
    artifact_client = ArtifactsClient(workspace.service_context)
    if dependency.startswith('http') or dependency.startswith('wasb'):
        return dependency, urlparse(dependency).path.split('/')[-1]
    if not os.path.exists(dependency):
        raise WebserviceException('Error resolving dependency: '
                                  'no such file or directory {}'.format(dependency), logger=module_logger)

    dependency = dependency.rstrip(os.sep)
    dependency_name = os.path.basename(dependency)
    dependency_path = dependency

    if create_tar:
        dependency_name = str(uuid.uuid4())[:8] + '.tar.gz'
        tmpdir = tempfile.mkdtemp()
        dependency_path = os.path.join(tmpdir, dependency_name)
        dependency_tar = tarfile.open(dependency_path, 'w:gz')
        dependency_tar.add(dependency, arcname=arcname) if arcname else dependency_tar.add(dependency)
        dependency_tar.close()

    origin = 'LocalUpload'
    container = '{}'.format(str(uuid.uuid4())[:8])

    if show_output:
        print("Uploading dependency {}.".format(dependency_path))
    result = artifact_client.upload_artifact_from_path(dependency_path, origin, container, dependency_name)
    artifact_content = result.artifacts[dependency_name]
    dependency_name = artifact_content.path
    uploaded_location = "aml://artifact/" + artifact_content.artifact_id
    return uploaded_location, dependency_name


def wrap_execution_script_with_source_directory(source_directory, execution_script, log_aml_debug):
    """
    Wrap the user's execution script in our own in order to provide schema validation.
    :param execution_script: str path to execution script
    :param schema_file: str path to schema file
    :param dependencies: list of str paths to dependencies
    :return: str path to wrapped execution script
    """
    new_script_loc = tempfile.mkstemp(suffix='.py')[1]

    execution_script = os.path.join(source_directory, execution_script).replace(os.sep, '/')

    return generate_main(execution_script, '', new_script_loc, log_aml_debug, source_directory)


def wrap_execution_script(execution_script, schema_file, dependencies, log_aml_debug):
    """
    Wrap the user's execution script in our own in order to provide schema validation.
    :param execution_script: str path to execution script
    :param schema_file: str path to schema file
    :param dependencies: list of str paths to dependencies
    :return: str path to wrapped execution script
    """
    new_script_loc = tempfile.mkstemp(suffix='.py')[1]
    dependencies.append(execution_script)
    if not os.path.exists(execution_script):
        raise WebserviceException('Path to execution script {} does not exist.'.format(execution_script),
                                  logger=module_logger)
    if not schema_file:
        schema_file = ''
    else:
        # Fix path references in schema file,
        # since the container it will run in is a Linux container
        path_components = schema_file.split(os.sep)
        schema_file = "/".join(path_components)
    return generate_main(execution_script, schema_file, new_script_loc, log_aml_debug)


def generate_main(user_file, schema_file, main_file_name, log_aml_debug, source_directory=None):
    """

    :param user_file: str path to user file with init() and run()
    :param schema_file: str path to user schema file
    :param main_file_name: str full path of file to create
    :return: str filepath to generated file
    """

    data_directory = os.path.join(os.path.dirname(__file__), 'data')
    source_directory_import = ''
    if source_directory:
        source_directory_import = "sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), {}))"\
            .format(repr(source_directory))
    main_template_path = os.path.join(data_directory, 'main_template.txt')

    with open(main_template_path) as template_file:
        main_src = template_file.read()

    main_src = main_src.replace('<user_script>', user_file)\
        .replace('<source_directory_import>', source_directory_import)
    with open(main_file_name, 'w') as main_file:
        main_file.write(main_src)
    return main_file_name


def _validate_schema_version(schema_file):
    with open(schema_file, 'r') as outfile:
        schema_json = json.load(outfile)

    if "input" in schema_json:
        for key in schema_json["input"]:
            _check_inout_item_schema_version(schema_json["input"][key])
            break
    elif "output" in schema_json:
        for key in schema_json["input"]:
            _check_inout_item_schema_version(schema_json["output"][key])
            break


def _check_inout_item_schema_version(item_schema):
    _pup_version = "0.1.0a11"
    _schema_version_error_format = "The given schema cannot be loaded because it was generated on a SDK version - " \
                                   "{} that is not compatible with the one used to create the ML service - {}. " \
                                   "Please either update the SDK to the version for which the schema was generated, " \
                                   "or regenerate the schema file with the currently installed version of the SDK.\n" \
                                   "You can install the latest SDK with:\n" \
                                   "\t pip install azureml-model-management-sdk --upgrade\n " \
                                   "or upgrade to an earlier version with:\n" \
                                   "\t pip install azureml-model-management-sdk=<version>"
    if "version" not in item_schema:
        schema_version = _pup_version
    else:
        schema_version = item_schema["version"]

    # Check here if the schema version matches the current running SDK version (major version match)
    # and error out if not, since we do not support cross major version backwards compatibility.
    # Exception is given if schema is assumed to be _pup_version since the move from that to 1.0 was
    # not considered a major version release, and we should not fail for all PUP customers.
    sdk_version = '1.0.1b6'
    current_major = int(sdk_version.split('.')[0])
    deserialized_schema_major = int(schema_version.split('.')[0])
    if schema_version != _pup_version and current_major != deserialized_schema_major:
        error_msg = _schema_version_error_format.format(schema_version, sdk_version)
        raise ValueError(error_msg)

    return schema_version


def get_paginated_results(payload, headers):
    if 'value' not in payload:
        raise WebserviceException('Error, invalid paginated response payload, missing "value":\n'
                                  '{}'.format(payload), logger=module_logger)
    items = payload['value']
    while 'nextLink' in payload:
        next_link = payload['nextLink']

        try:
            resp = ClientBase._execute_func(get_requests_session().get, next_link, headers=headers,
                                            timeout=MMS_SYNC_TIMEOUT_SECONDS)
        except requests.Timeout:
            raise WebserviceException('Error, request to Model Management Service timed out to URL:\n'
                                      '{}'.format(next_link), logger=module_logger)
        if resp.status_code == 200:
            content = resp.content
            if isinstance(content, bytes):
                content = content.decode('utf-8')
            payload = json.loads(content)
        else:
            raise WebserviceException('Received bad response from Model Management Service while retrieving paginated'
                                      'results:\n'
                                      'Response Code: {}\n'
                                      'Headers: {}\n'
                                      'Content: {}'.format(resp.status_code, resp.headers, resp.content),
                                      logger=module_logger)
        if 'value' not in payload:
            raise WebserviceException('Error, invalid paginated response payload, missing "value":\n'
                                      '{}'.format(payload), logger=module_logger)
        items += payload['value']

    return items


def _get_mms_url(workspace):
    from azureml._restclient.assets_client import AssetsClient
    assets_client = AssetsClient(workspace.service_context)
    mms_url = assets_client.get_cluster_url()
    uri = '/modelmanagement/{}/subscriptions/{}/resourceGroups/{}/providers/' \
          'Microsoft.MachineLearningServices/workspaces/{}'.format(MMS_ROUTE_API_VERSION,
                                                                   workspace.subscription_id,
                                                                   workspace.resource_group,
                                                                   workspace.name)
    return mms_url + uri


def get_docker_client():
    """
    Retrieves the docker client for performing docker operations
    :return:
    :rtype: docker.DockerClient
    :raises: WebserviceException
    """
    try:
        client = docker.from_env(version='auto')
    except docker.errors.DockerException as exc:
        raise WebserviceException('Failed to create Docker client. Is Docker running/installed?\n'
                                  'When you deploy locally, we download a dockerfile\n'
                                  'execute docker build on it, and docker run the built container for you\n'
                                  'Error: {}'.format(exc), logger=module_logger)

    try:
        client.version()
    except Exception as exc:
        raise WebserviceException('Unable to communicate with Docker daemon. Is Docker running/installed?\n'
                                  'When you deploy locally, we download a dockerfile\n'
                                  'execute docker build on it, and docker run the built container for you\n'
                                  '\n\nDocker Client {}'.format(exc), logger=module_logger)

    return client


def login_to_docker_registry(docker_client, username, password, location):
    """
    Logs into a Docker registry.
    :param docker_client: The Docker client.
    :type docker_client: docker.DockerClient
    :param username: Username for the Docker registry.
    :type username: str
    :param password: Password for the Docker registry.
    :type password: str
    :param location: Location of the Docker registry.
    :type location: str
    :raises: WebserviceException
    """
    try:
        print('Logging into Docker registry {}'.format(location))
        docker_client.login(username=username, password=password, registry=location)
    except Exception:
        raise WebserviceException('Unable to login to Docker registry. {}'.format(traceback.format_exc()),
                                  logger=module_logger)


def pull_docker_image(docker_client, image_location, username, password):
    """
    Pulls the docker image from the ACR
    :param docker_client:
    :type docker_client: docker.DockerClient
    :param image_location:
    :type image_location: str
    :param username:
    :type username: str
    :param password:
    :type password: str
    :return:
    :rtype: None
    """
    try:
        print('Pulling image from ACR (this may take a few minutes depending on image size)...\n')
        for message in docker_client.api.pull(image_location, stream=True, decode=True, auth_config={
            'username': username,
            'password': password
        }):
            prefix = '{}: '.format(message['id']) if 'id' in message else ''
            status = message['status']
            progress = message.get('progressDetails', {}).get('progress', '')
            print(prefix + status + progress)
    except docker.errors.APIError as e:
        raise WebserviceException('Error: docker image pull has failed:\n{}'.format(e), logger=module_logger)
    except Exception as exc:
        raise WebserviceException('Error with docker pull {}'.format(exc), logger=module_logger)


def build_docker_image(docker_client, dockerfile, tag, custom_context=False, dockerfile_path=None, pull=False):
    """
    Builds a Docker image given a Dockerfile.
    :param docker_client: Docker client instance.
    :type docker_client: docker.DockerClient
    :param dockerfile: Path to Dockerfile or file-like object containing Dockerfile or custom build context.
    :type dockerfile: str or file
    :param custom_context: Whether dockerfile is a tar file object with a custom build context. Defaults to False
    :type custom_context: bool
    :param dockerfile_path: Path of Dockerfile within custom build context.
    :type dockerfile_path: str
    :param pull: Update base images in Dockerfile. Defaults to False
    :type pull: bool
    """
    try:
        print('Building Docker image from Dockerfile...')

        for status in docker_client.api.build(
                path=dockerfile if isinstance(dockerfile, str) else None,
                fileobj=dockerfile if not isinstance(dockerfile, str) else None,
                tag=tag,
                custom_context=custom_context,
                pull=pull,
                dockerfile=dockerfile_path,
                decode=True):
            if 'stream' in status:
                print(status['stream'], end='', flush=True)

            if 'error' in status:
                print(status, flush=True)
                raise WebserviceException('Error: docker image build has failed: {}'.format(status['error']),
                                          logger=module_logger)

        return docker_client.images.get(tag)
    except docker.errors.APIError as e:
        raise WebserviceException('Error: docker image build has failed:\n{}'.format(e), logger=module_logger)
    except Exception as e:
        if isinstance(e, WebserviceException):
            raise

        raise WebserviceException('Error with docker build: {}'.format(e), logger=module_logger)


def create_docker_container(docker_client, image_location, container_name, auto_remove=False, environment=None,
                            labels=None, ports=None, volumes=None):
    """
    Creates (but does not start) a docker container given the image location and container name.
    """
    try:
        labels = labels or {}
        labels['containerName'] = container_name

        return docker_client.containers.create(image_location,
                                               auto_remove=auto_remove,
                                               detach=True,
                                               environment=environment,
                                               labels=labels,
                                               ports=ports,
                                               volumes=volumes)
    except docker.errors.APIError as exc:
        raise WebserviceException('Docker container create has failed:\n{}'.format(exc), logger=module_logger)


def create_docker_volume(docker_client, volume_name, labels=None):
    """
    Creates a Docker volume.
    """
    try:
        return docker_client.volumes.create(volume_name, labels=labels)
    except docker.errors.APIError as e:
        raise WebserviceException('Error: Docker volume create has failed: {}'.format(e), logger=module_logger)


def start_docker_container(container):
    """
    Starts an existing Docker container.
    """
    try:
        print("Starting Docker container...")

        container.start()

        print('Docker container running.')
    except docker.errors.APIError as e:
        if 'port is already allocated' in e.explanation:
            raise WebserviceException('Docker container start has failed, the port you are attempting to use '
                                      'is already in use:\n{}'.format(e), logger=module_logger)
        else:
            raise WebserviceException('Docker container start has failed:\n{}'.format(e), logger=module_logger)


def run_docker_container(docker_client, image_location, container_name, **kwargs):
    """
    Creates and starts the docker container given the image location and container name
    :param docker_client:
    :type docker_client: docker.DockerClient
    :param image_location:
    :type image_location: str
    :param container_name:
    :type container_name: str
    :param ports: Mapping of container ports to host ports.
    :type ports: dict
    :return:
    :rtype: docker.Container
    """
    try:
        container = create_docker_container(docker_client, image_location, container_name, **kwargs)
        start_docker_container(container)

        return container
    except docker.errors.APIError as exc:
        raise WebserviceException('Docker container run has failed:\n{}'.format(exc), logger=module_logger)


def write_dir_in_container(container, container_path, local_path):
    """
    Copies a folder to a container's filesystem.
    """
    try:
        archive_bytes = io.BytesIO()

        with tarfile.open(fileobj=archive_bytes, mode='w') as archive_tar:
            archive_tar.add(local_path, arcname=container_path)

        archive_bytes.seek(0)

        container.put_archive('/', archive_bytes)
    except docker.errors.APIError as e:
        raise WebserviceException('Docker file copy has failed:\n{}'.format(e), logger=module_logger)


def write_file_in_container(container, path, data):
    """
    Writes data to a file in a container's filesystem.
    """
    try:
        archive_bytes = io.BytesIO()

        with tarfile.open(fileobj=archive_bytes, mode='w') as archive_tar:
            data_tarinfo = tarfile.TarInfo(name=path)
            data_tarinfo.mtime = time.time()
            data_tarinfo.size = len(data)

            archive_tar.addfile(data_tarinfo, io.BytesIO(data))

        archive_bytes.seek(0)

        container.put_archive('/', archive_bytes)
    except docker.errors.APIError as e:
        raise WebserviceException('Docker file copy has failed:\n{}'.format(e), logger=module_logger)


def get_docker_container(docker_client, container_name, all=None, limit=None):
    """
    Retrieves the container object for a local service.
    """
    try:
        containers = docker_client.containers.list(filters={'label': 'containerName={}'.format(container_name)},
                                                   all=all,
                                                   limit=limit)
    except docker.errors.DockerException as exc:
        raise WebserviceException('Failed to get Docker container:\n{}'.format(exc))

    if len(containers) == 0:
        raise WebserviceException('WebserviceNotFound: No local container with name {}'.format(container_name))

    if len(containers) != 1:
        raise WebserviceException('Error: Multiple containers with container name {}'.format(container_name))

    return containers[0]


def get_docker_containers(docker_client, label_name=None, label_value=None, all=None):
    """
    Retrieves all container objects matching the given label.
    """
    try:
        filters = {}

        if label_name is not None:
            if label_value is not None:
                filters['label'] = '{}={}'.format(label_name, label_value)
            else:
                filters['label'] = label_name

        return docker_client.containers.list(filters=filters, all=all)
    except docker.errors.DockerException as e:
        raise WebserviceException('Failed to get Docker containers:\n{}'.format(e), logger=module_logger)


def get_docker_host_container(docker_client):
    """
    Retrieves the Docker container we're currently running in, if any.
    """
    try:
        # Ask the Linux kernel.
        with open('/proc/self/cgroup') as f:
            cgroup_info = f.read()

        container_id = re.search('/docker/([0-9a-fA-F]+)', cgroup_info).groups()[0]

        return docker_client.containers.get(container_id)
    except docker.errors.APIError as e:
        raise WebserviceException('Failed to get Docker container:\n{}'.format(e), logger=module_logger)
    except Exception:
        pass

    try:
        # HOSTNAME is set to the container short ID, by default.
        return docker_client.containers.get(os.environ['HOSTNAME'])
    except docker.errors.APIError as e:
        raise WebserviceException('Failed to get Docker container:\n{}'.format(e), logger=module_logger)
    except Exception:
        pass

    return None


def get_docker_logs(container, num_lines=5000):
    try:
        return container.logs(tail=num_lines).decode()
    except docker.errors.APIError as e:
        raise WebserviceException('Failed to get Docker container logs: {}'.format(e), logger=module_logger)


def get_docker_network(docker_client, network_name):
    """
    Gets (or creates) a Docker network by its name.
    """
    try:
        return docker_client.networks.list(network_name)[0]
    except docker.errors.APIError as e:
        raise WebserviceException('Failed to get Docker network {}: {}'.format(network_name, e), logger=module_logger)
    except IndexError:
        pass  # Network doesn't exist

    try:
        return docker_client.networks.create(network_name)
    except docker.errors.APIError as e:
        raise WebserviceException('Failed to create Docker network {}: {}'.format(network_name, e),
                                  logger=module_logger)


def get_docker_port(docker_client, container_name, container, internal_port=None):
    """
    Starts the docker container given the image location and container name and returns the docker port
    :param docker_client:
    :type docker_client: docker.DockerClient
    :param container_name:
    :type container_name: str
    :param container:
    :type container: docker.Container
    :param internal_port
    :type internal_port: int
    :return:
    :rtype: str
    """
    container = get_docker_container(docker_client, container_name)
    port = internal_port or DOCKER_IMAGE_HTTP_PORT
    for key, value in container.attrs['NetworkSettings']['Ports'].items():
        if str(port) in key and value:
            port = value[0]['HostPort']
            return port

    _raise_for_container_failure(container, True,
                                 'Error: Container port ({}) is unreachable.'.format(DOCKER_IMAGE_HTTP_PORT))


def get_docker_volume(docker_client, volume_name, must_exist=True):
    """
    Gets a Docker volume by name.

    :param docker_client:
    :type docker_client: docker.DockerClient
    :param volume_name:
    :type volume_name: str
    :param must_exist: Raise exception if the volume does not exist.
    :type must_exist: bool
    :return:
    :rtype: docker.Volume
    :raises: WebserviceException
    """
    try:
        return docker_client.volumes.get(volume_name)
    except docker.errors.NotFound:
        if not must_exist:
            return None

        raise WebserviceException('Error: Docker volume {} does not exist.'.format(volume_name), logger=module_logger)
    except Exception as e:
        raise WebserviceException('Error: Failed to get Docker volume {}: {}'.format(volume_name, e),
                                  logger=module_logger)


def connect_docker_network(network, container, aliases=[None]):
    """
    Attaches a Docker container to a Docker network.
    """
    try:
        network.reload()

        # Connecting is slow, so don't redo it if we're already connected.
        if any(c.id == container.id for c in network.containers):
            return

        network.connect(container, aliases=aliases)
    except docker.errors.APIError as e:
        raise WebserviceException('Error: Failed to attach Docker container {} to Docker network {}: {}'
                                  .format(container.short_id, network.name, e), logger=module_logger)


def container_health_check(docker_port, container, health_url=None, cleanup_if_failed=True):
    """
    Sends a post request to check the health of the new container
    :param docker_client:
    :type docker_client: docker.DockerClient
    :param container:
    :type container: docker.Container
    :return:
    :rtype: str
    """
    print("Checking container health...")
    health_url = health_url or 'http://127.0.0.1:{}/'.format(docker_port)

    health_check_iteration = 0
    while health_check_iteration < MAX_HEALTH_CHECK_TRIES:
        # Check for complete failure (the container crashed).
        container.reload()

        if container.status == 'created':
            # The container hasn't started, yet.
            time.sleep(HEALTH_CHECK_INTERVAL_SECONDS)
            continue
        elif container.status == 'exited':
            # The container has started and crashed.
            _raise_for_container_failure(container, cleanup_if_failed,
                                         'Error: Container has crashed. Did your init method fail?')

        # The container hasn't crashed, so try to ping the health endpoint.
        try:
            result = ClientBase._execute_func(get_requests_session().get, health_url, verify=False)
        except requests.ConnectionError:
            time.sleep(HEALTH_CHECK_INTERVAL_SECONDS)
            health_check_iteration += 1
            if health_check_iteration >= MAX_HEALTH_CHECK_TRIES and 'http://127.0.0.1' == health_url[:16]:
                print("Container health check failed on localhost. Checking container health using container IP...")
                process = subprocess.Popen(['docker', 'inspect', '--format', "\'{{ .NetworkSettings.IPAddress }}\'",
                                            container.id], stdout=subprocess.PIPE)
                stdout = process.communicate()[0]
                container_ip = stdout.decode().rstrip().replace("'", "")
                if not container_ip:
                    _raise_for_container_failure(container, cleanup_if_failed,
                                                 'Error: Connection to container has failed. '
                                                 'Did your init method fail?')
                health_url = 'http://{}:5001/'.format(container_ip)
                health_check_iteration = 0
            continue
        except requests.Timeout:
            time.sleep(HEALTH_CHECK_INTERVAL_SECONDS)
            health_check_iteration += 1
            continue

        if result.status_code == 200:
            return health_url
        elif result.status_code == 502:
            time.sleep(HEALTH_CHECK_INTERVAL_SECONDS)
            health_check_iteration += 1
            continue
        else:
            if 'http://127.0.0.1' == health_url[:16]:
                print("Container health check failed on localhost. Checking container health using container IP...")
                process = subprocess.Popen(['docker', 'inspect', '--format', "\'{{ .NetworkSettings.IPAddress }}\'",
                                            container.id], stdout=subprocess.PIPE)
                stdout = process.communicate()[0]
                container_ip = stdout.decode().rstrip().replace("'", "")
                if not container_ip:
                    _raise_for_container_failure(container, cleanup_if_failed,
                                                 'Error: Connection to container has failed. '
                                                 'Did your init method fail?')
                health_url = 'http://{}:5001/'.format(container_ip)
                health_check_iteration = 0
                continue
            else:
                _raise_for_container_failure(container, cleanup_if_failed,
                                             'Error: Connection to container has failed. Did your init method fail?')

    _raise_for_container_failure(container, cleanup_if_failed,
                                 'Error: Connection to container has timed out. Did your init method fail?')


def container_scoring_call(docker_port, input_data, container, health_url, cleanup_on_failure=True, score_url=None):
    """
    Sends a scoring call to the given container with the given input data
    :param docker_client:
    :type docker_client: docker.DockerClient
    :param input_data:
    :type input_data: str
    :param container:
    :type container: docker.Container
    :return:
    :rtype: str
    """
    if score_url is None:
        u = urlparse(health_url)
        scoring_url = "{}://{}{}".format(u.scheme, u.netloc, WEBSERVICE_SCORE_PATH)
    else:
        scoring_url = score_url

    headers = {'Content-Type': 'application/json'}
    try:
        result = ClientBase._execute_func(get_requests_session().post, scoring_url, headers=headers, data=input_data,
                                          verify=False)
    except requests.exceptions.HTTPError:
        _raise_for_container_failure(container, cleanup_on_failure,
                                     'Error: Connection with container for scoring has failed.')

    if result.status_code == 200:
        return result.json()
    else:
        content = result.content.decode('utf-8')
        if content == "ehostunreach":
            _raise_for_container_failure(container, cleanup_on_failure,
                                         'Error: Scoring was unsuccessful. Unable to reach the requested host.')

        if cleanup_on_failure:
            cleanup_container(container)

        raise WebserviceException('Error: Scoring was unsuccessful.', logger=module_logger)


def cleanup_container(container):
    """
    Tries to kill and remove the container
    :param container:
    :type container: docker.Container
    :return:
    :rtype: None
    """
    try:
        container.kill()
    except Exception:
        print('Container (name:{}, id:{}) cannot be killed.'.format(container.name, container.id))
    try:
        container.remove()
        print('Container has been successfully cleaned up.')
    except Exception:
        print('Container (name:{}, id:{}) cannot be removed.'.format(container.name, container.id))


def cleanup_docker_image(docker_client, image_id):
    """
    Deletes a local Docker image.

    :param docker_client:
    :type docker_client: docker.DockerClient
    :param image_id: Docker image ID.
    :type image_id: str
    :return:
    :rtype: None
    """
    try:
        docker_client.images.remove(image_id)
        print('Image {} successfully removed.'.format(image_id))
    except Exception as e:
        print('Image {} cannot be removed: {}'.format(image_id, str(e)))


def validate_path_exists_or_throw(member, name, extra_message=''):
    if not os.path.exists(member):
        raise WebserviceException("{0} {1} doesn't exist. {2}".format(name, member, extra_message),
                                  logger=module_logger)


def joinPath(source_directory, path):
    if path and source_directory:
        return os.path.join(source_directory, path).replace(os.sep, '/')
    else:
        return path


def validate_entry_script_name(name):
    """

    :param name:
    :type name: str
    :raises: WebserviceException
    """
    if not name.isidentifier() or iskeyword(name):
        raise WebserviceException('Error, invalid name "{}" for provided entry script. The script must be importable '
                                  'as a valid python module, and must adhere to standard module naming. This means it '
                                  'must be a valid python identifier and not a part of the set of python standard '
                                  'keywords.'.format(name), logger=module_logger)


def check_duplicate_properties(current_properties, new_properties):
    """

    :param current_properties:
    :type current_properties: dict[str, str]
    :param new_properties:
    :type new_properties: dict[str, str]
    :raises: WebserviceException
    """
    if new_properties:
        duplicate_properties = []
        for key in new_properties:
            if key in current_properties:
                duplicate_properties.append(key)
        if duplicate_properties:
            raise WebserviceException('Error, properties are immutable, the following keys cannot be added/modified: '
                                      '{}'.format(','.join(duplicate_properties)), logger=module_logger)


def download_docker_build_context(workspace, manifest, output_directory):
    """
    Download a Docker build context as generated by MMS.

    :param workspace: Workspace with the build context.
    :type workspace: azureml.core.Workspace
    :param manifest: Build context manifest generated by MMS.
    :type manifest: dict
    :param output_directory: Directory within which to download the build context.
    :type output_directory: str
    """
    os.makedirs(output_directory, exist_ok=True)

    from azureml._restclient.artifacts_client import ArtifactsClient
    artifacts_client = ArtifactsClient(workspace.service_context)

    with working_directory_context(output_directory), tempfile.TemporaryDirectory() as temp_dir:
        for entry in manifest['context']:
            if entry['entryType'] == 'inline':
                if entry['encoding'] == 'base64':
                    data = b64decode(entry['content'])
                else:
                    raise WebserviceException('Error, unknown encoding for build context entry: {}'
                                              .format(entry['encoding']), logger=module_logger)

                with open(entry['path'], 'wb') as f:
                    f.write(data)
            elif entry['entryType'] == 'remote':
                origin, container, prefix = re.match('aml://artifact/([^/]*)/([^/]*)(?:/(.*))?', entry['uri']).groups()

                if entry['unpack']:
                    temp_path = os.path.join(temp_dir, prefix.split('/')[-1])
                    artifacts_client.download_artifact(origin, container, prefix, temp_path)

                    if tarfile.is_tarfile(temp_path):
                        with tarfile.open(temp_path) as f:
                            f.extractall(entry['path'])
                    else:
                        raise WebserviceException('Error, unrecognized archive extension: {}'.format(temp_path),
                                                  logger=module_logger)
                else:
                    files = artifacts_client.get_files_by_artifact_prefix_id('{}/{}/{}'
                                                                             .format(origin, container, prefix))
                    path_split = prefix.rstrip('/').rfind('/') + 1
                    paths = [path[path_split:] for path, _ in files]

                    artifacts_client.download_artifacts_from_prefix(origin, container, prefix,
                                                                    output_directory=entry['path'], output_paths=paths)
            else:
                raise WebserviceException('Error, unknown context entry type: {}'.format(entry['entryType']),
                                          logger=module_logger)


def get_mms_operation(workspace, operation_id):
    """
    Retrieve the operation payload from MMS.

    :return: The json encoded content of the reponse.
    :rtype: dict
    """
    response = make_mms_request(workspace, 'GET', '/operations/' + operation_id, None)
    return response.json()


def get_requests_session():
    """

    :return: A requests.Session object
    :rtype: requests.Session
    """
    _session = requests.Session()
    _session.headers.update({
        'User-Agent': get_user_agent(),
        'x-ms-client-session-id': _ClientSessionId,
    })

    return _session


def get_workspace_registry_credentials(workspace):
    """
    Get the username and password for the ACR attached to a workspace.

    :param workspace: The workspace to query.
    :type workspace: :class:`azureml.core.Workspace`
    :return: The username and password.
    :rtype: tuple[str, str]
    :raises: :class:`azureml.exceptions.WebserviceException`
    """
    keys_endpoint = 'https://management.azure.com/subscriptions/{}/resourceGroups/{}/providers/' \
                    'Microsoft.MachineLearningServices/workspaces/' \
                    '{}/listKeys'.format(workspace.subscription_id,
                                         workspace.resource_group,
                                         workspace.name)
    headers = workspace._auth.get_authentication_header()
    params = {'api-version': WORKSPACE_RP_API_VERSION}
    response = make_http_request('POST', keys_endpoint, headers=headers, params=params)

    try:
        keys_dict = response.json()
        username = keys_dict['containerRegistryCredentials']['username']
        password = keys_dict['containerRegistryCredentials']['passwords'][0]['value']
    except KeyError:
        raise WebserviceException('Unable to retrieve workspace keys to run image, response '
                                  'payload missing container registry credentials.', logger=module_logger)

    return username, password


def make_http_request(method, url, check_response=None, *args, **kwargs):
    """
    Make an HTTP request with the global session.

    :return: The response object.
    :raises: WebserviceException
    """
    session = get_requests_session()
    response = ClientBase._execute_func(lambda *args, **kwargs: session.request(method, *args, **kwargs),
                                        url, *args, **kwargs)

    try:
        if check_response is None:
            response.raise_for_status()
        elif not check_response(response):
            raise WebserviceException('Received bad response from service:\n'
                                      'Response Code: {}\n'
                                      'Headers: {}\n'
                                      'Content: {}'.format(response.status_code, response.headers, response.content),
                                      logger=module_logger)
    except requests.Timeout:
        raise WebserviceException('Error, request to {} timed out.'.format(url), logger=module_logger)
    except requests.exceptions.HTTPError:
        raise WebserviceException('Received bad response from service:\n'
                                  'Response Code: {}\n'
                                  'Headers: {}\n'
                                  'Content: {}'.format(response.status_code, response.headers, response.content),
                                  logger=module_logger)

    return response


def make_mms_request(workspace, method, url_suffix, json_body, params={}):
    """
    Handle the common components of calling MMS (base URL, auth header).

    :return: The response object.
    """
    headers = {'Content-Type': 'application/json'}
    headers.update(workspace._auth.get_authentication_header())

    url = _get_mms_url(workspace) + url_suffix

    return make_http_request(method, url, params=params, headers=headers, json=json_body)


def submit_mms_operation(workspace, method, url_suffix, json_body):
    """
    Make a request for which MMS will create an async operation.

    :return: The operation ID.
    """
    response = make_mms_request(workspace, method, url_suffix, json_body)

    if 'Operation-Location' in response.headers:
        operation_location = response.headers['Operation-Location']
    else:
        raise WebserviceException('Missing response header key: Operation-Location', logger=module_logger)

    operation_id = operation_location.split('/')[-1]

    return operation_id


def get_operation_output(workspace, url_suffix, params={}):
    """
    Make a request for which MMS will retrieve an object produced as a result of a create operation.
    For example, this function can be used to retrieve the result of a profiling operation.

    :return: Response object
    :raises: WebserviceException
    """
    params.update({'orderBy': 'CreatedAtDesc', 'count': 1})
    resp = make_mms_request(workspace, 'GET', url_suffix, None, params)
    if resp.status_code not in (200, 404):
        raise WebserviceException('Received bad response from Model Management Service:\n'
                                  'Response Code: {}\n'
                                  'Headers: {}\n'
                                  'Content: {}'.format(resp.status_code, resp.headers, resp.content))
    return resp


def model_name_validation(name):
    """

    :param name:
    :type name: str
    :return:
    :rtype: bool
    :raises: WebserviceException
    """
    if not re.fullmatch(r'^[a-zA-Z0-9][a-zA-Z0-9\-\._]{0,254}$', name):
        raise WebserviceException('Error, provided model name is invalid. It must only consist of letters, '
                                  'numbers, dashes, periods, or underscores, start with a letter or number, and be '
                                  'between 1 and 255 characters long.')


def image_name_validation(name):
    """

    :param name:
    :type name: str
    :return:
    :rtype: bool
    :raises: WebserviceException
    """
    if not re.fullmatch(r'^[a-z0-9]([-\.a-z0-9]*[a-z0-9])?$', name) or len(name) < 3 or len(name) > 32:
        raise WebserviceException('Error, provided image name is invalid. It must only consist of lowercase letters, '
                                  'numbers, dashes or periods, start with a letter or number, end with a letter or '
                                  'number, and be between 3 and 32 characters long.')


def webservice_name_validation(name):
    """

    :param name:
    :type name: str
    :return:
    :rtype: bool
    :raises: WebserviceException
    """
    if not re.fullmatch(r'^[a-z]([-a-z0-9]*[a-z0-9])?$', name) or len(name) < 3 or len(name) > 32:
        raise WebserviceException('Error, provided service name is invalid. It must only consist of lowercase '
                                  'letters, numbers, or dashes, start with a letter, end with a letter or number, '
                                  'and be between 3 and 32 characters long.')


def build_and_validate_no_code_environment_image_request(models):
    """Construct and validate an environment image request payload solely from a set of models.

    Normally an environment image request sent to MMS contains an environment,
    the model IDs to load into that environment, and the driver program/any
    other files needed to run the models. However, in the no-code deploy case
    the environment and driver program are implicitly managed by MMS (and may
    not exist at all, in some cases), leaving only the model IDs.

    :param models:
    :type models: list[azureml.core.Model]
    :return: An environment image request payload.
    :rtype: dict
    """
    from azureml.core import Model

    # Check no-code deploy prerequisites.
    if len(models) != 1:
        raise UserErrorException('You must provide an InferenceConfig when deploying zero or multiple models.')

    model = models[0]

    if model.model_framework not in Model._SUPPORTED_FRAMEWORKS_FOR_NO_CODE_DEPLOY:
        raise UserErrorException('You must provide an InferenceConfig when deploying a model with model_framework '
                                 'set to {}. Default environments are only provided for these frameworks: {}.'
                                 .format(model.model_framework, Model._SUPPORTED_FRAMEWORKS_FOR_NO_CODE_DEPLOY))

    # Only specify the model IDs; MMS will provide the environment, driver program, etc.
    return {'modelIds': [model.id]}


def convert_parts_to_environment(service_name, inference_config):
    """Attempt to create an environment object from pieces of the inference config

    :param service_name:
    :type service_name: str
    :param inference_config:
    :type inference_config: azureml.core.model.InferenceConfig
    :return: Either the original InferenceConfig or a new one with an Environment.
    :return: tuple(azureml.core.model.InferenceConfig, bool)
            where
            - azureml.core.model.InferenceConfig = user provided input.
            - bool = Whether to use environment route or old image route.
    :rtype: (azureml.core.model.InferenceConfig, bool)
    """
    # If a user doesn't provide an environment object, we will attempt to create one if the provided parameters
    # are accepted by the current state of EMS. Currently that means no dockerfile, cuda_version 9.0 or 10.0,
    # and no specified gpu handling at all if they specify a base image.
    from azureml.core.environment import Environment, DEFAULT_GPU_IMAGE
    if not inference_config.environment and inference_config.runtime == 'python' \
            and not inference_config.extra_docker_file_steps and not inference_config.cuda_version == '9.1' \
            and not (inference_config.base_image and (inference_config.enable_gpu
                                                      or inference_config.cuda_version)):
        env_name = '{}-env'.format(service_name)
        if inference_config.conda_file:
            if inference_config.source_directory:
                conda_path = os.path.join(inference_config.source_directory, inference_config.conda_file)
            else:
                conda_path = inference_config.conda_file
            env = Environment.from_conda_specification(env_name, conda_path)
        else:
            env = Environment(env_name)
        if inference_config.base_image:
            env.docker.base_image = inference_config.base_image
            env.docker.base_image_registry = inference_config.base_image_registry
            env.inferencing_stack_version = 'latest'
        if inference_config.enable_gpu or inference_config.cuda_version == '10.0':
            env.docker.gpu_support = True
            env.docker.base_image = DEFAULT_GPU_IMAGE
        elif inference_config.cuda_version == '9.0':
            env.docker.gpu_support = True
            env.docker.base_image = CUDA_9_BASE_IMAGE

        # Avoid recursive import
        return type(inference_config)(entry_script=inference_config.entry_script,
                                      source_directory=inference_config.source_directory,
                                      description=inference_config.description,
                                      environment=env), True

    if not inference_config.environment and not inference_config.runtime \
            and not inference_config.extra_docker_file_steps and \
            not inference_config.conda_file and inference_config.entry_script:

        # Avoid recursive import
        return inference_config, True

    else:
        return inference_config, inference_config.environment is not None


def _raise_for_container_failure(container, cleanup, message):
    logs = get_docker_logs(container)

    print('\nContainer Logs:')
    print(logs)

    if 'exec: gunicorn: not found' in logs:
        module_logger.warning('The container is missing requirements for the Azure Machine Learning serving stack. '
                              'Please make sure the "azureml-defaults" pip package is included in your environment.')

    if cleanup:
        cleanup_container(container)

    raise WebserviceException(message, logger=module_logger)


def serialize_object_without_none_values(original):
    """Serialize an object while stripping out values that are None.

    :param initial_dict:
    :type initial_dict: Dict
    :return:
    """
    if original is not None:
        original_dict = original.serialize()
        return {k: v for k, v in original_dict.items() if v is not None}

    return None


def populate_model_not_found_details(**kwargs):
    return 'ModelNotFound: Model with ' + \
           ', '.join(["{} {}".format(key, val) for key, val in kwargs.items() if val]) + \
           ' not found in provided workspace'


def dataset_to_dataset_reference(scenario, model_dataset):
    return {'name': scenario, 'id': model_dataset._registration.saved_id}


def deploy_config_dict_to_obj(deploy_config_dict, tags_dict, properties_dict, description):
    """Takes a deploy_config_dict from a file and returns the deployment config object
    :param deploy_config_dict: Input object with deployment config parameters
    :type deploy_config_dict: dict
    :return: Deployment config object
    :rtype: varies
    """
    from azureml.core.webservice import AciWebservice, AksWebservice, AksEndpoint, LocalWebservice

    try:
        if COMPUTE_TYPE_KEY not in deploy_config_dict:
            raise WebserviceException("Need to specify {} in --deploy-config-file".format(COMPUTE_TYPE_KEY))

        deploy_compute_type = deploy_config_dict[COMPUTE_TYPE_KEY].lower()

        if deploy_compute_type == ACI_WEBSERVICE_TYPE.lower():
            # aci deployment
            config = AciWebservice.deploy_configuration(
                cpu_cores=deploy_config_dict.get('containerResourceRequirements', {}).get('cpu'),
                memory_gb=deploy_config_dict.get('containerResourceRequirements', {}).get('memoryInGB'),
                tags=tags_dict,
                properties=properties_dict,
                description=description,
                location=deploy_config_dict.get('location'),
                auth_enabled=deploy_config_dict.get('authEnabled'),
                ssl_enabled=deploy_config_dict.get('sslEnabled'),
                enable_app_insights=deploy_config_dict.get('appInsightsEnabled'),
                ssl_cert_pem_file=deploy_config_dict.get('sslCertificate'),
                ssl_key_pem_file=deploy_config_dict.get('sslKey'),
                ssl_cname=deploy_config_dict.get('cname'),
                dns_name_label=deploy_config_dict.get('dnsNameLabel'),
                cmk_vault_base_url=deploy_config_dict.get('vaultBaseUrl'),
                cmk_key_name=deploy_config_dict.get('keyName'),
                cmk_key_version=deploy_config_dict.get('keyVersion'),
                vnet_name=deploy_config_dict.get('vnetName'),
                subnet_name=deploy_config_dict.get('subnetName'),
                primary_key=deploy_config_dict.get('keys', {}).get('primaryKey'),
                secondary_key=deploy_config_dict.get('keys', {}).get('secondaryKey'),
            )
        elif deploy_compute_type == AKS_WEBSERVICE_TYPE.lower():
            # aks deployment
            config = AksWebservice.deploy_configuration(
                autoscale_enabled=deploy_config_dict.get('autoScaler', {}).get('autoscaleEnabled'),
                autoscale_min_replicas=deploy_config_dict.get('autoScaler', {}).get('minReplicas'),
                autoscale_max_replicas=deploy_config_dict.get('autoScaler', {}).get('maxReplicas'),
                autoscale_refresh_seconds=deploy_config_dict.get('autoScaler', {}).get('refreshPeriodInSeconds'),
                autoscale_target_utilization=deploy_config_dict.get('autoScaler', {}).get('targetUtilization'),
                collect_model_data=deploy_config_dict.get('dataCollection', {}).get('storageEnabled'),
                auth_enabled=deploy_config_dict.get('authEnabled'),
                cpu_cores=deploy_config_dict.get('containerResourceRequirements', {}).get('cpu'),
                memory_gb=deploy_config_dict.get('containerResourceRequirements', {}).get('memoryInGB'),
                enable_app_insights=deploy_config_dict.get('appInsightsEnabled'),
                scoring_timeout_ms=deploy_config_dict.get('scoringTimeoutMs'),
                replica_max_concurrent_requests=deploy_config_dict.get('maxConcurrentRequestsPerContainer'),
                max_request_wait_time=deploy_config_dict.get('maxQueueWaitMs'),
                num_replicas=deploy_config_dict.get('numReplicas'),
                primary_key=deploy_config_dict.get('keys', {}).get('primaryKey'),
                secondary_key=deploy_config_dict.get('keys', {}).get('secondaryKey'),
                tags=tags_dict,
                properties=properties_dict,
                description=description,
                gpu_cores=deploy_config_dict.get('gpuCores'),
                period_seconds=deploy_config_dict.get('livenessProbeRequirements', {}).get('periodSeconds'),
                initial_delay_seconds=deploy_config_dict.get('livenessProbeRequirements',
                                                             {}).get('initialDelaySeconds'),
                timeout_seconds=deploy_config_dict.get('livenessProbeRequirements', {}).get('timeoutSeconds'),
                success_threshold=deploy_config_dict.get('livenessProbeRequirements', {}).get('successThreshold'),
                failure_threshold=deploy_config_dict.get('livenessProbeRequirements', {}).get('failureThreshold'),
                namespace=deploy_config_dict.get('namespace'),
                token_auth_enabled=deploy_config_dict.get("tokenAuthEnabled"),
                compute_target_name=deploy_config_dict.get("computeTargetName"),
                cpu_cores_limit=deploy_config_dict.get('containerResourceRequirements', {}).get('cpuLimit'),
                memory_gb_limit=deploy_config_dict.get('containerResourceRequirements', {}).get('memoryInGBLimit'))
        elif deploy_compute_type == AKS_ENDPOINT_TYPE.lower():
            config = AksEndpoint.deploy_configuration(
                autoscale_enabled=deploy_config_dict.get('autoScaler', {}).get('autoscaleEnabled'),
                autoscale_min_replicas=deploy_config_dict.get('autoScaler', {}).get('minReplicas'),
                autoscale_max_replicas=deploy_config_dict.get('autoScaler', {}).get('maxReplicas'),
                autoscale_refresh_seconds=deploy_config_dict.get('autoScaler', {}).get('refreshPeriodInSeconds'),
                autoscale_target_utilization=deploy_config_dict.get('autoScaler', {}).get('targetUtilization'),
                collect_model_data=deploy_config_dict.get('dataCollection', {}).get('storageEnabled'),
                auth_enabled=deploy_config_dict.get('authEnabled'),
                cpu_cores=deploy_config_dict.get('containerResourceRequirements', {}).get('cpu'),
                memory_gb=deploy_config_dict.get('containerResourceRequirements', {}).get('memoryInGB'),
                enable_app_insights=deploy_config_dict.get('appInsightsEnabled'),
                scoring_timeout_ms=deploy_config_dict.get('scoringTimeoutMs'),
                replica_max_concurrent_requests=deploy_config_dict.get('maxConcurrentRequestsPerContainer'),
                max_request_wait_time=deploy_config_dict.get('maxQueueWaitMs'),
                num_replicas=deploy_config_dict.get('numReplicas'),
                primary_key=deploy_config_dict.get('keys', {}).get('primaryKey'),
                secondary_key=deploy_config_dict.get('keys', {}).get('secondaryKey'),
                tags=tags_dict,
                properties=properties_dict,
                description=description,
                gpu_cores=deploy_config_dict.get('gpuCores'),
                period_seconds=deploy_config_dict.get('livenessProbeRequirements', {}).get('periodSeconds'),
                initial_delay_seconds=deploy_config_dict.get('livenessProbeRequirements',
                                                             {}).get('initialDelaySeconds'),
                timeout_seconds=deploy_config_dict.get('livenessProbeRequirements', {}).get('timeoutSeconds'),
                success_threshold=deploy_config_dict.get('livenessProbeRequirements', {}).get('successThreshold'),
                failure_threshold=deploy_config_dict.get('livenessProbeRequirements', {}).get('failureThreshold'),
                namespace=deploy_config_dict.get('namespace'),
                token_auth_enabled=deploy_config_dict.get("tokenAuthEnabled"),
                version_name=deploy_config_dict.get("versionName"),
                traffic_percentile=deploy_config_dict.get("trafficPercentile"),
                cpu_cores_limit=deploy_config_dict.get('containerResourceRequirements', {}).get('cpuLimit'),
                memory_gb_limit=deploy_config_dict.get('containerResourceRequirements', {}).get('memoryInGBLimit'))
        elif deploy_compute_type == "local":
            # local deployment
            config = LocalWebservice.deploy_configuration(
                port=deploy_config_dict.get('port'))
        else:
            raise WebserviceException("unknown deployment type: {}".format(deploy_compute_type))
        return config
    except Exception as ex:
        raise WebserviceException('Error parsing --deploy-config-file.') from ex

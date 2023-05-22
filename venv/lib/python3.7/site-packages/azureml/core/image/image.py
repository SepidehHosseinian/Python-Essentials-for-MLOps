# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""Contains the base functionality for managing images in Azure Machine Learning.

An :class:`azureml.core.image.image.Image` encapsulates a model, script, and associated files to provide
the run time environment of a deployed web service. The image has a load-balanced, HTTP endpoint that can
receive scoring requests and return predictions.
"""

import copy
import logging
import json
import requests
import sys
import time
import warnings
from dateutil.parser import parse
from azureml.core.model import Model
from azureml.exceptions import WebserviceException
from azureml._model_management._constants import MMS_SYNC_TIMEOUT_SECONDS
from azureml._model_management._constants import UNKNOWN_IMAGE_TYPE, UNKNOWN_IMAGE_FLAVOR
from azureml._model_management._constants import ARCHITECTURE_AMD64
from azureml._model_management._util import get_paginated_results
from azureml._model_management._util import get_requests_session
from azureml._model_management._util import _get_mms_url
from azureml._model_management._util import check_duplicate_properties
from azureml._model_management._util import image_name_validation
from azureml._restclient.clientbase import ClientBase

try:
    from abc import ABCMeta

    ABC = ABCMeta('ABC', (), {})
except ImportError:
    from abc import ABC
from abc import abstractmethod

module_logger = logging.getLogger(__name__)


class Image(ABC):
    """Defines the abstract parent class for Azure Machine Learning Images.

    This class is DEPRECATED. Use the :class:`azureml.core.Environment` class instead.

    .. remarks::

        The Image constructor retrieves a cloud representation of an Image
        object associated with the provided workspace. It returns an instance of a child class corresponding to the
        specific type of the retrieved Image object.

        An Image object is used to deploy a user's :class:`azureml.core.Model` as a :class:`azureml.core.Webservice`.
        The Image object typically contains a Model, an execution script, and any dependencies needed for
        Model deployment. The Image class has multiple subclasses such as ContainerImage for Docker Images,
        and Images like FPGA.

        See the :class:`azureml.core.image.ContainerImage` class for an example of a class that inherits from the
        Image class.

        Images are typically used in workflows that require using an image. For most workflows, you should instead
        use the :class:`azureml.core.Environment` class to define your image. Then you can use the Environment object
        with the :class:`azureml.core.model.Model` ``deploy()`` method to deploy the model as a web service.
        You can also use the Model ``package()`` method to create an image that can be downloaded to your local
        Docker install as an image or as a Dockerfile.

        See the following link for an overview on deploying models in Azure:
        `https://aka.ms/azureml-how-deploy`.

    :param workspace: The Workspace object containing the Image to retrieve.
    :type workspace: azureml.core.workspace.Workspace
    :param name: The name of the Image to retrieve. Will return the latest version of the Image, if it exists.
    :type name: str
    :param id: The specific ID of the Image to retrieve. (ID is "&lt;name&gt;:&lt;version&gt;")
    :type id: str
    :param tags: Will filter Image results based on the provided list, by either 'key' or '[key, value]'.
        Ex. ['key', ['key2', 'key2 value']]
    :type tags: builtin.list
    :param properties: Will filter Image results based on the provided list, by either 'key' or '[key, value]'.
        Ex. ['key', ['key2', 'key2 value']]
    :type properties: builtin.list
    :param version: When version and name are both specified, will return the specific version of the Image.
    :type version: str
    """

    _expected_payload_keys = ['createdTime', 'creationState', 'description', 'id',
                              'imageLocation', 'imageType', 'modelIds', 'name', 'kvTags',
                              'properties', 'version']

    def __new__(cls, workspace, name=None, id=None, tags=None, properties=None, version=None):
        """Image constructor.

        This class is DEPRECATED. Use the :class:`azureml.core.Environment` class instead.

        Image constructor retrieves a cloud representation of an Image object associated with the
        provided workspace. Returns an instance of a child class corresponding to the specific type of the
        retrieved Image object.

        :param workspace: The workspace object containing the Image to retrieve
        :type workspace: azureml.core.workspace.Workspace
        :param name: The name of the Image to retrieve. Returns the latest version, if it exists
        :type name: str
        :param id: The specific ID of the Image to retrieve. (ID is "<name>:<version>")
        :type id: str
        :param tags: Will filter Image results based on the provided list, by either 'key' or '[key, value]'.
            Ex. ['key', ['key2', 'key2 value']]
        :type tags: builtin.list
        :param properties: Will filter Image results based on the provided list, by either 'key' or '[key, value]'.
            Ex. ['key', ['key2', 'key2 value']]
        :type properties: builtin.list
        :param version: When version and name are both specified, will return the specific version of the Image.
        :type version: str
        :return: An instance of a child of Image corresponding to the specific type of the retrieved Image object
        :rtype: azureml.core.Image
        :raises: azureml.exceptions.WebserviceException
        """
        warnings.warn("Image class has been deprecated and will be removed in a future release. "
                      + "Please migrate to using Environments. "
                      + "https://docs.microsoft.com/en-us/azure/machine-learning/how-to-use-environments",
                      category=DeprecationWarning, stacklevel=2)

        cls.models = None
        cls.workspace = None
        cls._auth = None
        cls._mms_endpoint = None
        cls._operation_endpoint = None
        for key in (cls._expected_payload_keys):
            setattr(cls, key, None)

        if workspace:
            get_response_payload = cls._get(workspace, name, id, tags, properties, version)
            if get_response_payload:
                image_type = get_response_payload['imageType']
                if 'imageFlavor' in get_response_payload:
                    image_flavor = get_response_payload['imageFlavor']
                else:
                    image_flavor = UNKNOWN_IMAGE_FLAVOR

                unknown_image = None
                for child in Image.__subclasses__():
                    if image_type == child._image_type and image_flavor == child._image_flavor:
                        image = super(Image, cls).__new__(child)
                        image._initialize(workspace, get_response_payload)
                        return image
                    elif child._image_type == UNKNOWN_IMAGE_TYPE:
                        unknown_image = super(Image, cls).__new__(child)
                        unknown_image._initialize(workspace, get_response_payload)
                return unknown_image
            else:
                error_message = 'ImageNotFound: Image with '
                if id:
                    error_message += 'ID {}'.format(id)
                else:
                    error_message += 'name {}'.format(name)
                if tags:
                    error_message += ', tags {}'.format(tags)
                if properties:
                    error_message += ', properties {}'.format(properties)
                if version:
                    error_message += ', version {}'.format(version)
                error_message += ' not found in provided workspace'

                raise WebserviceException(error_message)
        return super(Image, cls).__new__(cls)

    def __init__(self, workspace, name=None, id=None, tags=None, properties=None, version=None):
        """Image constructor.

        This class is DEPRECATED. Use the :class:`azureml.core.Environment` class instead.

        Image constructor is used to retrieve a cloud representation of a Image object associated with the
        provided workspace. Will return an instance of a child class corresponding to the specific type of the
        retrieved Image object.

        :param workspace: The workspace object containing the Image to retrieve
        :type workspace: azureml.core.workspace.Workspace
        :param name: The name of the Image to retrieve. Will return the latest version, if it exists
        :type name: str
        :param id: The specific ID of the Image to retrieve. (ID is "<name>:<version>")
        :type id: str
        :param tags: Will filter Image results based on the provided list, by either 'key' or '[key, value]'.
            Ex. ['key', ['key2', 'key2 value']]
        :type tags: builtin.list
        :param properties: Will filter Image results based on the provided list, by either 'key' or '[key, value]'.
            Ex. ['key', ['key2', 'key2 value']]
        :type properties: builtin.list
        :param version: When version and name are both specified, will return the specific version of the Image.
        :type version: str
        :return: An instance of a child of Image corresponding to the specific type of the retrieved Image object
        :rtype: azureml.core.Image
        :raises: azureml.exceptions.WebserviceException
        """
        pass

    def __repr__(self):
        """Return the string representation of the Image object.

        :return: String representation of the Image object
        :rtype: str
        """
        return "{}(workspace={}, name={}, id={}, tags={}, properties={}, " \
               "version={})".format(self.__class__.__name__,
                                    self.workspace.__repr__(),
                                    self.name,
                                    self.id,
                                    self.tags,
                                    self.properties,
                                    self.version)

    def _initialize(self, workspace, obj_dict):
        """Initialize the Image instance.

        This is used because the constructor is used as a getter.

        :param workspace:
        :type workspace: Workspace
        :param obj_dict:
        :type obj_dict: dict
        """
        self._validate_get_payload(obj_dict)

        self.created_time = parse(obj_dict['createdTime'])
        self.creation_state = obj_dict['creationState']
        self.description = obj_dict['description']
        self.id = obj_dict['id']
        self.image_build_log_uri = obj_dict.get('imageBuildLogUri', None)
        self.generated_dockerfile_uri = obj_dict.get('generatedDockerFileUri', None)
        self.image_flavor = obj_dict.get('imageFlavor', None)
        self.image_location = obj_dict['imageLocation']
        self.image_type = obj_dict['imageType']
        self.model_ids = obj_dict['modelIds']
        self.name = obj_dict['name']
        self.tags = obj_dict['kvTags']
        self.properties = obj_dict['properties']
        self.version = obj_dict['version']

        models = []
        if 'modelDetails' in obj_dict:
            models = [Model.deserialize(workspace, model_payload) for model_payload in obj_dict['modelDetails']]
        self.models = models
        self.workspace = workspace
        self._mms_endpoint = _get_mms_url(workspace) + '/images/{}'.format(self.id)
        self._auth = workspace._auth

    @staticmethod
    def _get(workspace, name=None, id=None, tags=None, properties=None, version=None):
        """Get the image with the given filtering criteria.

        :param workspace:
        :type workspace: azureml.core.workspace.Workspace
        :param name:
        :type name: str
        :param id:
        :type id: str
        :param tags:
        :type tags: dict[str, str]
        :param properties:
        :type properties: dict[str, str]
        :param version:
        :type version: str
        :return: azureml.core.Image payload dictionary
        :rtype: dict
        :raises: azureml.exceptions.WebserviceException
        """
        if not name and not id:
            raise WebserviceException('Error, one of id or name must be provided.', logger=module_logger)

        headers = workspace._auth.get_authentication_header()
        params = {'orderBy': 'CreatedAtDesc', 'count': 1, 'expand': 'true'}
        base_endpoint = _get_mms_url(workspace)
        mms_endpoint = base_endpoint + '/images'

        if id:
            image_url = mms_endpoint + '/{}'.format(id)
        else:
            image_url = mms_endpoint
            params['name'] = name
        if tags:
            tags_query = ""
            for tag in tags:
                if type(tag) is list:
                    tags_query = tags_query + tag[0] + "=" + tag[1] + ","
                else:
                    tags_query = tags_query + tag + ","
            tags_query = tags_query[:-1]
            params['tags'] = tags_query
        if properties:
            properties_query = ""
            for prop in properties:
                if type(prop) is list:
                    properties_query = properties_query + prop[0] + "=" + prop[1] + ","
                else:
                    properties_query = properties_query + prop + ","
            properties_query = properties_query[:-1]
            params['properties'] = properties_query
        if version:
            params['version'] = version

        resp = ClientBase._execute_func(get_requests_session().get, image_url, headers=headers, params=params,
                                        timeout=MMS_SYNC_TIMEOUT_SECONDS)

        if resp.status_code == 200:
            content = resp.content
            if isinstance(content, bytes):
                content = content.decode('utf-8')
            image_payload = json.loads(content)
            if id:
                return image_payload
            else:
                paginated_results = get_paginated_results(image_payload, headers)
                if paginated_results:
                    return paginated_results[0]
                else:
                    return None
        elif resp.status_code == 404:
            return None
        else:
            raise WebserviceException('Received bad response from Model Management Service:\n'
                                      'Response Code: {}\n'
                                      'Headers: {}\n'
                                      'Content: {}'.format(resp.status_code, resp.headers, resp.content),
                                      logger=module_logger)

    @staticmethod
    def create(workspace, name, models, image_config):
        """Create an image in the provided workspace.

        :param workspace: The workspace to associate with this image.
        :type workspace: workspace: azureml.core.workspace.Workspace
        :param name: The name to associate with this image.
        :type name: str
        :param models: A list of Model objects to package with this image. Can be an empty list.
        :type models: builtin.list[azureml.core.Model]
        :param image_config: The image config object to use to configure this image.
        :type image_config: azureml.core.image.image.ImageConfig
        :return: The created Image object.
        :rtype: azureml.core.Image
        :raises: azureml.exceptions.WebserviceException
        """
        warnings.warn("Image class has been deprecated and will be removed in a future release. "
                      + "Please migrate to using Environments. "
                      + "https://docs.microsoft.com/en-us/azure/machine-learning/how-to-use-environments",
                      category=DeprecationWarning, stacklevel=2)

        image_name_validation(name)
        model_ids = Model._resolve_to_model_ids(workspace, models, name)

        headers = {'Content-Type': 'application/json'}
        headers.update(workspace._auth.get_authentication_header())
        params = {}
        base_endpoint = _get_mms_url(workspace)
        image_url = base_endpoint + '/images'

        json_payload = image_config.build_create_payload(workspace, name, model_ids)

        print('Creating image')
        resp = ClientBase._execute_func(get_requests_session().post, image_url, params=params, headers=headers,
                                        json=json_payload)
        try:
            resp.raise_for_status()
        except requests.exceptions.HTTPError:
            raise WebserviceException('Received bad response from Model Management Service:\n'
                                      'Response Code: {}\n'
                                      'Headers: {}\n'
                                      'Content: {}'.format(resp.status_code, resp.headers, resp.content),
                                      logger=module_logger)
        if resp.status_code >= 400:
            raise WebserviceException('Error occurred creating image:\n'
                                      'Response Code: {}\n'
                                      'Headers: {}\n'
                                      'Content: {}'.format(resp.status_code, resp.headers, resp.content),
                                      logger=module_logger)

        if 'Operation-Location' in resp.headers:
            operation_location = resp.headers['Operation-Location']
        else:
            raise WebserviceException('Missing response header key: Operation-Location', logger=module_logger)

        create_operation_status_id = operation_location.split('/')[-1]
        operation_url = base_endpoint + '/operations/{}'.format(create_operation_status_id)
        operation_headers = workspace._auth.get_authentication_header()

        operation_resp = ClientBase._execute_func(get_requests_session().get, operation_url, params=params,
                                                  headers=operation_headers, timeout=MMS_SYNC_TIMEOUT_SECONDS)
        try:
            operation_resp.raise_for_status()
        except requests.Timeout:
            raise WebserviceException('Error, request to {} timed out.'.format(operation_url), logger=module_logger)
        except requests.exceptions.HTTPError:
            raise WebserviceException('Received bad response from Model Management Service:\n'
                                      'Response Code: {}\n'
                                      'Headers: {}\n'
                                      'Content: {}'.format(operation_resp.status_code,
                                                           operation_resp.headers,
                                                           operation_resp.content), logger=module_logger)

        content = operation_resp.content
        if isinstance(content, bytes):
            content = content.decode('utf-8')
        operation_content = json.loads(content)
        if 'resourceLocation' in operation_content:
            image_id = operation_content['resourceLocation'].split('/')[-1]
        else:
            raise WebserviceException('Invalid operation payload, missing resourceLocation:\n'
                                      '{}'.format(operation_content), logger=module_logger)

        image = Image(workspace, id=image_id)
        image._operation_endpoint = operation_url
        return image

    def wait_for_creation(self, show_output=False):
        """Wait for the image to finish creating.

        Wait for image creation to reach a terminal state. Will throw a WebserviceException if it reaches a
        non-successful terminal state.

        :param show_output: Boolean option to print more verbose output. Defaults to False.
        :type show_output: bool
        :raises: azureml.exceptions.WebserviceException
        """
        operation_state, error = self._get_operation_state()
        current_state = operation_state

        if show_output:
            sys.stdout.write('{}'.format(current_state))
            sys.stdout.flush()

        while operation_state != 'Cancelled' and operation_state != 'Succeeded' and operation_state != 'Failed' \
                and operation_state != 'TimedOut':
            time.sleep(5)
            operation_state, error = self._get_operation_state()
            if show_output:
                sys.stdout.write('.')
                if operation_state != current_state:
                    sys.stdout.write('\n{}'.format(operation_state))
                    current_state = operation_state
                sys.stdout.flush()
        sys.stdout.write('\n')
        sys.stdout.flush()

        self.update_creation_state()
        if operation_state != 'Succeeded':
            if error and 'statusCode' in error and 'message' in error:
                error_response = ('StatusCode: {}\n'
                                  'Message: {}'.format(error['statusCode'], error['message']))
            else:
                error_response = error

            print('More information about this error is available here: {}\n'
                  'For more help with troubleshooting, see https://aka.ms/debugimage'.format(self.image_build_log_uri))
            raise WebserviceException('Image creation polling reached non-successful terminal state, '
                                      'current state: {}\n'
                                      'Error response from server:\n'
                                      '{}'.format(self.creation_state, error_response), logger=module_logger)

        print('Image creation operation finished for image {}, operation "{}"'.format(self.id, operation_state))

    def _get_operation_state(self):
        """Get the current async operation state for the image.

        :return:
        :rtype: (str, dict)
        """
        if not self._operation_endpoint:
            self.update_deployment_state()
            raise WebserviceException('Long running operation information not known, unable to poll. '
                                      'Current state is {}'.format(self.creation_state), logger=module_logger)

        headers = {'Content-Type': 'application/json'}
        headers.update(self._auth.get_authentication_header())
        params = {}

        resp = ClientBase._execute_func(get_requests_session().get, self._operation_endpoint, headers=headers,
                                        params=params, timeout=MMS_SYNC_TIMEOUT_SECONDS)
        try:
            resp.raise_for_status()
        except requests.exceptions.HTTPError:
            raise WebserviceException('Received bad response from Resource Provider:\n'
                                      'Response Code: {}\n'
                                      'Headers: {}\n'
                                      'Content: {}'.format(resp.status_code, resp.headers, resp.content),
                                      logger=module_logger)
        content = resp.content
        if isinstance(content, bytes):
            content = content.decode('utf-8')
        content = json.loads(content)
        state = content['state']
        error = content['error'] if 'error' in content else None
        return state, error

    def update_creation_state(self):
        """Refresh the current state of the in-memory object.

        Perform an in-place update of the properties of the object based on the current state of the
        corresponding cloud object. Primarily useful for manual polling of creation state.

        :raises: azureml.exceptions.WebserviceException
        """
        headers = {'Content-Type': 'application/json'}
        headers.update(self._auth.get_authentication_header())
        params = {}

        resp = ClientBase._execute_func(get_requests_session().get, self._mms_endpoint, headers=headers, params=params,
                                        timeout=MMS_SYNC_TIMEOUT_SECONDS)

        if resp.status_code == 200:
            image = Image(self.workspace, id=self.id)
            for key in image.__dict__.keys():
                if key != "_operation_endpoint":
                    self.__dict__[key] = image.__dict__[key]
        elif resp.status_code == 404:
            raise WebserviceException('Error: image {} not found:\n'
                                      'Response Code: {}\n'
                                      'Headers: {}\n'
                                      'Content: {}'.format(self.id, resp.status_code, resp.headers, resp.content),
                                      logger=module_logger)
        else:
            raise WebserviceException('Received bad response from Model Management Service:\n'
                                      'Response Code: {}\n'
                                      'Headers: {}\n'
                                      'Content: {}'.format(resp.status_code, resp.headers, resp.content),
                                      logger=module_logger)

    @staticmethod
    def list(workspace, image_name=None, model_name=None, model_id=None, tags=None, properties=None):
        """List the Images associated with the corresponding workspace. Can be filtered with specific parameters.

        :param workspace: The Workspace object to list the Images in.
        :type workspace: azureml.core.workspace.Workspace
        :param image_name: Filter list to only include Images deployed with the specific image name.
        :type image_name: str
        :param model_name: Filter list to only include Images deployed with the specific model name.
        :type model_name: str
        :param model_id: Filter list to only include Images deployed with the specific model ID.
        :type model_id: str
        :param tags: Will filter based on the provided list, by either 'key' or '[key, value]'.
            Ex. ['key', ['key2', 'key2 value']]
        :type tags: builtin.list
        :param properties: Will filter based on the provided list, by either 'key' or '[key, value]'.
            Ex. ['key', ['key2', 'key2 value']]
        :type properties: builtin.list
        :return: A filtered list of Images in the provided workspace.
        :rtype: builtin.list[Images]
        :raises: azureml.exceptions.WebserviceException
        """
        warnings.warn("Image class has been deprecated and will be removed in a future release. "
                      + "Please migrate to using Environments. "
                      + "https://docs.microsoft.com/en-us/azure/machine-learning/how-to-use-environments",
                      category=DeprecationWarning, stacklevel=2)

        headers = workspace._auth.get_authentication_header()
        params = {'expand': 'true'}
        base_url = _get_mms_url(workspace)
        mms_url = base_url + '/images'

        if image_name:
            params['name'] = image_name
        if model_name:
            params['modelName'] = model_name
        if model_id:
            params['modelId'] = model_id
        if tags:
            tags_query = ""
            for tag in tags:
                if type(tag) is list:
                    tags_query = tags_query + tag[0] + "=" + tag[1] + ","
                else:
                    tags_query = tags_query + tag + ","
            tags_query = tags_query[:-1]
            params['tags'] = tags_query
        if properties:
            properties_query = ""
            for prop in properties:
                if type(prop) is list:
                    properties_query = properties_query + prop[0] + "=" + prop[1] + ","
                else:
                    properties_query = properties_query + prop + ","
            properties_query = properties_query[:-1]
            params['properties'] = properties_query
        try:
            resp = ClientBase._execute_func(get_requests_session().get, mms_url, headers=headers, params=params,
                                            timeout=MMS_SYNC_TIMEOUT_SECONDS)
            resp.raise_for_status()
        except requests.Timeout:
            raise WebserviceException('Error, request to Model Management Service timed out to URL:\n'
                                      '{}'.format(mms_url), logger=module_logger)
        except requests.exceptions.HTTPError:
            raise WebserviceException('Received bad response from Model Management Service:\n'
                                      'Response Code: {}\n'
                                      'Headers: {}\n'
                                      'Content: {}'.format(resp.status_code, resp.headers, resp.content),
                                      logger=module_logger)

        content = resp.content
        if isinstance(content, bytes):
            content = content.decode('utf-8')
        image_payload = json.loads(content)
        paginated_results = get_paginated_results(image_payload, headers)

        return [Image.deserialize(workspace, image_dict) for image_dict in paginated_results]

    def update(self, tags):
        """Update the image.

        :param tags: A dictionary of tags to update the image with. Will overwrite any existing tags.
        :type tags: dict[str, str]
        :raises: azureml.exceptions.WebserviceException
        """
        headers = {'Content-Type': 'application/json-patch+json'}
        headers.update(self._auth.get_authentication_header())
        params = {}

        patch_list = []
        self.tags = tags
        patch_list.append({'op': 'replace', 'path': '/kvTags', 'value': self.tags})

        resp = ClientBase._execute_func(get_requests_session().patch, self._mms_endpoint, headers=headers,
                                        params=params, json=patch_list, timeout=MMS_SYNC_TIMEOUT_SECONDS)

        if resp.status_code >= 400:
            raise WebserviceException('Received bad response from Model Management Service:\n'
                                      'Response Code: {}\n'
                                      'Headers: {}\n'
                                      'Content: {}'.format(resp.status_code, resp.headers, resp.content),
                                      logger=module_logger)

    def add_tags(self, tags):
        """Add tags to the image.

        :param tags: A dictionary of tags to add.
        :type tags: dict[str, str]
        :raises: azureml.exceptions.WebserviceException
        """
        headers = {'Content-Type': 'application/json-patch+json'}
        headers.update(self._auth.get_authentication_header())
        params = {}

        patch_list = []
        if self.tags is None:
            self.tags = copy.deepcopy(tags)
        else:
            for key in tags:
                if key in self.tags:
                    print("Replacing tag {} -> {} with {} -> {}".format(key, self.tags[key], key, tags[key]))
                self.tags[key] = tags[key]

        patch_list.append({'op': 'replace', 'path': '/kvTags', 'value': self.tags})

        resp = ClientBase._execute_func(get_requests_session().patch, self._mms_endpoint, headers=headers,
                                        params=params, json=patch_list, timeout=MMS_SYNC_TIMEOUT_SECONDS)

        if resp.status_code >= 400:
            raise WebserviceException('Received bad response from Model Management Service:\n'
                                      'Response Code: {}\n'
                                      'Headers: {}\n'
                                      'Content: {}'.format(resp.status_code, resp.headers, resp.content),
                                      logger=module_logger)

        print('Image tag add operation complete.')

    def remove_tags(self, tags):
        """Remove tags from the image.

        :param tags: A list of keys corresponding to tags to be removed.
        :type tags: builtin.list[str]
        :raises: azureml.exceptions.WebserviceException
        """
        headers = {'Content-Type': 'application/json-patch+json'}
        headers.update(self._auth.get_authentication_header())
        params = {}

        patch_list = []
        if self.tags is None:
            print('Image has no tags to remove.')
            return
        else:
            if type(tags) is not list:
                tags = [tags]
            for key in tags:
                if key in self.tags:
                    del self.tags[key]
                else:
                    print('Tag with key {} not found.'.format(key))

        patch_list.append({'op': 'replace', 'path': '/kvTags', 'value': self.tags})

        resp = ClientBase._execute_func(get_requests_session().patch, self._mms_endpoint, headers=headers,
                                        params=params, json=patch_list, timeout=MMS_SYNC_TIMEOUT_SECONDS)

        if resp.status_code >= 400:
            raise WebserviceException('Received bad response from Model Management Service:\n'
                                      'Response Code: {}\n'
                                      'Headers: {}\n'
                                      'Content: {}'.format(resp.status_code, resp.headers, resp.content),
                                      logger=module_logger)

        print('Image tag remove operation complete.')

    def add_properties(self, properties):
        """Add properties to the image.

        :param properties: A dictionary of properties to add.
        :type properties: dict[str, str]
        :raises: azureml.exceptions.WebserviceException
        """
        check_duplicate_properties(self.properties, properties)

        headers = {'Content-Type': 'application/json-patch+json'}
        headers.update(self._auth.get_authentication_header())
        params = {}

        patch_list = []
        if self.properties is None:
            self.properties = copy.deepcopy(properties)
        else:
            for key in properties:
                self.properties[key] = properties[key]

        patch_list.append({'op': 'add', 'path': '/properties', 'value': self.properties})

        resp = ClientBase._execute_func(get_requests_session().patch, self._mms_endpoint, headers=headers,
                                        params=params, json=patch_list, timeout=MMS_SYNC_TIMEOUT_SECONDS)

        if resp.status_code >= 400:
            raise WebserviceException('Received bad response from Model Management Service:\n'
                                      'Response Code: {}\n'
                                      'Headers: {}\n'
                                      'Content: {}'.format(resp.status_code, resp.headers, resp.content),
                                      logger=module_logger)

        print('Image properties add operation complete.')

    def delete(self):
        """Delete an image from its corresponding workspace.

        .. remarks::

            This method fails if the image has been deployed to a live webservice.

        :raises: azureml.exceptions.WebserviceException
        """
        headers = self._auth.get_authentication_header()
        params = {}

        resp = ClientBase._execute_func(get_requests_session().delete, self._mms_endpoint, headers=headers,
                                        params=params, timeout=MMS_SYNC_TIMEOUT_SECONDS)

        if resp.status_code >= 400:
            if resp.status_code == 412 and "DeletionRequired" in resp.content:
                raise WebserviceException('The image cannot be deleted because it is currently being used in one or '
                                          'more webservices. To know what webservices contain the image, run '
                                          '"Webservice.list(<workspace>, image_id={})"'.format(self.id),
                                          logger=module_logger)

            raise WebserviceException('Received bad response from Model Management Service:\n'
                                      'Response Code: {}\n'
                                      'Headers: {}\n'
                                      'Content: {}'.format(resp.status_code, resp.headers, resp.content),
                                      logger=module_logger)
        else:
            self.creation_state = 'Deleting'

    def serialize(self):
        """Convert this Image object into a JSON serialized dictionary.

        :return: The JSON representation of this Image object.
        :rtype: dict
        """
        created_time = self.created_time.isoformat() if self.created_time else None
        models = [model.serialize() for model in self.models] if self.models else None

        return {'createdTime': created_time, 'creationState': self.creation_state,
                'description': self.description, 'id': self.id,
                'imageBuildLogUri': self.image_build_log_uri, 'imageLocation': self.image_location,
                'generatedDockerFileUri': self.generated_dockerfile_uri,
                'imageType': self.image_type, 'imageFlavor': self.image_flavor,
                'modelIds': self.model_ids, 'modelDetails': models, 'name': self.name,
                'tags': self.tags, 'properties': self.properties,
                'version': self.version, 'workspaceName': self.workspace.name}

    @classmethod
    def deserialize(cls, workspace, image_payload):
        """Convert a json object into a Image object.

        .. remarks::

            This method fails if the provided workspace is not the workspace the image is registered under.

        :param cls: Indicates class method.
        :type cls:
        :param workspace: The workspace object the Image is registered under.
        :type workspace: azureml.core.workspace.Workspace
        :param image_payload: A JSON object to convert to a Image object.
        :type image_payload: dict
        :return: The Image representation of the provided JSON object.
        :rtype: azureml.core.Image
        """
        image_type = image_payload['imageType']
        image_flavor = image_payload['imageFlavor'] if 'imageFlavor' in image_payload else None

        unknown_image = None
        for child in Image.__subclasses__():
            if image_type == child._image_type and image_flavor == child._image_flavor:
                return child._deserialize(workspace, image_payload)
            elif child._image_type == UNKNOWN_IMAGE_TYPE:
                unknown_image = child._deserialize(workspace, image_payload)
        return unknown_image

    @classmethod
    def _deserialize(cls, workspace, image_payload):
        """Convert a JSON object into a Image object.

        :param workspace:
        :type workspace: azureml.core.workspace.Workspace
        :param image_payload:
        :type image_payload: dict
        :return:
        :rtype: azureml.core.Image
        """
        cls._validate_get_payload(image_payload)
        image = cls(None)
        image._initialize(workspace, image_payload)
        return image

    @classmethod
    def _validate_get_payload(cls, payload):
        """Validate the returned image payload.

        :param payload:
        :type payload: dict
        :return:
        :rtype: None
        """
        for payload_key in cls._expected_payload_keys:
            if payload_key not in payload:
                raise WebserviceException('Invalid image payload, missing {} for image:\n'
                                          '{}'.format(payload_key, payload), logger=module_logger)

    @staticmethod
    @abstractmethod
    def image_configuration():
        """Abstract method for creating an image configuration object."""
        pass


class Asset(object):
    """Represents an asset to be used with an Image class.

    This class is DEPRECATED.

    Assets are specified during configuration of an image. For an example, see the ``image_configuration`` method
    of the :class:`azureml.core.image.container.ContainerImage` class.

    :param id: The ID corresponding to the image asset.
    :type id: str
    :param mime_type: The MIME type of the asset.
    :type mime_type: str
    :param unpack: Whether the asset needs to be unpacked as a part of image setup.
    :type unpack: bool
    :param url: A URL pointer to where the asset is stored.
    :type url: str
    """

    _expected_payload_keys = ['id', 'mimeType', 'unpack', 'url']

    def __init__(self, id, mime_type, unpack, url):
        """Initialize the Asset object for images.

        :param id: The ID corresponding to the image asset.
        :type id: str
        :param mime_type: The MIME type of the asset.
        :type mime_type: str
        :param unpack: Whether the asset needs to be unpacked as a part of image setup.
        :type unpack: bool
        :param url: A URL pointer to where the asset is stored.
        :type url: str
        """
        self.id = id
        self.mime_type = mime_type
        self.unpack = unpack
        self.url = url

    def serialize(self):
        """Convert this Asset into a JSON serialized dictionary.

        :return: The JSON representation of this Asset object.
        :rtype: dict
        """
        return {'id': self.id, 'mimeType': self.mime_type, 'unpack': self.unpack, 'url': self.url}

    @staticmethod
    def deserialize(payload_obj):
        """Convert a JSON object into an Asset object.

        :param payload_obj: A JSON object to convert to an Asset object.
        :type payload_obj: dict
        :return: The Asset representation of the provided JSON object.
        :rtype: Asset
        :raises: azureml.exceptions.WebserviceException
        """
        for payload_key in Asset._expected_payload_keys:
            if payload_key not in payload_obj:
                raise WebserviceException('Invalid image payload, missing {} for asset:\n'
                                          '{}'.format(payload_key, payload_obj), logger=module_logger)

        return Asset(payload_obj['id'], payload_obj['mimeType'], payload_obj['unpack'], payload_obj['url'])


class ImageConfig(ABC):
    """Defines the abstract class for image configuration objects.

    This class is DEPRECATED.

    .. remarks::

        The ImageConfig class is one of a set of classes that are designed to facilitate deploying models in Azure.

        One way to deploy a model that you've trained is to package it as an image (e.g., a Docker image) containing
        the dependencies needed to run the model. An Image configuration is used to specify key information about the
        image (such as conda environment info and execution scripts). The ImageConfig class is the abstract class that
        all such configuration objects will inherit from. For example, the
        :class:`azureml.core.image.container.ContainerImageConfig` class inherits from the ImageConfig class.

        For an overview on deploying models in Azure, see `https://aka.ms/azureml-how-deploy`.
    """

    @abstractmethod
    def build_create_payload(self, workspace, name, model_ids):
        """Abstract method for building the creation payload associated with this configuration object.

        .. remarks::

            See the :class:`azureml.core.image.container.ContainerImageConfig` class for an example of a
            concrete instantiation of this abstract method.

        :param workspace: The workspace associated with the image.
        :type workspace: azureml.core.workspace.Workspace
        :param name: The name of the image.
        :type name: str
        :param model_ids: Specifies list of model IDs, corresponding to models to be packaged with the image.
        :type model_ids: builtin.list[str]
        :return: The creation payload to use for Image creation.
        :rtype: dict
        """
        pass

    @abstractmethod
    def validate_configuration(self):
        """Check that the specified configuration values are valid.

        .. remarks::

            See the :class:`azureml.core.image.container.ContainerImageConfig` class for an example of a
            concrete instantiation of this abstract method.

            Raises a :class:`azureml.exceptions.WebserviceException` if validation fails.

        :raises: azureml.exceptions.WebserviceException
        """
        pass


class TargetRuntime(object):
    """Represents properties, runtime type, and other settings used in an Image.

    This class is DEPRECATED.

    Some TargetRuntime properties are specified during configuration of an image. For examples, see the
    ``image_configuration`` method of the :class:`azureml.core.image.container.ContainerImage` class.

    :param properties: A dictionary of properties associated with the target runtime.
    :type properties: dict[str, str]
    :param runtime_type: A string representation of the runtime type.
    :type runtime_type: str
    """

    _expected_payload_keys = ['properties', 'runtimeType']

    def __init__(self, properties, runtime_type, targetArchitecture):
        """Initialize the TargetRuntime object.

        :param properties: A dictionary of properties associated with the target runtime.
        :type properties: dict[str, str]
        :param runtime_type: A string representation of the runtime type.
        :type runtime_type: str
        """
        self.properties = properties
        self.runtime_type = runtime_type
        self.targetArchitecture = targetArchitecture

    def serialize(self):
        """Convert this TargetRuntime into a JSON serialized dictionary.

        :return: The JSON representation of this TargetRuntime object.
        :rtype: dict
        """
        return {'properties': self.properties,
                'runtimeType': self.runtime_type,
                'targetArchitecture': self.targetArchitecture}

    @staticmethod
    def deserialize(payload_obj):
        """Convert a JSON object into a TargetRuntime object.

        :param payload_obj: A JSON object to convert to a TargetRuntime object.
        :type payload_obj: dict
        :return: The TargetRuntime representation of the provided JSON object.
        :rtype: TargetRuntime
        :raises: azureml.exceptions.WebserviceException
        """
        for payload_key in TargetRuntime._expected_payload_keys:
            if payload_key not in payload_obj:
                raise WebserviceException('Invalid image payload, missing {} for targetRuntime:\n'
                                          '{}'.format(payload_key, payload_obj), logger=module_logger)

        targetArchitecture = ARCHITECTURE_AMD64
        if 'targetArchitecture' in payload_obj:
            targetArchitecture = payload_obj['targetArchitecture']

        return TargetRuntime(payload_obj['properties'], payload_obj['runtimeType'], targetArchitecture)

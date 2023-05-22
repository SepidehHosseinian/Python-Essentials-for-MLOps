# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""Contains functionality for managing unknown Webservices in Azure Machine Learning."""

from azureml._model_management._constants import UNKNOWN_WEBSERVICE_TYPE
from azureml.core.webservice import Webservice


class UnknownWebservice(Webservice):
    """For internal use only.

    This class is used by the :class:`azureml.core.webservice.webservice.Webservice` class to get or
    list service subtypes when the web service was created from a package that wasn't imported,
    for example, for a service created with the :mod:`azureml.accel` package.
    """

    # _expected_payload_keys inherited from Webservice

    _webservice_type = UNKNOWN_WEBSERVICE_TYPE

    def _initialize(self, workspace, obj_dict):
        """Initialize the Webservice instance.

        This is used because the constructor is used as a getter.

        :param workspace:
        :type workspace: azureml.core.Workspace
        :param obj_dict:
        :type obj_dict: dict
        :return:
        :rtype: None
        """
        # Validate obj_dict with _expected_payload_keys from parent Webservice
        UnknownWebservice._validate_get_payload(obj_dict)

        # Initialize common Webservice attributes
        super(UnknownWebservice, self)._initialize(workspace, obj_dict)

        # Initialize optional UnknownWebservice specific attributes (based on the BrainwaveWebservice)
        mms_sdk_map = {
            'ipAddress': 'ip_address',
            'port': 'port',
            'numReplicas': 'num_replicas',
            'sslEnabled': 'ssl'
        }
        for key in mms_sdk_map:
            sdk_key = mms_sdk_map[key]
            self.__dict__[sdk_key] = obj_dict[key] if key in obj_dict else None

        # check to get scoring URI
        if self.ip_address and self.port:
            self.scoring_uri = "{}:{}".format(self.ip_address, self.port)

    @staticmethod
    def _deploy(workspace, name, image, deployment_config, deployment_target):
        """Deploy the Webservice.

        :param workspace:
        :type workspace: azureml.core.Workspace
        :param name:
        :type name: str
        :param image:
        :type image: azureml.core.Image
        :param deployment_config:
        :type deployment_config: WebserviceDeploymentConfiguration
        :param deployment_target:
        :type deployment_target: azureml.core.ComputeTarget
        :return:
        :rtype: Webservice
        """
        raise NotImplementedError("Cannot deploy Webservice because webservice type is not known.")

    def run(self, input):
        """Call this Webservice with the provided input.

        .. note::
            Not implemented.

        :param input: The input to call the Webservice with.
        :type input: varies
        :return: The result of calling the Webservice.
        :rtype: dict
        :raises: NotImplementedError
        """
        raise NotImplementedError("Cannot run Webservice because webservice type is not known.")

    def update(self, *args):
        """Update the Webservice. Possible options to update vary based on Webservice type.

        .. note::
            Not implemented.

        :param args: Values to update
        :type args: varies
        :raises: NotImplementedError
        """
        raise NotImplementedError("Cannot update Webservice because webservice type is not known.")

    def get_token(self):
        """Retrieve auth token for this Webservice, scoped to the current user.

        .. note::
            Not implemented.

        :return: The auth token for this Webservice and when it should be refreshed after.
        :rtype: str, datetime
        :raises: NotImplementedError
        """
        raise NotImplementedError("Unknown webservices do not support Token Authentication.")

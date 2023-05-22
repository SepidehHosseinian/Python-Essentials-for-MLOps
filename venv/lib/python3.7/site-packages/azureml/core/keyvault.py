# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Contains functionality for managing secrets in the Key Vault associated with an Azure Machine Learning workspace.

This module contains convenience methods for adding, retrieving, deleting and listing secrets from the
[Azure Key Vault](https://docs.microsoft.com/azure/key-vault/key-vault-overview) associated with a workspace.
"""
import logging
from azureml._restclient.secrets_client import SecretsClient
from azureml._restclient.models.credential_enums import KeyVaultContentType
module_logger = logging.getLogger(__name__)


class Keyvault(object):
    """Manages secrets stored in the Azure Key Vault associated with an Azure Machine Learning workspace.

    Each Azure Machine Learning workspace has an associated `Azure Key
    Vault <https://docs.microsoft.com/azure/key-vault/key-vault-overview>`_.
    The Keyvault class is a simplified wrapper of the Azure Key Vault that allows you to manage
    secrets in the key vault including setting, retrieving, deleting, and listing secrets. Use
    the Keyvault class to pass secrets to remote runs securely without exposing sensitive information
    in cleartext.

    For more information, see `Using secrets in training
    runs <https://docs.microsoft.com/azure/machine-learning/how-to-use-secrets-in-runs>`_.

    .. remarks::

        In submitted runs on local and remote compute, you can use the :meth:`azureml.core.Run.get_secret`
        method of the Run instance to get the secret value from Key Vault. To get multiple secrets, use
        the :meth:`azureml.core.Run.get_secrets` method of the Run instance.

        These Run methods gives you a simple shortcut because the Run instance is aware of its Workspace and
        Keyvault, and can directly obtain the secret without the need to instantiate the Workspace and
        Keyvault within the remote run.

        The following example shows how to access the default key vault associated with
        a workspace and set a secret.

        .. code-block:: python

            import uuid

            local_secret = os.environ.get("LOCAL_SECRET", default = str(uuid.uuid4())) # Use random UUID as a substitute for real secret.
            keyvault = ws.get_default_keyvault()
            keyvault.set_secret(name="secret-name", value = local_secret)

        Full sample is available from
        https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/manage-azureml-service/authentication-in-azureml/authentication-in-azureml.ipynb


    :param workspace: The Azure Machine Learning Workspace associated with this key vault.
    :type workspace: azureml.core.Workspace
    """

    def __init__(self, workspace):
        """Class Keyvault constructor.

        :param workspace: The Azure Machine Learning Workspace associated with this key vault.
        :type workspace: azureml.core.Workspace
        """
        self.workspace = workspace

    def set_secret(self, name, value, content_type=KeyVaultContentType.not_provided):
        """Add a secret to the Azure Key Vault associated with the workspace.

        :param name: The name of the secret to add.
        :type name: str
        :param value: The value of the secret to add.
        :type value: str
        :param value: The content type of the secret to add.
        :type value: azureml.core.azureml._restclient.models.KeyVaultContentType
        :return:
        :rtype: None
        """
        Keyvault._client(self.workspace)._add_secret(name, value, content_type)

    def set_secrets(self, secrets_batch):
        """Add the dictionary of secrets to the Azure Key Vault associated with the workspace.

        :param secrets_batch: A dictionary of secret names and values to add.
        :type secrets_batch: dict(str:str)
        :return:
        :rtype: None
        """
        Keyvault._client(self.workspace)._add_secrets(secrets_batch)

    def get_secret(self, name):
        """Return the secret value for a given secret name.

        :param name: The secret name to return the value for.
        :type name: str
        :return: The secret value for a specified secret name.
        :rtype: str
        """
        return Keyvault._client(self.workspace)._get_secret(name)

    def get_secrets(self, secrets):
        """Return the secret values for a given list of secret names.

        :param secrets: The list of secret names to retrieve values for.
        :type secrets: builtin.list[str]
        :return: A dictionary of found and not found secrets.
        :rtype: dict(str: str)
        """
        return Keyvault._client(self.workspace)._get_secrets(secrets)

    def get_secret_content_type(self, name):
        """Return the secret's content type for a given secret name.

        :param name: The secret name to return the content type for.
        :type name: str
        :return: The secret content type for a specified secret name.
        :rtype: str
        """
        return Keyvault._client(self.workspace)._get_secret_content_type(name)

    def delete_secret(self, name):
        """Delete the secret with the specified name.

        :param name: The name of the secret to delete.
        :type name: str
        :return:
        :rtype: None
        """
        Keyvault._client(self.workspace)._delete_secret(name)

    def delete_secrets(self, secrets):
        """Delete a list of secrets from the Azure Key Vault associated with the workspace.

        :param secrets_batch: The list of secrets to delete.
        :type secrets_batch: builtin.list[str]
        :return:
        :rtype: None
        """
        Keyvault._client(self.workspace)._delete_secrets(secrets)

    def list_secrets(self):
        """Return the list of secret names from the Azure Key Vault associated with the workspace.

        This method does not return the secret values.

        :return: A list of dictionary of secret names with format {name : "secretName"}
        :rtype: dict(str:str)
        """
        return Keyvault._client(self.workspace)._list_secrets()

    @staticmethod
    def _client(workspace):
        """Get a client.

        :return: Returns the client
        :rtype: SecretsClient
        """
        secrets_client = SecretsClient(workspace.service_context)
        return secrets_client

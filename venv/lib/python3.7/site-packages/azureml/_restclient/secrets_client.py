# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Access SecretsClient"""

from .models.credential_dto import CredentialDto
from .models.credential_enums import KeyVaultContentType
from .models.credential_batch_dto import CredentialBatchDto
from .workspace_client import WorkspaceClient


class SecretsClient(WorkspaceClient):
    """Secrets client class"""

    def get_rest_client(self, user_agent=None):
        """get service rest client"""
        return self._service_context._get_credential_restclient(user_agent=user_agent)

    def _add_secret(self, name, value, content_type=KeyVaultContentType.not_provided):
        """
        :param name:
        :type name: str
        :param value:
        :type value: str
        :param value:
        :type value: KeyVaultContentType
        :return Add the secret to the keyvault assosciated with your workspace:
        """
        credential_dto = CredentialDto(name=name, value=value, content_type=content_type)
        self._execute_with_workspace_arguments(
            self._client.credential.create, credential_dto=credential_dto)

    def _add_secrets(self, secrets):
        """
        :param secrets_batch:
        :type secrets_batch: dict
        :return Add the list of specified secrets and their values to the keyvault assosciated with your workspace:
        """
        secrets_list = []
        for key in secrets:
            credential_dto = CredentialDto(name=key, value=secrets[key])
            secrets_list.append(credential_dto)

        credential_batch_dto = CredentialBatchDto(secrets=secrets_list)
        self._execute_with_workspace_arguments(self._client.credential.create_batch,
                                               credential_batch_dto=credential_batch_dto)

    def _get_secret(self, name):
        """
        :param secret_name:
        :type name: str
        :return Return the secret value for a given name:
        """
        credential_dto = self._execute_with_workspace_arguments(
            self._client.credential.get, credential_name=name)

        return credential_dto.value

    def _get_secrets(self, secrets):
        """
        :param secrets_batch:
        :type name: dict
        :return Returns the dict of found and not found secrets for a batch of secret names specified:
        """
        secrets_list = []
        for secret in secrets:
            credential_dto = CredentialDto(name=secret)
            secrets_list.append(credential_dto)

        credential_batch_dto = CredentialBatchDto(secrets=secrets_list)
        batch_retrieved = self._execute_with_workspace_arguments(
            self._client.credential.get_batch, credential_batch_dto=credential_batch_dto)

        found_secrets = batch_retrieved.secrets
        not_found_secrets = batch_retrieved.not_found_secrets

        secrets_dict = {secret.name: secret.value for secret in found_secrets}
        secrets_dict.update({missing_secret: None for missing_secret in not_found_secrets})

        return secrets_dict

    def _get_secret_content_type(self, name):
        """
        :param name:
        :type name: str
        :return Return the secret content type for a given name:
        """
        credential_dto = self._execute_with_workspace_arguments(
            self._client.credential.get, credential_name=name)

        return credential_dto.content_type

    def _delete_secret(self, name):
        """
        :param name:
        :type name: string
        :return Delete the secret with the name specified:
        """
        self._execute_with_workspace_arguments(
            self._client.credential.delete, credential_name=name)

    def _delete_secrets(self, secrets):
        """
        :param secrets:
        :type name: list
        :return Delete the list of secrets with the name specified:
        """
        for secret in secrets:
            self._execute_with_workspace_arguments(
                self._client.credential.delete, credential_name=secret)

    def _list_secrets(self):
        """
        :return: List all user secret keys in the workspace
        """
        secret_items = self._execute_with_workspace_arguments(self._client.credential.list,
                                                              is_paginated=True)

        return [secret.as_dict() for secret in secret_items]

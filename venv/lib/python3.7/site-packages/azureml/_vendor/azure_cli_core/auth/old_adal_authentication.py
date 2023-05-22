# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for license information.
# --------------------------------------------------------------------------------------------
import requests
import time
from typing import TYPE_CHECKING

import socket
import logging

from azureml._common._core_user_error.user_error import CredentialsExpireInactivity, AccountConfigurationChanged, \
    CredentialExpiredPasswordChanged, CertificateVerificationFailure, NetworkConnectionFailed
from azureml._common._error_definition import AzureMLError
from msrest.authentication import Authentication

from azureml._vendor.azure_cli_core.util import in_cloud_console
from azureml._vendor.azure_cli_core.auth.util import scopes_to_resource
from azureml.exceptions import AuthenticationException, AzureMLException

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from typing import NamedTuple
    AccessToken = NamedTuple("AccessToken", [("token", str), ("expires_on", int)])
else:
    from collections import namedtuple
    AccessToken = namedtuple("AccessToken", ["token", "expires_on"])


class AdalAuthentication(Authentication):  # pylint: disable=too-few-public-methods

    def __init__(self, token_retriever, external_tenant_token_retriever=None):
        self._token_retriever = token_retriever
        self._external_tenant_token_retriever = external_tenant_token_retriever

    def _get_token(self, sdk_resource=None):
        """
        :param sdk_resource: `resource` converted from Track 2 SDK's `scopes`
        """
        external_tenant_tokens = None
        try:
            scheme, token, full_token = self._token_retriever(sdk_resource)
            if self._external_tenant_token_retriever:
                external_tenant_tokens = self._external_tenant_token_retriever(sdk_resource)
        except Exception as err:
            # pylint: disable=no-member
            if in_cloud_console():
                AdalAuthentication._log_hostname()
                raise err

            err = (getattr(err, 'error_response', None) or {}).get('error_description') or str(err)
            if 'AADSTS70008' in err:  # all errors starting with 70008 should be creds expiration related
                message = "Please run 'az login'" if not in_cloud_console() else ''
                azureml_error = AzureMLError.create(
                    CredentialsExpireInactivity, message=message
                )
                raise AzureMLException._with_error(azureml_error)
            if 'AADSTS50079' in err:
                message = "Please run 'az login'" if not in_cloud_console() else ''
                azureml_error = AzureMLError.create(
                    AccountConfigurationChanged, message=message
                )
                raise AzureMLException._with_error(azureml_error)
            if 'AADSTS50173' in err:
                message = "Please clear browser's cookies and run 'az login'" if not in_cloud_console() else ''
                azureml_error = AzureMLError.create(
                    CredentialExpiredPasswordChanged, message=message
                )
                raise AzureMLException._with_error(azureml_error)

            raise AzureMLException(err)
        except requests.exceptions.SSLError:
            azureml_error = AzureMLError.create(
                CertificateVerificationFailure
            )
            raise AzureMLException._with_error(azureml_error)
        except requests.exceptions.ConnectionError as err:
            azureml_error = AzureMLError.create(
                NetworkConnectionFailed, error=str(err)
            )
            raise AzureMLException._with_error(azureml_error)

        return scheme, token, full_token, external_tenant_tokens

    def get_all_tokens(self, *scopes):
        scheme, token, full_token, external_tenant_tokens = self._get_token(_try_scopes_to_resource(scopes))
        return scheme, token, full_token, external_tenant_tokens

    # This method is exposed for Azure Core.
    def get_token(self, *scopes, **kwargs):  # pylint:disable=unused-argument
        logger.debug("AdalAuthentication.get_token invoked by Track 2 SDK with scopes=%s", scopes)

        _, token, full_token, _ = self._get_token(_try_scopes_to_resource(scopes))

        # NEVER use expiresIn (expires_in) as the token is cached and expiresIn will be already out-of date
        # when being retrieved.

        # User token entry sample:
        # {
        #     "tokenType": "Bearer",
        #     "expiresOn": "2020-11-13 14:44:42.492318",
        #     "resource": "https://management.core.windows.net/",
        #     "userId": "test@azuresdkteam.onmicrosoft.com",
        #     "accessToken": "eyJ0eXAiOiJKV...",
        #     "refreshToken": "0.ATcAImuCVN...",
        #     "_clientId": "04b07795-8ddb-461a-bbee-02f9e1bf7b46",
        #     "_authority": "https://login.microsoftonline.com/54826b22-38d6-4fb2-bad9-b7b93a3e9c5a",
        #     "isMRRT": True,
        #     "expiresIn": 3599
        # }

        # Service Principal token entry sample:
        # {
        #     "tokenType": "Bearer",
        #     "expiresIn": 3599,
        #     "expiresOn": "2020-11-12 13:50:47.114324",
        #     "resource": "https://management.core.windows.net/",
        #     "accessToken": "eyJ0eXAiOiJKV...",
        #     "isMRRT": True,
        #     "_clientId": "22800c35-46c2-4210-b8a7-d8c3ec3b526f",
        #     "_authority": "https://login.microsoftonline.com/54826b22-38d6-4fb2-bad9-b7b93a3e9c5a"
        # }
        if 'expiresOn' in full_token:
            import datetime
            expires_on_timestamp = int(_timestamp(
                datetime.datetime.strptime(full_token['expiresOn'], '%Y-%m-%d %H:%M:%S.%f')))
            return AccessToken(token, expires_on_timestamp)

        # Cloud Shell (Managed Identity) token entry sample:
        # {
        #     "access_token": "eyJ0eXAiOiJKV...",
        #     "refresh_token": "",
        #     "expires_in": "2106",
        #     "expires_on": "1605686811",
        #     "not_before": "1605682911",
        #     "resource": "https://management.core.windows.net/",
        #     "token_type": "Bearer"
        # }
        if 'expires_on' in full_token:
            return AccessToken(token, int(full_token['expires_on']))

        from azureml._vendor.azure_cli_core.azclierror import CLIInternalError
        raise CLIInternalError("No expiresOn or expires_on is available in the token entry.")

    # This method is exposed for msrest.
    def signed_session(self, session=None, sdk_resource=None):  # pylint: disable=arguments-differ
        session = session or super(AdalAuthentication, self).signed_session()
        external_tenant_tokens = None
        try:
            scheme, token, _ = self._token_retriever(sdk_resource)
            if self._external_tenant_token_retriever:
                external_tenant_tokens = self._external_tenant_token_retriever(sdk_resource)
        except AuthenticationException as err:
            if in_cloud_console():
                AdalAuthentication._log_hostname()
            raise err
        except requests.exceptions.ConnectionError as err:
            raise AuthenticationException('Please ensure you have network connection. Error detail: ' + str(err))
        except Exception as err:
            # pylint: disable=no-member
            if in_cloud_console():
                AdalAuthentication._log_hostname()
            if 'AADSTS70008:' in (getattr(err, 'error_response', None) or {}).get('error_description') or '':
                raise AuthenticationException("Credentials have expired due to inactivity.{}".format(
                    " Please run 'az login'" if not in_cloud_console() else ''))

            raise AuthenticationException('Authentication Error.', inner_exception=err)

        header = "{} {}".format(scheme, token)
        session.headers['Authorization'] = header
        if external_tenant_tokens:
            aux_tokens = ';'.join(['{} {}'.format(scheme2, tokens2) for scheme2, tokens2, _ in external_tenant_tokens])
            session.headers['x-ms-authorization-auxiliary'] = aux_tokens
        return session

    @staticmethod
    def _log_hostname():
        logger.warning("A Cloud Shell credential problem occurred. When you report the issue with the error "
                       "below, please mention the hostname '%s'", socket.gethostname())


class BasicTokenCredential:
    # pylint:disable=too-few-public-methods
    """A Track 2 implementation of msrest.authentication.BasicTokenAuthentication."""
    def __init__(self, access_token):
        self.access_token = access_token
        self.scheme = 'Bearer'

    def get_token(self, *scopes, **kwargs):  # pylint:disable=unused-argument
        return AccessToken(self.access_token, int(time.time() + 3600))

    def signed_session(self, session=None):
        session = session or requests.Session()
        header = "{} {}".format(self.scheme, self.access_token)
        session.headers['Authorization'] = header
        return session


def _timestamp(dt):
    # datetime.datetime can't be patched:
    #   TypeError: can't set attributes of built-in/extension type 'datetime.datetime'
    # So we wrap datetime.datetime.timestamp with this function.
    # https://docs.python.org/3/library/unittest.mock-examples.html#partial-mocking
    # https://williambert.online/2011/07/how-to-unit-testing-in-django-with-mocking-and-patching/
    return dt.timestamp()


def _try_scopes_to_resource(scopes):
    """Wrap scopes_to_resource to workaround some SDK issues."""

    # Track 2 SDKs generated before https://github.com/Azure/autorest.python/pull/239 don't maintain
    # credential_scopes and call `get_token` with empty scopes.
    # As a workaround, return None so that the CLI-managed resource is used.
    if not scopes:
        logger.debug("No scope is provided by the SDK, use the CLI-managed resource.")
        return None

    # Track 2 SDKs generated before https://github.com/Azure/autorest.python/pull/745 extend default
    # credential_scopes with custom credential_scopes. Instead, credential_scopes should be replaced by
    # custom credential_scopes. https://github.com/Azure/azure-sdk-for-python/issues/12947
    # As a workaround, remove the first one if there are multiple scopes provided.
    if len(scopes) > 1:
        logger.debug("Multiple scopes are provided by the SDK, discarding the first one: %s", scopes[0])
        return scopes_to_resource(scopes[1:])

    # Exactly only one scope is provided
    return scopes_to_resource(scopes)

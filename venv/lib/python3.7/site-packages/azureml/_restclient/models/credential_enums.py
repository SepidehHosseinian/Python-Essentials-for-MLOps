# coding=utf-8
# --------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from enum import Enum


class KeyVaultContentType(str, Enum):
    """Contains enumeration values describing the content type of the key vault secret.
    """
    not_provided='application/NotProvided'
    storage_account_access_key='application/vnd.ms-StorageAccountAccessKey'
    storage_account_server_side_encryption_key='application/vnd.ms-StorageAccountServerSideEncryptionKey'
    storage_sas_token='application/vnd.ms-StorageSAStoken'
    storage_connection_string='application/vnd.ms-StorageConnectionString'
    client_side_encryption_key='application/vnd.ms-ClientSideEncryptionKey'
    password='application/vnd.ms-password' #[SuppressMessage("Microsoft.Security", "CS001:SecretInLine", Justification="Metadata to describe a password but not a password")]
    servicebus_sas_token='application/vnd.ms-ServiceBusSAStoken'
    servicebus_connection_string='application/vnd.ms-ServiceBusConnectionString'
    sql_connection_string='application/vnd.ms-SQLConnectionString'
    pem_file='application/x-pem-file'
    bek='BEK'
    wrapped_bek='Wrapped BEK'
    pkcs12='application/x-pkcs12'
    text_configuration_value='application/vnd.bag-textConfigurationValue'
    third_party_hosted_secret_no_rotation='application/vnd.bag-3rdPartyHostedSecretNoRotation'
    standard_user_account_test='application/vnd.bag-StandardUserAccountTest'
    strong_enc_connection_string='application/vnd.bag-StrongEncConnectionString'
    strong_enc_password_string='application/vnd.bag-StrongEncPasswordString'
    azure_dev_ops_pat='application/vnd.bag-AzureDevOpsPAT'

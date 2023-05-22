# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import os
import sys

_ENV_AZ_INSTALLER = 'AZ_INSTALLER'
_AZUREML_AUTH_CONFIG_DIR_ENV_NAME = 'AZUREML_AUTH_CONFIG_DIR'


def get_config_dir():
    """Folder path for azureml-core to store authentication config"""
    _AUTH_FOLDER_PATH = os.path.expanduser(os.path.join('~', '.azureml', "auth"))
    if os.getenv(_AZUREML_AUTH_CONFIG_DIR_ENV_NAME, None):
        return os.getenv(_AZUREML_AUTH_CONFIG_DIR_ENV_NAME, None)
    else:
        if sys.version_info > (3, 0):
            os.makedirs(_AUTH_FOLDER_PATH, exist_ok=True)
        else:
            if not os.path.exists(_AUTH_FOLDER_PATH):
                os.makedirs(_AUTH_FOLDER_PATH)

        return _AUTH_FOLDER_PATH


def get_az_config_dir():
    return os.getenv('AZURE_CONFIG_DIR', None) or os.path.expanduser(os.path.join('~', '.azure'))

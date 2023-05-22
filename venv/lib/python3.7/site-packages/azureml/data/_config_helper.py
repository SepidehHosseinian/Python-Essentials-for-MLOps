# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
import logging
import re

from azureml.data.constants import _PROHIBITED_NAMES


def _validate_name(name, mode):
    from azureml.exceptions import UserErrorException

    if not name:
        raise UserErrorException('Invalid {} name. Name cannot be empty or None, please use a valid name.')

    if name.lower() in _PROHIBITED_NAMES:
        error_message = ('{} name "{}" has the potential to collide with common '.format(mode.capitalize(), name)
                         + 'environment variable names and can lead to unexpected behavior. Please consider using a '
                         'different name. This will result in an error in future versions of the SDK.')
        logging.getLogger().warning(error_message)

    if re.search(r"^[a-zA-Z_]+[a-zA-Z0-9_]*$", name):
        return name

    raise UserErrorException(
        'Invalid {} name "{}". The name can only be alphanumeric characters and underscore, '.format(mode, name)
        + 'and must not begin with a number.'
    )

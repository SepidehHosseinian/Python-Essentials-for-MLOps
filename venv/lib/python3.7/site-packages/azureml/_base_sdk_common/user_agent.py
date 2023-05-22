# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import json
import logging
import os

module_logger = logging.getLogger(__name__)


class UserAgentProduct(object):
    """
    :param name(str): A product identifier.
    :param version(str): A version number of the product.
    :param comment(str): Comment contains sub product information.
    """

    def __init__(self, name, version, comment):
        self._name = name
        self._version = version
        self._comment = comment

    @property
    def name(self):
        return self._name

    @property
    def version(self):
        return self._version

    @property
    def comment(self):
        return self._comment

    def __str__(self):
        result = self.name
        if self.version is not None:
            result += "/{}".format(self.version)
        if self.comment is not None:
            result += " ({})".format(self.comment)
        return result


def get_client_info():
    try:
        info_path = os.path.abspath(os.path.join(os.path.expanduser("~"), ".azureml", "clientinfo.json"))
        module_logger.debug("Fetching client info from %s", info_path)
        with open(info_path, 'rt') as cif:
            client_info = json.load(cif)
        module_logger.debug("Loaded client info as %s : %s", client_info['name'], client_info['version'])
        return [UserAgentProduct(client_info['name'], client_info['version'], None)]
    except Exception as e:  # noqa
        module_logger.debug("Error loading client info: %s", str(e))
        return []


user_agent_product_list = get_client_info()


def append(product_name, product_version=None, comment=None):
    product = UserAgentProduct(product_name, product_version, comment)
    user_agent_product_list.append(product)


def get_user_agent():
    return " ".join(str(product) for product in user_agent_product_list)

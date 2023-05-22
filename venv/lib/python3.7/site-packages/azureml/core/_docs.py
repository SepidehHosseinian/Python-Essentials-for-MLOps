# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import logging


module_logger = logging.getLogger(__name__)

# TODO: Infer this somehow - embedded package names?
_namespace_prefix_to_package_mapping = {
    'azureml.core': "azureml-core",
    'azureml.data': "azureml-core",
}

DOCS_URL_ROOT = 'https://docs.microsoft.com/en-us/python/api'
OVERVIEW_PATH = 'overview/azure/ml/intro'
DOCS_VIEW_FILTER = '?view=azure-ml-py'


def _warn_on_preview_url(url):
    """
    :param url:
    :type url: str
    """
    # type: (str) -> bool
    if "review.docs.microsoft.com" in url or "&branch=" in url:
        logging.warn("URL is not final public url")


def _type_to_full_type(intype):
    """
    :param intype:
    :type intype: list
    :return:
    :rtype: dict
    """
    ret = "{0}.{1}".format(intype.__module__, intype.__name__)
    module_logger.debug("%s fully qualified type: %s", intype, ret)
    return ret


def _package_to_url_path(package_name):
    """
    :param package_name:
    :type package_name: str
    :return:
    :rtype: str
    """
    return package_name


def get_docs_url(intype):
    """
    :param intype:
    :type intype: list
    :return: Returns the docs url.
    :rtype: str
    """
    full_type = _type_to_full_type(intype)

    api_suffix = OVERVIEW_PATH
    for known_prefix, pkg_name in _namespace_prefix_to_package_mapping.items():
        if full_type.startswith(known_prefix):
            # pkg_name/fqtn
            api_suffix = "{0}/{1}".format(_package_to_url_path(pkg_name), full_type)
            break

    full_url = "{0}/{1}{2}".format(DOCS_URL_ROOT, api_suffix, DOCS_VIEW_FILTER)
    _warn_on_preview_url(full_url)
    return full_url

# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for license information.
# ---------------------------------------------------------

"""_restclient helper function utilities"""
import re
import requests
from requests.adapters import HTTPAdapter
from urllib3 import Retry
from .constants import (PAGE_SIZE_KEY, TOP_KEY, CALLER_KEY, CUSTOM_HEADERS_KEY, RequestHeaders,
                        FILTER_KEY, ORDER_BY_KEY, QUERY_PARAMS_KEY, VIEW_TYPE_KEY)
from .contracts.query_params import create_query_params, create_experiment_query_params


# https://stackoverflow.com/questions/19053707
def snake_to_camel(text):
    """convert snake name to camel"""
    return re.sub('_([a-zA-Z0-9])', lambda m: m.group(1).upper(), text)


def camel_case_transformer(key, attr_desc, value):
    """transfer string to camel case"""
    return (snake_to_camel(key), value)


def create_session_with_retry(retry=3):
    """
    Create requests.session with retry

    :type retry: int
    rtype: Response
    """
    retry_policy = get_retry_policy(num_retry=retry)

    session = requests.Session()
    session.mount('https://', HTTPAdapter(max_retries=retry_policy))
    session.mount('http://', HTTPAdapter(max_retries=retry_policy))
    return session


def get_retry_policy(num_retry=3):
    """
    :return: Returns the msrest or requests REST client retry policy.
    :rtype: urllib3.Retry
    """
    status_forcelist = [413, 429, 500, 502, 503, 504]
    backoff_factor = 0.4
    retry_policy = Retry(
        total=num_retry,
        read=num_retry,
        connect=num_retry,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        # By default this is True. We set it to false to get the full error trace, including url and
        # status code of the last retry. Otherwise, the error message is 'too many 500 error responses',
        # which is not useful.
        raise_on_status=False
    )
    return retry_policy


def _generate_client_kwargs(has_query_params=False, generate_experiment_query_params=False, **kwargs):
    if not kwargs:
        return None

    if PAGE_SIZE_KEY in kwargs:
        kwargs[TOP_KEY] = kwargs.pop(PAGE_SIZE_KEY)

    if kwargs.get(CALLER_KEY):
        call = {RequestHeaders.CALL_NAME: kwargs.pop(CALLER_KEY)}
        if kwargs.get(CUSTOM_HEADERS_KEY):
            kwargs[CUSTOM_HEADERS_KEY].update(call)
        else:
            kwargs[CUSTOM_HEADERS_KEY] = call

    if has_query_params:
        tmp_order_by = kwargs.pop(ORDER_BY_KEY, None)
        if isinstance(tmp_order_by, list):
            tmp_order_by = tmp_order_by[0]
        if generate_experiment_query_params:
            query_params = create_experiment_query_params(filter=kwargs.pop(FILTER_KEY, None),
                                                          top=kwargs.pop(TOP_KEY, None),
                                                          orderby=tmp_order_by,
                                                          view_type=kwargs.pop(VIEW_TYPE_KEY, None))
        else:
            query_params = create_query_params(filter=kwargs.pop(FILTER_KEY, None),
                                               top=kwargs.pop(TOP_KEY, None),
                                               orderby=tmp_order_by)
        kwargs[QUERY_PARAMS_KEY] = query_params

    keys_to_delete = []
    for key, value in kwargs.items():
        if value is None:
            keys_to_delete.append(key)
    for key in keys_to_delete:
        kwargs.pop(key)
    return kwargs


def _validate_order_by(order_by):
    if not order_by:
        return None
    if not isinstance(order_by, tuple) or len(order_by) != 2:
        raise TypeError("order_by should be two-elements tuple type.")
    if not isinstance(order_by[0], str) or not isinstance(order_by[1], str):
        raise TypeError("expecting string value in order_by elements.")
    order = ['asc', 'desc']
    lowerCase = order_by[1].lower()
    if lowerCase not in order:
        raise ValueError("The second element in order_by should be 'asc' or 'desc'.")
    return ["{} {}".format(order_by[0], lowerCase)]

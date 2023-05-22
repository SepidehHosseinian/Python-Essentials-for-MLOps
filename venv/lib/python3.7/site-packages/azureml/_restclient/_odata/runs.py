# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
import datetime
import six

from .constants import (PROP_EQ_FORMAT_STR, PROP_EXISTS_FORMAT_STR,
                        TAG_EQ_FORMAT_STR, TAG_EXISTS_FORMAT_STR,
                        TYPE_EQ_FORMAT_STR, RUNTYPEV2_ORCHESTRATOR_EQ_FORMAT_STR,
                        RUNTYPEV2_TRAITS_EQ_FORMAT_STR, CREATED_AFTER_FORMAT_STR,
                        STATUS_EQ_FORMAT_STR, NO_CHILD_RUNS_QUERY, TARGET_EQ_FORMAT_STR,
                        OR_OP, RUN_ID_EXPRESSION, METRIC_TYPE_EXPRESSION,
                        NAME_EXPRESSION, AFTER_TIMESTAMP_EXPRESSION)
from .expressions import and_join
from .helpers import convert_dict_values
from azureml._restclient.contracts.utils import string_timestamp


def get_filter_run_type(filter_val):
    if isinstance(filter_val, six.text_type):
        return TYPE_EQ_FORMAT_STR.format(filter_val)
    else:
        raise ValueError("Unknown filter type for run type: {0}".format(type(filter_val)))


def get_filter_run_type_v2_orchestrator(filter_val):
    if isinstance(filter_val, six.text_type):
        return RUNTYPEV2_ORCHESTRATOR_EQ_FORMAT_STR.format(filter_val)
    else:
        raise ValueError("Unknown filter type for orchestrator in run type v2: {0}".format(type(filter_val)))


def get_filter_run_type_v2_traits(filter_val):
    if isinstance(filter_val, six.text_type):
        return RUNTYPEV2_TRAITS_EQ_FORMAT_STR.format(filter_val)
    else:
        raise ValueError("Unknown filter type for traits in run type v2: {0}".format(type(filter_val)))


def get_filter_run_status(filter_val):
    if isinstance(filter_val, six.text_type):
        return STATUS_EQ_FORMAT_STR.format(filter_val)
    else:
        raise ValueError("Unknown filter type for run status: {0}".format(type(filter_val)))


def get_filter_include_children(filter_val):
    if isinstance(filter_val, bool):
        return None if filter_val else NO_CHILD_RUNS_QUERY  # Include children = no filter
    else:
        raise ValueError("Unknown filter type for include_children: {0}".format(type(filter_val)))


def get_filter_run_created_after(filter_val):
    if isinstance(filter_val, datetime.datetime):
        return CREATED_AFTER_FORMAT_STR.format(filter_val)
    else:
        raise ValueError("Unknown filter type for run type: {0}".format(type(filter_val)))


def get_filter_run_tags(filter_val):
    if isinstance(filter_val, six.text_type):
        return get_filter_run_has_tag(filter_val)
    elif isinstance(filter_val, dict):
        return get_filter_run_tag_equals(filter_val)
    else:
        raise ValueError("Unknown filter type for run tags: {0}".format(type(filter_val)))


def get_filter_run_has_tag(tag):
    return TAG_EXISTS_FORMAT_STR.format(tag)


def get_filter_run_tag_equals(tag_value_dict):
    value_dict = convert_dict_values(tag_value_dict)
    exprs = [TAG_EQ_FORMAT_STR.format(tag, value)
             for tag, value in value_dict.items()]
    return and_join(exprs)


def get_filter_run_properties(filter_val):
    if isinstance(filter_val, six.text_type):
        return get_filter_run_has_property(filter_val)
    elif isinstance(filter_val, dict):
        return get_filter_run_property_equals(filter_val)
    else:
        raise ValueError("Unknown filter type for run properties: {0}".format(type(filter_val)))


def get_filter_run_has_property(property_key):
    return PROP_EXISTS_FORMAT_STR.format(property_key)


def get_filter_run_property_equals(prop_value_dict):
    value_dict = convert_dict_values(prop_value_dict)
    exprs = [PROP_EQ_FORMAT_STR.format(tag, value)
             for tag, value in value_dict.items()]
    return and_join(exprs)


def get_filter_run_target_name(target_name):
    return TARGET_EQ_FORMAT_STR.format(target_name)


def get_run_ids_filter_expression(run_ids):
    """get run ids filter expression"""
    separator = " {0} ".format(OR_OP)
    run_filter = [(RUN_ID_EXPRESSION + run_id) for run_id in run_ids]
    return separator.join(run_filter)


def get_metric_types_filter_expression(metric_types):
    separator = " {0} ".format(OR_OP)
    metric_filter = [(METRIC_TYPE_EXPRESSION + metric_type) for metric_type in metric_types]
    return separator.join(metric_filter)


def get_metric_name_filter_expression(metric_name):
    return NAME_EXPRESSION + metric_name


def get_after_timestamp_filter_expression(after_timestamp):
    return AFTER_TIMESTAMP_EXPRESSION + string_timestamp(after_timestamp)


def get_filter_expression(run_ids=None, metric_types=None, metric_name=None, after_timestamp=None):
    if run_ids is None and metric_types is None and metric_name is None and after_timestamp is None:
        return None

    expression_string_list = []

    if run_ids:
        run_filter = get_run_ids_filter_expression(run_ids)
        expression_string_list.append(run_filter)
    if metric_types:
        metric_types_filter = get_metric_types_filter_expression(metric_types)
        expression_string_list.append(metric_types_filter)
    if metric_name:
        metric_name_filter = get_metric_name_filter_expression(metric_name)
        expression_string_list.append(metric_name_filter)
    if after_timestamp:
        after_timestamp_filter = get_after_timestamp_filter_expression(after_timestamp)
        expression_string_list.append(after_timestamp_filter)

    return and_join(expression_string_list)

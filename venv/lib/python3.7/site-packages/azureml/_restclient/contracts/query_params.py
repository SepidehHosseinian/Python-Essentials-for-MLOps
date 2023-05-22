# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for license information.
# ---------------------------------------------------------
"""helper to create contracts of query_params"""
from ..models.query_params_dto import QueryParamsDto
from ..models.experiment_query_params_dto import ExperimentQueryParamsDto


def create_query_params(filter=None, orderby=None, top=None, view_type=None):
    return QueryParamsDto(filter=filter, continuation_token=None, orderby=orderby, top=top)


def create_experiment_query_params(filter=None, orderby=None, top=None, view_type=None):
    return ExperimentQueryParamsDto(filter=filter, continuation_token=None, orderby=orderby,
                                    top=top, view_type=view_type)

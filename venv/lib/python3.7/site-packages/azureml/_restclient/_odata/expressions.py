# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

from .constants import AND_OP


def and_join(expression_string_list):
    separator = " {0} ".format(AND_OP)
    expression = separator.join(expression_string_list)
    if len(expression_string_list) > 1:
        return "({0})".format(expression)
    else:
        return expression

# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------


def value_to_odata_string(value):
    # type: (...) -> str
    return "null" if value is None else "\"{0}\"".format(value)


def convert_dict_values(original_dict):
    assert isinstance(original_dict, dict)
    return {
        key: value_to_odata_string(value)
        for key, value in original_dict.items()
    }

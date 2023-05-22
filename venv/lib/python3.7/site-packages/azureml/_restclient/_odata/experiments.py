from .constants import TAG_EXISTS_FORMAT_STR, TAG_EQ_FORMAT_STR, NAME_EXPRESSION
from .expressions import and_join


def get_filter_expression(experiment_name=None, tags=None):
    expression_string_list = []

    if experiment_name is not None:
        expression_string_list.append("({})".format(NAME_EXPRESSION + experiment_name))

    if tags is not None:
        if isinstance(tags, str):
            expression_string_list.append(TAG_EXISTS_FORMAT_STR.format(tags))
        elif isinstance(tags, dict):
            for tag_key, tag_value in tags.items():
                expression_string_list.append(TAG_EQ_FORMAT_STR.format(tag_key, tag_value))

    return and_join(expression_string_list)

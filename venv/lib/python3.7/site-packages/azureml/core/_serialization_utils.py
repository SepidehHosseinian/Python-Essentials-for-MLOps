# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Contains class for serializing and deserializing python objects."""
import azureml._vendor.ruamel.yaml as ruamelyaml
from enum import Enum
from azureml._base_sdk_common.common import to_camel_case
from azureml.exceptions import UserErrorException, RunConfigurationException


def _serialize_to_dict(entity, use_commented_map=False):
    """Serialize entity into a dictionary recursively.

    Entity can be a python class object, dict or list.

    :param entity:
    :type entity: class object, dict, list or Enum.
    :param use_commented_map: use_commented_map=True, uses the ruamel's CommentedMap instead of dict.
        CommentedMap gives us an ordered dict in which we also add default comments before dumping into the file.
    :type use_commented_map: bool
    :return: The serialized dictionary.
    :rtype: azureml._vendor.ruamel.yaml.comments.CommentedMap
    """
    # This is a way to find out whether entity is a python object or not.
    if hasattr(entity, "__dict__") and hasattr(entity.__class__, "_field_to_info_dict"):
        if use_commented_map:
            from azureml._vendor.ruamel.yaml.comments import CommentedMap
            # Preserving the comments from load load()
            if hasattr(entity, "_loaded_commented_map") and entity._loaded_commented_map:
                result = entity._loaded_commented_map
            else:
                result = CommentedMap()
        else:
            result = dict()

        first = True
        for key, field_info in entity.__class__._field_to_info_dict.items():
            serialized_name = field_info.serialized_name or key
            if key in entity.__dict__ and (serialized_name[0] != "_") and \
                    not (field_info.exclude_if_none and entity.__dict__[key] is None):
                name = to_camel_case(serialized_name)
                if hasattr(field_info.field_type, "_serialize_to_dict"):
                    result[name] = field_info.field_type._serialize_to_dict(
                        entity.__dict__[key], use_commented_map=use_commented_map)
                else:
                    result[name] = _serialize_to_dict(
                        entity.__dict__[key], use_commented_map=use_commented_map)
                if use_commented_map and field_info.documentation:
                    # TODO: Indenting
                    if not _check_before_comment(result, name, first=first):
                        _yaml_set_comment_before_after_key_with_error(result, name,
                                                                      field_info.documentation)
                first = False

        return result

    if isinstance(entity, list):
        return [_serialize_to_dict(x, use_commented_map=use_commented_map) for x in entity]

    if isinstance(entity, dict):
        # Converting a CommentedMap into a regular dict.
        if not use_commented_map:
            entity = dict(entity)

        for key, value in entity.items():
            if hasattr(value, "__dict__"):
                if hasattr(value.__class__, "_field_to_info_dict"):
                    entity[key] = _serialize_to_dict(
                        value, use_commented_map=use_commented_map)
                else:
                    raise UserErrorException("Dictionary with non-native python type values are not "
                                             "supported in runconfigs.{}".format(entity))

    if isinstance(entity, Enum):
        return entity.value

    # A simple literal, so we just return.
    return entity


def _deserialize_and_add_to_object(class_name, serialized_dict, object_to_populate=None):
    """Deserialize serialized_dict into an object of class_name.

    Implementation details: object_to_populate is an object of class_name class, if object_to_populate=None,
    then we create class_name(), an object with empty constructor.

    :param class_name: The class name, mainly classes of this file.
    :type class_name: str
    :param serialized_dict: The serialized dict.
    :type serialized_dict: azureml._vendor.ruamel.yaml.comments.CommentedMap
    :param object_to_populate: An object of class_name class.
    :type object_to_populate: class_name
    :return: Adds the fields and returns object_to_populate
    :rtype: class_name
    """
    # Check for string keys in dictionaries
    if class_name == str:
        return serialized_dict
    if not hasattr(class_name, "_field_to_info_dict"):
        raise RunConfigurationException("{} class doesn't have _field_to_info_dict "
                                        "field, which is required for deserializaiton".format(class_name))

    if not object_to_populate:
        object_to_populate = class_name()
        # For preserving comments in load() and save()
        if (hasattr(object_to_populate, "_loaded_commented_map")
                and type(serialized_dict) == ruamelyaml.comments.CommentedMap):
            object_to_populate._loaded_commented_map = serialized_dict

    for field, field_info in class_name._field_to_info_dict.items():
        serialized_name = field_info.serialized_name or field
        dict_key = to_camel_case(serialized_name)
        # We skip the fields that are not present in the serialized_dict
        if dict_key not in serialized_dict:
            continue

        # This should NEVER be replaced with not serialized_dict[dict_key], as that also
        # includes cases like [], False, {}, which are then written as None.
        if serialized_dict[dict_key] is None:
            # None doesn't have a type and doesn't cast well, so make it a special case.
            setattr(object_to_populate, field, None)
        elif field_info.field_type == str:
            setattr(object_to_populate, field, serialized_dict[dict_key])
        elif field_info.field_type == list:
            if field_info.list_element_type:
                list_element_type = field_info.list_element_type
                if list_element_type == str:
                    setattr(object_to_populate, field,
                            serialized_dict[dict_key])
                else:
                    # TODO: Add support for other basic types too.
                    # Else case is our custom class case.
                    list_dict = serialized_dict[dict_key]
                    list_with_objects = list()
                    for object_dict in list_dict:
                        class_object = _deserialize_and_add_to_object(
                            list_element_type, object_dict)
                        list_with_objects.append(class_object)

                    setattr(object_to_populate, field, list_with_objects)
            else:
                raise RunConfigurationException("Error in deserialization. List fields "
                                                "don't have list element type information."
                                                "field={}, serialized_dict={}".format(field, serialized_dict))
        elif field_info.field_type == dict:
            populate_dict_value = serialized_dict[dict_key]
            if field_info.list_element_type:
                try:
                    list_element_type = field_info.list_element_type
                    dict_with_objects = dict()
                    for key, value in populate_dict_value.items():
                        class_object = _deserialize_and_add_to_object(
                            list_element_type, value)
                        if hasattr(class_object, "_validate"):
                            validate_method = getattr(
                                class_object, "_validate")
                            validate_method()

                        dict_with_objects[key] = class_object
                    populate_dict_value = dict_with_objects
                except Exception as ex:
                    raise RunConfigurationException("Error in deserialization. dict fields "
                                                    "don't have list element type information. "
                                                    "field={}, list_element_type={}, serialized_dict={} with "
                                                    "exception {}".format(field, list_element_type,
                                                                          serialized_dict, ex))
            setattr(object_to_populate, field, populate_dict_value)
        elif field_info.field_type == bool:
            setattr(object_to_populate, field, bool(serialized_dict[dict_key]))
        elif field_info.field_type == int:
            setattr(object_to_populate, field, int(serialized_dict[dict_key]))
        elif issubclass(field_info.field_type, Enum):
            setattr(object_to_populate, field, field_info.field_type(serialized_dict[dict_key]))
        else:
            # Custom class case.
            if hasattr(field_info.field_type, "_deserialize_and_add_to_object"):
                setattr(object_to_populate, field,
                        field_info.field_type._deserialize_and_add_to_object(
                            serialized_dict[dict_key]))
            else:
                setattr(object_to_populate, field,
                        _deserialize_and_add_to_object(
                            field_info.field_type, serialized_dict[dict_key], getattr(object_to_populate, field)))

    return object_to_populate


def _check_before_comment(commented_map, key_to_check, first=False):
    """Check if commented_map has a comment before key_to_check or not.

    All our default comments are before a key, so we just check for that.

    :param commented_map:
    :type commented_map: azureml._vendor.ruamel.yaml.comments.CommentedMap
    :param key_to_check:
    :type key_to_check: str
    :param first: True if the key is the first key in the yaml file, as that comment is associated with the file
    and not with the key.
    :type first: bool
    :return: True if there is a comment before a key.
    :rtype: bool
    """
    if first:
        # In the first case, the comment is associated to the CommentedMap, and not to the key.
        comments_list = commented_map.ca.comment
        if not comments_list:
            return False

        # This is the comment structure in ruamel. They don't have any good method for us to check.
        return len(comments_list) > 1 and comments_list[1] and len(comments_list[1]) > 0
    else:
        comments_dict = commented_map.ca.items
        if not comments_dict:
            return False
        if key_to_check in comments_dict:
            comments_list = comments_dict[key_to_check]
            # This is the comment structure in ruamel. They don't have nay good method for us to check.
            if len(comments_list) > 1 and comments_list[1] and len(comments_list[1]) > 0:
                return True
            else:
                return False
        else:
            # No key exists, so no comments.
            return False


def _yaml_set_comment_before_after_key_with_error(commented_map, field_name, field_documentation):
    """Add the comment with error handling, which is because of a bug in ruamel.

    :param commented_map:
    :type commented_map: CommentedMap
    :param field_name:
    :type field_name: str
    :param field_documentation:
    :type field_documentation: str
    :return:
    """
    try:
        commented_map.yaml_set_comment_before_after_key(
            field_name, field_documentation)
    except Exception:
        commented_map.ca.items[field_name] = [None, [], None, None]
        commented_map.yaml_set_comment_before_after_key(
            field_name, field_documentation)

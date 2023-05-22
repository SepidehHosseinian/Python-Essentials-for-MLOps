# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import functools
import logging
import inspect
import sys


module_logger = logging.getLogger(__name__)
_experimental_link_msg = "and may change at any time. " \
                         "Please see https://aka.ms/azuremlexperimental for more information."
_docstring_template = ".. note::\n" \
                      "        {0} {1}\n\n"
_class_msg = "This is an experimental class,"
_method_msg = "This is an experimental method,"
_default_indentation = 4


def _add_class_docstring(cls):
    """Add experimental tag to the class doc string"""
    def _add_class_warning(func=None):
        """Add warning message for class init"""
        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            module_logger.warning("Class {0}: {1} {2}".format(cls.__name__, _class_msg, _experimental_link_msg))
            return func(*args, **kwargs)
        return wrapped

    doc_string = _docstring_template.format(_class_msg, _experimental_link_msg)
    if cls.__doc__:
        cls.__doc__ = _add_note_to_docstring(cls.__doc__, doc_string)
    else:
        cls.__doc__ = doc_string + '>'
    cls.__init__ = _add_class_warning(cls.__init__)
    return cls


def _add_method_docstring(func=None):
    """Add experimental tag to the method doc string"""
    doc_string = _docstring_template.format(_method_msg, _experimental_link_msg)
    if func.__doc__:
        func.__doc__ = _add_note_to_docstring(func.__doc__, doc_string)
    else:
        # '>' is required. Otherwise the note section can't be generated
        func.__doc__ = doc_string + '>'

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        module_logger.warning("Method {0}: {1} {2}".format(func.__name__, _method_msg, _experimental_link_msg))
        return func(*args, **kwargs)
    return wrapped


def experimental(wrapped):
    """Add experimental tag to a class or a method"""
    if inspect.isclass(wrapped):
        return _add_class_docstring(wrapped)
    elif inspect.isfunction(wrapped):
        return _add_method_docstring(wrapped)
    else:
        return wrapped


def _add_note_to_docstring(doc_string, note):
    """Adds experimental note to docstring at the top and
    correctly indents original docstring.
    """
    indent = _get_indentation_size(doc_string)
    doc_string = doc_string.rjust(len(doc_string) + indent)
    note = note.rjust(len(note) + indent)

    # to fix indentation and to accommodate new rules that come from using new Sphinx
    temp_note = note.split('\n')
    if (len(temp_note[0]) - len(temp_note[0].lstrip())) == (len(temp_note[1]) - len(temp_note[1].lstrip())):
        temp_note[1] = (" " * 4) + temp_note[1]
    note = "\n".join(temp_note)
    return note + doc_string


def _get_indentation_size(doc_string):
    """Finds the minimum indentation of all non-blank lines after the first line"""
    lines = doc_string.expandtabs().splitlines()
    indent = sys.maxsize
    for line in lines[1:]:
        stripped = line.lstrip()
        if stripped:
            indent = min(indent, len(line) - len(stripped))
    return indent if indent < sys.maxsize else _default_indentation

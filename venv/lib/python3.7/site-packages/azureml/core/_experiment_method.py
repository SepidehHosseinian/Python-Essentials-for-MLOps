# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

""" experiment_method.py """
import functools
import logging
import threading
from os import path, environ

from azureml._common._error_definition import AzureMLError
from azureml._common._core_user_error.user_error import MethodAlreadyRegistered, MethodNotRegistered
from azureml.exceptions import AzureMLException

module_logger = logging.getLogger(__name__)

lock_warning_msg = "The conda environment is currently locked by another AzureML job. Further job submission will " \
                   "wait until the other process finishes. If there are no other jobs running, please delete {}"


class ExperimentSubmitRegistrar(object):
    """
    A class for handling submit functions.
    """
    _method_to_submit_dict = {}
    _lock = threading.Lock()

    @classmethod
    def register_submit_function(cls, method_class, submit_function):
        """
        :param cls:
        :type cls: object
        :param method_class:
        :type method_class: str
        :param submit_function:
        :type submit_function: object
        """
        function_name = submit_function.__name__
        module_logger.debug("Trying to register submit_function {}, on method {}".format(function_name, method_class))
        with cls._lock:
            if method_class in cls._method_to_submit_dict and \
               cls._method_to_submit_dict[method_class] != submit_function:
                azureml_error = AzureMLError.create(
                    MethodAlreadyRegistered, method_class=method_class
                )
                raise AzureMLException._with_error(azureml_error)

            cls._method_to_submit_dict[method_class] = submit_function

        module_logger.debug("Registered submit_function {}, on method {}".format(function_name, method_class))

    @classmethod
    def get_submit_function(cls, method):
        """
        :param cls:
        :type cls: object
        :param method:
        :type method: object
        :return: submit_function
        :rtype: object
        """
        method_class = method.__class__
        module_logger.debug("Trying to get submit_function for method_class {}".format(method_class))
        with cls._lock:
            if method_class not in cls._method_to_submit_dict:
                azureml_error = AzureMLError.create(
                    MethodNotRegistered
                )
                raise AzureMLException._with_error(azureml_error)
            submit_function = cls._method_to_submit_dict[method_class]
            function_name = submit_function.__name__
            module_logger.debug("Retrieved submit_function {} for method {}".format(function_name, method_class))
            return submit_function


# Wrap a class to be used in experiment.submit()
def experiment_method(submit_function=None):
    """
    :param submit_function:
    :type submit_function:
    :return:
    :rtype: object
    """
    def real_decorator(init_func):
        """
        :param init_func:
        :type init_func: object
        :return:
        :rtype:
        """
        @functools.wraps(init_func)
        def wrapper(self, *args, **kwargs):
            """
            :param init_func:
            :type init_func: object
            :param args:
            :type args: list
            :param kwargs:
            :type kwargs: dict
            :return:
            :rtype: object
            """
            ExperimentSubmitRegistrar.register_submit_function(self.__class__, submit_function)
            return init_func(self, *args, **kwargs)
        return wrapper
    return real_decorator


# Get the submit function associated to the method
def get_experiment_submit(method):
    """
    :param method:
    :type method: object
    :return: submit_function
    :rtype: object
    """
    submit_function = ExperimentSubmitRegistrar.get_submit_function(method)
    return submit_function


# Check that conda lock file from any previous run(s) are gone prior to submission
def check_for_lock_file():
    lock_path = path.join(".azureml", "locks", "azureml_conda_lock")
    linux_path = path.join(environ.get('HOME', ''), lock_path)
    windows_path = path.join(environ.get('USERPROFILE', ''), lock_path)

    try:
        if path.exists(linux_path):
            logging.warning(lock_warning_msg.format(linux_path))
        elif path.exists(windows_path):
            logging.warning(lock_warning_msg.format(windows_path))
    except Exception:
        pass

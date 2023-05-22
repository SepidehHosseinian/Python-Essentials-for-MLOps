# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Contains functionality for managing conda environment dependencies.

Use the :class:`azureml.core.conda_dependencies.CondaDependencies` class to load existing conda environment
files and configure and manage new environments where experiments execute.
"""
import json
from typing import Any
import os
import azureml._vendor.ruamel.yaml as ruamelyaml
import re
from azureml._base_sdk_common.common import get_run_config_dir_name
from azureml._base_sdk_common.common import normalize_windows_paths
from azureml._base_sdk_common import __version__ as VERSION
from pkg_resources import resource_stream
from azureml.exceptions import UserErrorException
from io import StringIO

BASE_PROJECT_MODULE = 'azureml._project'
BASE_PROJECT_FILE_RELATIVE_PATH = 'base_project_files/conda_dependencies.yml'
DEFAULT_SDK_ORIGIN = 'https://pypi.python.org/simple'
CONDA_FILE_NAME = 'auto_conda_dependencies.yml'
CHANNELS = 'channels'
PACKAGES = 'dependencies'
PIP = 'pip'
PYTHON_PREFIX = 'python'
VERSION_REGEX = re.compile(r'(\d+)\.(\d+)(\.(\d+))?([ab](\d+))?$')
CNTK_DEFAULT_VERSION = '2.7'
PYTHON_DEFAULT_VERSION = '3.8.13'
LINUX_PLATFORM = 'linux'
WINDOWS_PLATFORM = 'win32'
TENSORFLOW_DEFAULT_VERSION = '2.2.0'
PYTORCH_DEFAULT_VERSION = '1.6.0'
TORCHVISION_DEFAULT_VERSION = '0.7.0'
HOROVOD_DEFAULT_VERSION = '0.19.5'
DEFAULT_CHAINER_VERSION = "7.7.0"
CUPY_DEFAULT_VERSION = "cupy-cuda102"
CPU = 'cpu'
GPU = 'gpu'
CNTK_PACKAGE_PREFIX = 'cntk'
TENSORFLOW_PACKAGE_PREFIX = 'tensorflow'
PYTORCH_PACKAGE_PREFIX = 'torch'
TORCHVISION_PACKAGE_PREFIX = 'torchvision'
CHAINER_PACKAGE_PREFIX = "chainer"
INVALID_PATHON_MESSAGE = "Invalid python version {0}," + \
    "only accept '3.8'"


class CondaDependencies(object):
    """Manages application dependencies in an Azure Machine Learning environment.

    .. note::

        If no parameters are specified, `azureml-defaults` is added as the only pip dependency.

    If the ``conda_dependencies_file_path`` parameter is not specified, then
    the CondaDependencies object contains only the Azure Machine Learning packages (`azureml-defaults`).
    The `azureml-defaults` dependency will not be pinned to a specific version and will
    target the latest version available on PyPi.

    .. remarks::

        You can load an existing conda environment file or choose to configure and manage
        the application dependencies in memory. During experiment submission, a preparation step is executed
        which creates and caches a conda environment within which the experiment executes.

        If your dependency is available through both Conda and pip (from PyPi),
        use the Conda version, as Conda packages typically come with pre-built binaries that make
        installation more reliable. For more information, see `Understanding Conda
        and Pip <https://www.anaconda.com/understanding-conda-and-pip/>`_.

        See the repository https://github.com/Azure/AzureML-Containers for details on base image dependencies.

        The following example shows how to add a package using the
        :meth:`azureml.core.conda_dependencies.CondaDependencies.add_conda_package`.

        .. code-block:: python

            from azureml.core.authentication import MsiAuthentication

            msi_auth = MsiAuthentication()

            ws = Workspace(subscription_id="my-subscription-id",
                           resource_group="my-ml-rg",
                           workspace_name="my-ml-workspace",
                           auth=msi_auth)

            print("Found workspace {} at location {}".format(ws.name, ws.location))

        Full sample is available from
        https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/manage-azureml-service/authentication-in-azureml/authentication-in-azureml.ipynb


        A pip package can also be added and the dependencies set in the :class:`azureml.core.Environment` object.

        .. code-block:: python

            conda_dep.add_pip_package("pillow==6.2.1")
            myenv.python.conda_dependencies=conda_dep

        Full sample is available from
        https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/training/using-environments/using-environments.ipynb


    :param conda_dependencies_file_path: A local path to a conda configuration file. Using this parameter
        allows for loading and editing of an existing Conda environment file.
    :type conda_dependencies_file_path: str
    """

    DEFAULT_NUMBER_OF_CONDA_PACKAGES = 0
    DEFAULT_NUMBER_OF_PIP_PACKAGES = 0
    _VALID_YML_KEYS = ['name', 'channels', 'dependencies', 'prefix', 'variables']

    @staticmethod
    def _validate_yaml(ruamel_yaml_object):
        if not isinstance(ruamel_yaml_object, dict):
            raise UserErrorException("Environment error: not a valid YAML structure")
        for key in ruamel_yaml_object.keys():
            if not str(key) in CondaDependencies._VALID_YML_KEYS:
                msg = "Environment error: unknown {} key in environment specification".format(str(key))
                raise UserErrorException(msg)

    def __init__(self, conda_dependencies_file_path=None, _underlying_structure=None):
        """Initialize a new object to manage dependencies."""
        if conda_dependencies_file_path:
            with open(conda_dependencies_file_path, "r") as input:
                self._conda_dependencies = ruamelyaml.round_trip_load(input)
        elif _underlying_structure:
            self._conda_dependencies = _underlying_structure
        else:
            with resource_stream(
                BASE_PROJECT_MODULE,
                BASE_PROJECT_FILE_RELATIVE_PATH
            ) as base_stream:
                self._conda_dependencies = ruamelyaml.round_trip_load(base_stream)
                base_stream.close()

        CondaDependencies._validate_yaml(self._conda_dependencies)
        self._python_version = self.get_python_version()

    @staticmethod
    def create(pip_indexurl=None, pip_packages=None, conda_packages=None,
               python_version=PYTHON_DEFAULT_VERSION, pin_sdk_version=True):
        r"""Initialize a new CondaDependencies object.

        Returns an instance of a CondaDependencies object with user specified dependencies.

        .. note::

            If `pip_packages` is not specified, `azureml-defaults` will be added as the default dependencies. User \
            specified `pip_packages` dependencies will override the default values.

            If `pin_sdk_version` is set to true, pip dependencies of the packages distributed as a part of Azure \
            Machine Learning Python SDK will be pinned to the SDK version installed in the current environment.

        :param pip_indexurl: The pip index URL. If not specified, the SDK origin index URL will be used.
        :type pip_indexurl: str
        :param pip_packages: A list of pip packages.
        :type pip_packages: builtin.list[str]
        :param conda_packages: A list of conda packages.
        :type conda_packages: builtin.list[str]
        :param python_version: The Python version.
        :type python_version: str
        :param pin_sdk_version: Indicates whether to pin SDK packages to the client version.
        :type pin_sdk_version: bool
        :return: A conda dependency object.
        :rtype: azureml.core.conda_dependencies.CondaDependencies
        """
        cd = CondaDependencies()
        _sdk_origin_url = CondaDependencies.sdk_origin_url().rstrip('/')

        # set index url to sdk origin pypi index if not specified
        if pip_indexurl or _sdk_origin_url != DEFAULT_SDK_ORIGIN:
            cd.set_pip_index_url(
                "--index-url {}".format(pip_indexurl if pip_indexurl else _sdk_origin_url))
            cd.set_pip_option("--extra-index-url {}".format(DEFAULT_SDK_ORIGIN))

        cd.set_python_version(python_version)

        if pip_packages is None:
            pip_packages = ['azureml-defaults']
        else:
            # clear defaults if pip packages were specified
            for package in cd.pip_packages:
                cd.remove_pip_package(package)

        scope = CondaDependencies._sdk_scope()
        # adding specified pip packages
        for package in pip_packages:
            # pin current sdk version assuming all azureml-* are a part of sdk
            if pin_sdk_version and cd._get_package_name(package) in scope:
                cd.add_pip_package("{}~={}".format(cd._get_package_name_with_extras(package), VERSION))
            else:
                cd.add_pip_package(package)

        if conda_packages:
            # clear defaults if conda packages were specified
            for conda_package in conda_packages:
                cd.remove_conda_package(conda_package)
            # adding specified conda packages
            for conda_package in conda_packages:
                cd.add_conda_package(conda_package)

        return cd

    @staticmethod
    def merge_requirements(requirements):
        """Merge package requirements.

        :param requirements: A list of packages requirements to merge.
        :type requirements: builtin.list[str]
        :return: A list of merged package requirements.
        :rtype: builtin.list[str]
        """
        packages = {}
        for req in requirements:
            package = CondaDependencies._get_package_name(req)
            if packages.get(package, None):
                packages[package].append(req[len(package):].strip())
            else:
                packages[package] = [req[len(package):].strip()]
        newpackages = []
        for pack, req in packages.items():
            newpackages.append("{}{}".format(pack, ",".join([x for x in req if x])))
        return newpackages

    @staticmethod
    def _sdk_scope():
        """Return list of SDK packages.

        :return: list of SDK packages
        :rtype: list
        """
        from azureml._project.project_manager import _sdk_scope
        return _sdk_scope()

    @staticmethod
    def sdk_origin_url():
        """Return the SDK origin index URL.

        :return: Returns the SDK origin index URL.
        :rtype: str
        """
        from azureml._project.project_manager import _current_index
        index = _current_index()
        if index:
            return index
        else:
            return "https://pypi.python.org/simple"

    def _merge_dependencies(self, conda_dependencies):
        if not conda_dependencies:
            return

        # merge channels, conda dependencies, pip packages
        for channel in conda_dependencies.conda_channels:
            self.add_channel(channel)

        for package in conda_dependencies.conda_packages:
            self.add_conda_package(package)

        for package in conda_dependencies.pip_packages:
            self.add_pip_package(package)

    def set_pip_index_url(self, index_url):
        """Set pip index URL.

        :param index_url: The pip index URL to use.
        :type index_url: str
        """
        self.set_pip_option(index_url)

    @property
    def conda_channels(self):
        """Return conda channels.

        :return: Returns the channel dependencies. The returned dependencies are a copy, and any changes to the
            returned channels won't update the conda channels in this object.
        :rtype: iter
        """
        conda_channels = []
        # We are returning a copy because self._conda_dependencies[CHANNELS] is of CommentedSeq ruamel.yaml
        # type, which extends from list. But, self._conda_dependencies[CHANNELS] contains the
        # comments and list elements, so in case user removes some element then it would mess up the
        # self._conda_dependencies[CHANNELS] and conda file completely.
        # Any edits to self._conda_dependencies[CHANNELS] should be done using the provided public
        # methods in this class.
        if CHANNELS in self._conda_dependencies:
            for ditem in self._conda_dependencies[CHANNELS]:
                conda_channels.append(ditem)
        return iter(conda_channels)

    @property
    def conda_packages(self):
        """Return conda packages.

        :return: Returns the package dependencies. Returns a copy of conda packages, and any edits to
            the returned list won't be reflected in the conda packages of this object.
        :rtype: iter
        """
        conda_dependencies = []
        if PACKAGES in self._conda_dependencies:
            for ditem in self._conda_dependencies[PACKAGES]:
                if PIP not in ditem and not isinstance(ditem, dict):
                    conda_dependencies.append(ditem)
                if PIP in ditem and isinstance(ditem, str):
                    conda_dependencies.append(ditem)
        return iter(conda_dependencies)

    def _is_option(self, item, startswith='-'):
        """Check if parameter is option.

        :param item:
        :type item: str
        :param startswith:
        :type startswith: char
        :return: Returns if item starts with '-'
        :rtype: bool
        """
        return item.startswith('-')

    def _filter_options(self, items, startswith='-', keep=True):
        """Filter options.

        :param items:
        :type items: builtin.list
        :param startswith:
        :type startswith: char
        :param keep:
        :type keep: bool
        :return: Returns the filtered options
        :rtype: builtin.list
        """
        if keep:
            return [x for x in items if self._is_option(x, startswith)]
        else:
            return [x for x in items if not self._is_option(x, startswith)]

    @property
    def pip_packages(self):
        """Return pip dependencies.

        :return: Returns the pip dependencies. Returns a copy of pip packages, and any edits to
            the returned list won't be reflected in the pip packages of this object.
        :rtype: iter
        """
        pip_dependencies = []
        if PACKAGES in self._conda_dependencies:
            for ditem in self._conda_dependencies[PACKAGES]:
                if PIP in ditem and isinstance(ditem, dict):
                    pip_dependencies = self._filter_options(ditem[PIP], keep=False)

        return iter(pip_dependencies)

    @property
    def pip_options(self):
        """Return pip options.

        :return: Returns the pip options. Returns a copy of pip options, and any edits to
            the returned list won't be reflected in the pip options of this object.
        :rtype: iter
        """
        pip_options = []
        if PACKAGES in self._conda_dependencies:
            for ditem in self._conda_dependencies[PACKAGES]:
                if PIP in ditem and isinstance(ditem, dict):
                    pip_options = self._filter_options(ditem[PIP])

        return iter(pip_options)

    def get_default_number_of_packages(self):
        """Return the default number of packages.

        :return: The default number of conda and pip packages.
        :rtype: int
        """
        if self.DEFAULT_NUMBER_OF_CONDA_PACKAGES == 0 and \
                self.DEFAULT_NUMBER_OF_PIP_PACKAGES == 0:
            conda_packages = 0
            pip_packages = 0
            with resource_stream(
                    BASE_PROJECT_MODULE,
                    BASE_PROJECT_FILE_RELATIVE_PATH
            ) as base_stream:
                conda_dependencies = ruamelyaml.round_trip_load(base_stream)
                if PACKAGES in conda_dependencies:
                    for ditem in conda_dependencies[PACKAGES]:
                        if PIP not in ditem and not isinstance(ditem, dict):
                            conda_packages += 1
                        else:
                            pip_packages = len(self._filter_options(ditem[PIP], keep=False))
                base_stream.close()
            self.DEFAULT_NUMBER_OF_CONDA_PACKAGES = conda_packages
            self.DEFAULT_NUMBER_OF_PIP_PACKAGES = pip_packages

        return self.DEFAULT_NUMBER_OF_CONDA_PACKAGES, self.DEFAULT_NUMBER_OF_PIP_PACKAGES

    def get_python_version(self):
        """Return the Python version.

        :return: The Python version.
        :rtype: str
        """
        if PACKAGES in self._conda_dependencies:
            for ditem in self._conda_dependencies[PACKAGES]:
                # package name must match python, exclude package names like python-blah
                if isinstance(ditem, str) and ditem.startswith(PYTHON_PREFIX) and \
                        CondaDependencies._get_package_name(ditem) == PYTHON_PREFIX:
                    return self._get_version(ditem)
        return None

    def set_python_version(self, version):
        """Set the Python version.

        :param version: The Python version to add.
        :type version: str
        :rtype: None
        """
        if PACKAGES in self._conda_dependencies:
            has_python, index = self._has_python_package()
            if has_python:
                # TODO: update the cntk url if exists
                # and update the python package dependency
                if self._python_version != version:
                    # Doing an inplace update to preserve the comment above this field in the file.
                    self._conda_dependencies[PACKAGES][index] = PYTHON_PREFIX + '=' + version
            else:
                self._conda_dependencies[PACKAGES].append(PYTHON_PREFIX + '=' + version)
        else:
            self._conda_dependencies[PACKAGES] = [PYTHON_PREFIX + '=' + version]
        self._python_version = version
        return None

    def add_channel(self, channel):
        """Add a conda channel.

        A list of channels can be found at https://docs.anaconda.com/anaconda/user-guide/tasks/using-repositories/

        :param channel: The conda channel to add.
        :type channel: str
        :rtype: None
        """
        if CHANNELS in self._conda_dependencies:
            if channel not in self._conda_dependencies[CHANNELS]:
                self._conda_dependencies[CHANNELS].append(channel)
        else:
            self._conda_dependencies[CHANNELS] = [channel]

    def remove_channel(self, channel):
        """Remove a conda channel.

        :param channel: The conada channel to remove.
        :type channel: str
        """
        if CHANNELS in self._conda_dependencies and \
                channel in self._conda_dependencies[CHANNELS]:
            self._remove_from_list(self._conda_dependencies[CHANNELS], channel)

    def add_conda_package(self, conda_package):
        """Add a conda package.

        :param conda_package: The conda package to add.
        :type conda_package: str
        """
        if PACKAGES in self._conda_dependencies:
            if conda_package not in self._conda_dependencies[PACKAGES]:
                # package name must match python, exclude package names like python-blah
                if conda_package.startswith(PYTHON_PREFIX) and \
                        CondaDependencies._get_package_name(conda_package) == PYTHON_PREFIX:
                    python_version = self._get_version(conda_package)
                    self.set_python_version(python_version)
                else:
                    self._conda_dependencies[PACKAGES].append(conda_package)
        else:
            self._conda_dependencies[PACKAGES] = [conda_package]

    def remove_conda_package(self, conda_package):
        """Remove a conda package.

        :param conda_package: The conda package to remove.
        :type conda_package: str
        """
        if PACKAGES in self._conda_dependencies and \
                conda_package in self._conda_dependencies[PACKAGES]:
            # package name must match python, exclude package names like python-blah
            if conda_package.startswith(PYTHON_PREFIX) and \
                    CondaDependencies._get_package_name(conda_package) == PYTHON_PREFIX:
                raise UserErrorException(
                    "python must not be removed from conda dependencies")

            self._remove_from_list(self._conda_dependencies[PACKAGES], conda_package)

    def add_pip_package(self, pip_package):
        r"""Add a pip package.

        .. note::

            Adding a dependency of an already referenced package will remove the previous reference and add a new \
            reference to the end of the dependencies list. This may change the order of the dependencies.

        :param pip_package: The pip package to be add.
        :type pip_package: str
        """
        if self._is_option(pip_package):
            raise UserErrorException(
                "Invalid package name {}".format(
                    pip_package
                ))

        self.remove_pip_package(pip_package)

        if not self._has_pip_package():
            pip_obj = {PIP: [pip_package]}
            if PACKAGES in self._conda_dependencies:
                self._conda_dependencies[PACKAGES].append(pip_obj)
            else:
                self._conda_dependencies[PACKAGES] = [pip_obj]
        elif pip_package not in self.pip_packages:
            for pitem in self._conda_dependencies[PACKAGES]:
                if PIP in pitem and isinstance(pitem, dict):
                    pitem[PIP].append(pip_package)

    def set_pip_option(self, pip_option):
        """Add a pip option.

        :param pip_option: The pip option to add.
        :type pip_option: str
        """
        if not self._is_option(pip_option):
            raise UserErrorException(
                "Invalid pip option {}".format(
                    pip_option
                ))

        if not self._has_pip_package():
            pip_obj = {PIP: [pip_option]}
            if PACKAGES in self._conda_dependencies:
                self._conda_dependencies[PACKAGES].append(pip_obj)
            else:
                self._conda_dependencies[PACKAGES] = [pip_obj]

        else:
            options = [x.split()[0] for x in self.pip_options]
            option_to_add = pip_option.split()[0]
            for pitem in self._conda_dependencies[PACKAGES]:
                if PIP in pitem and isinstance(pitem, dict):
                    if option_to_add not in options:
                        pitem[PIP].append(pip_option)
                    else:
                        for i in range(len(pitem[PIP])):
                            if pitem[PIP][i].split()[0] == option_to_add:
                                pitem[PIP][i] = pip_option

    def remove_pip_option(self, pip_option):
        """Remove a pip option.

        :param pip_option: The pip option to remove.
        :type pip_option: str
        """
        if not self._is_option(pip_option):
            raise UserErrorException(
                "Invalid pip option {}".format(
                    pip_option
                ))

        if self._has_pip_package():
            options = [x.split()[0] for x in self.pip_options]
            option_to_remove = pip_option.split()[0]
            if option_to_remove in options:
                for pitem in self._conda_dependencies[PACKAGES]:
                    if PIP in pitem and isinstance(pitem, dict):
                        to_remove = None
                        for i in range(len(pitem[PIP])):
                            if pitem[PIP][i].split()[0] == option_to_remove:
                                to_remove = pitem[PIP][i]
                        if to_remove:
                            self._remove_from_list(pitem[PIP], to_remove)

    @staticmethod
    def _get_package_name(value):
        """Return package name.

        :param value:
        :type value: str
        :return: Returns the package name
        :rtype: str
        """
        # If it is whl file get package name by parsing the whl file pattern
        # https://www.python.org/dev/peps/pep-0427/#file-name-convention
        if value.endswith(".whl"):
            # whl file can be a URL, so add split based on /
            return re.split("-", re.split("/", value)[-1])[0]
        else:
            pattern = r"[\[=|<|>|~|!]"
            return re.split(pattern, value)[0]

    @staticmethod
    def _get_package_name_with_extras(value):
        """Return package name with extras.

        :param value:
        :type value: str
        :return: Returns the package name with extras
        :rtype: str
        """
        pattern = "[=|<|>|~|!]"
        return re.split(pattern, value)[0]

    def remove_pip_package(self, pip_package):
        """Remove a pip package.

        :param pip_package: The pip package to remove.
        :type pip_package: str
        """
        if self._is_option(pip_package):
            raise UserErrorException(
                "Invalid package name {}".format(
                    pip_package
                ))

        # strip version and extras
        packages = [CondaDependencies._get_package_name(x) for x in self.pip_packages]
        package_to_remove = CondaDependencies._get_package_name(pip_package)
        if pip_package in self.pip_packages or package_to_remove in packages:
            for pitem in self._conda_dependencies[PACKAGES]:
                if PIP in pitem and isinstance(pitem, dict):
                    to_remove = None
                    for i in range(len(pitem[PIP])):
                        if CondaDependencies._get_package_name(pitem[PIP][i]) == package_to_remove:
                            to_remove = pitem[PIP][i]
                    if to_remove:
                        self._remove_from_list(pitem[PIP], to_remove)

    def set_pip_requirements(self, pip_requirements):
        """Overwrite the entire pip section of conda dependencies.

        :param pip_requirements: The list of pip packages and options.
        :type pip_requirements: builtin.list[str]
        """
        if not self._has_pip_package():
            pip_obj = {PIP: pip_requirements}
            if PACKAGES in self._conda_dependencies:
                self._conda_dependencies[PACKAGES].append(pip_obj)
            else:
                self._conda_dependencies[PACKAGES] = [pip_obj]
        else:
            for pitem in self._conda_dependencies[PACKAGES]:
                if isinstance(pitem, dict) and PIP in pitem:
                    pitem[PIP] = pip_requirements

    def add_cntk_package(self, core_type=CPU):
        """Add a Microsoft Cognitive Toolkit (CNTK) package.

        :param core_type: 'cpu' or 'gpu'.
        :type core_type: str
        """
        self._validate_core_type(core_type)
        self._remove_pip_package_with_prefix(CNTK_PACKAGE_PREFIX)
        self.add_pip_package(self._get_cntk_package(core_type))

    def add_tensorflow_pip_package(self, core_type=CPU, version=None):
        """Add a Tensorflow pip package.

        :param core_type: 'cpu' or 'gpu'.
        :type core_type: str
        :param version: The package version.
        :type version: str
        """
        self._validate_core_type(core_type)
        self._remove_pip_package_with_prefix(TENSORFLOW_PACKAGE_PREFIX)
        self.add_pip_package(self._get_tensorflow_pip_package(core_type, version))

    def add_tensorflow_conda_package(self, core_type=CPU, version=None):
        """Add a Tensorflow conda package.

        :param core_type: 'cpu' or 'gpu'.
        :type core_type: str
        :param version: The package version.
        :type version: str
        """
        self._validate_core_type(core_type)
        self._remove_conda_package_with_prefix(TENSORFLOW_PACKAGE_PREFIX)
        self.add_conda_package(self._get_tensorflow_conda_package(core_type, version))

    def save_to_file(self, base_directory, conda_file_path=None):
        """DEPRECATED, use :func:`save`.

        Save the conda dependencies object to file.

        :param base_directory: The base directory to save the file.
        :type base_directory: str
        :param conda_file_path: The file name.
        :type conda_file_path: str
        :return: The normalized conda path.
        :rtype: str
        """
        run_config_dir_name = get_run_config_dir_name(base_directory)
        if not conda_file_path:
            conda_file_path = "{}/{}".format(run_config_dir_name, CONDA_FILE_NAME)

        # currently we need to return the relative path of the conda file
        # with the linux path style
        # the full_path is the full path of the conda file which might
        # be the window path style. That's why we have two separate path here
        normalized_conda_path = normalize_windows_paths(conda_file_path)
        full_path = os.path.join(
            os.path.abspath(base_directory), normalized_conda_path)
        self._validate()

        with open(full_path, 'w') as outfile:
            ruamelyaml.round_trip_dump(self._conda_dependencies, outfile)
        return normalized_conda_path

    def save(self, path=None):
        """Save the conda dependencies object to file.

        :param path: The fully qualified path of the file you want to save to.
        :type path: str
        :return: The normalized conda path.
        :rtype: str
        :raises azureml.exceptions.UserErrorException: Raised for issues saving the dependencies.
        """
        if os.path.isdir(path):
            raise UserErrorException("Cannot save a conda environment specification file to a directory. "
                                     "Please specify a fully qualified path along with the "
                                     "file name to save the file.")

        parent_dir = os.path.dirname(path)

        if parent_dir == "" or os.path.exists(parent_dir) and os.path.isdir(parent_dir):
            normalized_conda_path = normalize_windows_paths(path)
        else:
            raise UserErrorException(
                "Cannot save the conda environment specification file to an invalid path.")

        self._validate()

        with open(normalized_conda_path, 'w') as outfile:
            ruamelyaml.round_trip_dump(self._conda_dependencies, outfile)
        return normalized_conda_path

    def serialize_to_string(self):
        """Serialize conda dependencies object into a string.

        :return: The conda dependencies object serialized into a string.
        :rtype: str
        """
        with StringIO() as output:
            ruamelyaml.round_trip_dump(self._conda_dependencies, output)
            content = output.getvalue()
            output.close()
            return content

    def as_dict(self) -> Any:
        """Return conda dependecies."""
        # do ser-de using json to get simple dict
        return json.loads(json.dumps(self._conda_dependencies))

    def _remove_from_list(self, element_list, element):
        """Remove element from list.

        Remove element from element_list considering both cases if type(element_list)=
        =azureml._vendor.ruamel.yaml.comments.CommentedSeq or type(element_list)==list.

        :param element_list: The list of elements.
        :type element_list: azureml._vendor.ruamel.yaml.comments.CommentedSeq or list
        :param element: The element to remove.
        :type element: str
        :return:
        :rtype: None
        """
        if type(element_list) == ruamelyaml.comments.CommentedSeq:
            # We need to delete the element, and any associated comment, so we need to
            # ruamel.yaml methods.
            count = 0
            for litem in element_list:
                if litem == element:
                    element_list.pop(count)
                count = count + 1
        else:
            element_list.remove(element)

    def _validate_core_type(self, core_type):
        if core_type != CPU and core_type != GPU:
            raise UserErrorException(
                "Invalid core type {0}. Only accept 'cpu' or 'gpu'".format(
                    core_type
                ))

    def _has_pip_package(self):
        """Check if object has pip packages.

        :return: Returns if conda dependencies have pip packages
        :rtype: bool
        """
        has_pip = False
        if PACKAGES in self._conda_dependencies:
            for ditem in self._conda_dependencies[PACKAGES]:
                if PIP in ditem and isinstance(ditem, dict):
                    return True
        return has_pip

    def _has_python_package(self):
        """Check if the conda file, dict, hsa python package or not.

        :return: Returns bool and index of the python package.
        :rtype: :class:`list`
        """
        has_python = False
        index = -1
        count = 0
        if PACKAGES in self._conda_dependencies:
            for ditem in self._conda_dependencies[PACKAGES]:
                # package name must match python, exclude package names like python-blah
                if isinstance(ditem, str) and ditem.startswith(PYTHON_PREFIX) and \
                        CondaDependencies._get_package_name(ditem) == PYTHON_PREFIX:
                    index = count
                    return True, index
                count = count + 1
        return has_python, index

    def _validate(self):
        has_python, index = self._has_python_package()
        if not has_python:
            raise UserErrorException("Python package is missing.")

    def _get_version_major_minor(self, version):
        """Return version groups.

        :param version:
        :type version: str
        :return: Returns group of matched items
        :rtype: :class:`list`
        """
        match = VERSION_REGEX.match(version)
        if not match:
            raise UserErrorException(
                "invalid version number '{0}'".format(version))
        return match.group(1) + '.' + match.group(2)

    def _get_version(self, version):
        """Return version.

        :param version:
        :type version: str
        :return: Returns group of matched items
        :rtype: :class:`list`
        """
        matched_group = VERSION_REGEX.search(version)
        if matched_group:
            return matched_group.group()

    def _remove_pip_package_with_prefix(self, prefix):
        """Remove pip package that starts with prefix.

        :param prefix:
        :type prefix: str
        """
        items_to_remove = self._get_pip_package_with_prefix(prefix)
        if items_to_remove:
            for item in items_to_remove:
                self.remove_pip_package(item)

    def _get_pip_package_with_prefix(self, prefix):
        """Return list of pip packages with prefix.

        :param prefix:
        :type prefix: str
        :return: Returns the list of items to remove
        :rtype: :class:`list`
        """
        items_to_remove = []
        if PACKAGES in self._conda_dependencies:
            for ditem in self._conda_dependencies[PACKAGES]:
                if PIP in ditem and isinstance(ditem, dict):
                    for pitem in ditem[PIP]:
                        if isinstance(pitem, str) and pitem.startswith(prefix):
                            items_to_remove.append(pitem)
        return items_to_remove

    def _get_conda_package_with_prefix(self, prefix):
        """Return list of conda packages with prefix.

        :param prefix:
        :type prefix: str
        :return: Returns the list of items to remove
        :rtype: :class:`list`
        """
        items_to_remove = []
        if PACKAGES in self._conda_dependencies:
            for ditem in self._conda_dependencies[PACKAGES]:
                if isinstance(ditem, str) and ditem.startswith(prefix):
                    items_to_remove.append(ditem)
        return items_to_remove

    def _remove_conda_package_with_prefix(self, prefix):
        """Remove conda package that starts with prefix.

        :param prefix:
        :type prefix: str
        """
        items_to_remove = []
        if PACKAGES in self._conda_dependencies:
            for ditem in self._conda_dependencies[PACKAGES]:
                if isinstance(ditem, str) and ditem.startswith(prefix):
                    items_to_remove.append(ditem)
        if items_to_remove:
            for item in items_to_remove:
                self.remove_conda_package(item)

    def _get_cntk_package(self, core_type):
        """Return cntk package.

        :param core_type:
        :type core_type: str
        :return: Returns cntk package
        :rtype: str
        """
        cntk_version_suffix = "==" + CNTK_DEFAULT_VERSION
        if core_type.lower() == GPU:
            return CNTK_PACKAGE_PREFIX + '-gpu' + cntk_version_suffix
        return CNTK_PACKAGE_PREFIX + cntk_version_suffix

    def _get_tensorflow_pip_package(self, core_type, version=None):
        """Return tensoflow pip package.

        :param core_type:
        :type core_type: str
        :param version:
        :type version: str
        :return: Returns tensorflow pip package
        :rtype: str
        """
        tf_version_suffix = "==" + (version if version else TENSORFLOW_DEFAULT_VERSION)
        if core_type.lower() == GPU:
            return TENSORFLOW_PACKAGE_PREFIX + '-gpu' + tf_version_suffix
        return TENSORFLOW_PACKAGE_PREFIX + tf_version_suffix

    def _get_tensorflow_conda_package(self, core_type, version=None):
        """Return tensorflow conda package.

        :param core_type:
        :type core_type: str
        :param version:
        :type version: str
        :return: Returns tensorflow conda package
        :rtype: str
        """
        tf_version_suffix = "=" + (version if version else TENSORFLOW_DEFAULT_VERSION)
        if core_type.lower() == GPU:
            return TENSORFLOW_PACKAGE_PREFIX + '-gpu' + tf_version_suffix
        return TENSORFLOW_PACKAGE_PREFIX + tf_version_suffix

    def _pin_sdk_packages(self, version=None):
        """Pin all AzureML SDK packages to a uniform version.

        :param version: Version to pin to; uses installed SDK version by default.
        :type version: str
        """
        if version is None:
            version = VERSION

        scope = CondaDependencies._sdk_scope()

        if PACKAGES in self._conda_dependencies:
            for ditem in self._conda_dependencies[PACKAGES]:
                if isinstance(ditem, dict) and PIP in ditem and isinstance(ditem[PIP], list):
                    old_pip_packages = ditem[PIP]
                    ditem[PIP] = []

                    for package in old_pip_packages:
                        if isinstance(package, str) and CondaDependencies._get_package_name(package) in scope:
                            name_with_extras = CondaDependencies._get_package_name_with_extras(package)
                            ditem[PIP].append('{}~={}'.format(name_with_extras, version))
                        else:
                            ditem[PIP].append(package)

    @staticmethod
    def _register_private_pip_wheel_to_blob(workspace, file_path, container_name=None, blob_name=None):
        """Register the private pip package wheel file on disk to the Azure storage blob attached to the workspace.

        :param workspace: Workspace object to use to register the private pip package wheel.
        :type workspace: azureml.core.workspace.Workspace
        :param file_path: Path to the local pip wheel file on disk, including the file extension.
        :type file_path: str
        :param container_name: Container name to use to store the pip wheel. Defaults to private-packages.
        :type container_name: str
        :param blob_name: Full path to use to store the pip wheel on the blob container.
        :type blob_name: str
        :return: Returns the full URI to the uploaded pip wheel on Azure blob storage to use in conda dependencies.
        :rtype: str
        """
        import logging
        logging.warning("_register_private_pip_wheel_to_blob() is going to be removed in the next SDK release."
                        "Please use Environment.add_private_pip_wheel() instead.")
        from azureml.core.environment import Environment
        return Environment.add_private_pip_wheel(workspace, file_path)

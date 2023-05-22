# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Contains functionality for managing Databricks environments in Azure Machine Learning.

For more information on working with Databricks in Azure Machine Learning, see
[Configure a development environment for Azure Machine
Learning](https://docs.microsoft.com/azure/machine-learning/how-to-configure-environment#aml-databricks).
"""
import collections
import logging

from azureml._base_sdk_common.abstract_run_config_element import _AbstractRunConfigElement
from azureml._base_sdk_common.field_info import _FieldInfo
from azureml.exceptions import UserErrorException

module_logger = logging.getLogger(__name__)


class MavenLibrary(_AbstractRunConfigElement):
    """Specifies a Maven library for use in a :class:`azureml.pipeline.steps.DatabricksStep` pipeline step.

    For an example of working with a DatabricksStep and specifying libraries,
    see the notebook https://aka.ms/pl-databricks.

    :param coordinates: Gradle-style maven coordinates. For example: 'org.jsoup:jsoup:1.7.2'.
    :type coordinates: str
    :param repo: The Maven repo to install the Maven package from. If omitted,
        both Maven Central Repository and Spark Packages are searched.
    :type repo: str
    :param exclusions: A List of dependencies to exclude.
        For more information on Maven dependency exclusions, see
        https://maven.apache.org/guides/introduction/introduction-to-optional-and-excludes-dependencies.html.
    :type exclusions: builtin.list
    """

    # This is used to deserialize.
    # This is also the order for serialization into a file.
    _field_to_info_dict = collections.OrderedDict([
        ("coordinates", _FieldInfo(
            str, "Gradle-style maven coordinates. For example: 'org.jsoup:jsoup:1.7.2'.")),
        ("repo", _FieldInfo(str, "Maven repo to install the Maven package from.")),
        ("exclusions", _FieldInfo(
            list, "List of dependences to exclude.", list_element_type=str))
    ])

    def __init__(self, coordinates=None, repo=None, exclusions=None):
        """Initialize MavenLibrary."""
        super(MavenLibrary, self).__init__()
        if coordinates is not None and isinstance(coordinates, str):
            self.coordinates = coordinates
        else:
            self.coordinates = ""
        if repo is not None and isinstance(repo, str):
            self.repo = repo
        else:
            self.repo = ""
        if exclusions is not None and isinstance(exclusions, list):
            self.exclusions = exclusions
        else:
            self.exclusions = []

        self._initialized = True


class PyPiLibrary(_AbstractRunConfigElement):
    """Specifies a PyPi library for use in a :class:`azureml.pipeline.steps.DatabricksStep` pipeline step.

    For an example of working with a DatabricksStep and specifying libraries, see the
    notebook https://aka.ms/pl-databricks.

    :param package: The name of the PyPi package to install. An optional exact version specification is also supported.
    :type package: str
    :param repo: The repository where the package can be found. If not specified, the default pip index is used.
    :type repo: str
    """

    # This is used to deserialize.
    # This is also the order for serialization into a file.
    _field_to_info_dict = collections.OrderedDict([
        ("package", _FieldInfo(str, " The name of the pypi package to install.")),
        ("repo", _FieldInfo(str, "The repository where the package can be found."))
    ])

    def __init__(self, package=None, repo=None):
        """Initialize PyPiLibrary."""
        super(PyPiLibrary, self).__init__()
        if package is not None and isinstance(package, str):
            self.package = package
        else:
            self.package = ""
        if repo is not None and isinstance(repo, str):
            self.repo = repo
        else:
            self.repo = ""
        self._initialized = True


class RCranLibrary(_AbstractRunConfigElement):
    """Specifies a CRAN library for use in a :class:`azureml.pipeline.steps.DatabricksStep` pipeline step.

    For an example of working with a DatabricksStep and specifying libraries,
    see the notebook https://aka.ms/pl-databricks.

    :param package: The name of the CRAN package to install.
    :type package: str
    :param repo: The repository where the package can be found. If not specified, the default CRAN repo is used.
    :type repo: str
    """

    # This is used to deserialize.
    # This is also the order for serialization into a file.
    _field_to_info_dict = collections.OrderedDict([
        ("package", _FieldInfo(str, "The name of the CRAN package to install.")),
        ("repo", _FieldInfo(str, "The repository where the package can be found."))
    ])

    def __init__(self, package=None, repo=None):
        """Initialize RCranLibrary."""
        super(RCranLibrary, self).__init__()
        if package is not None and isinstance(package, str):
            self.package = package
        else:
            self.package = ""
        if repo is not None and isinstance(repo, str):
            self.repo = repo
        else:
            self.repo = ""
        self._initialized = True


class JarLibrary(_AbstractRunConfigElement):
    """Specifies a JAR library for use in a :class:`azureml.pipeline.steps.DatabricksStep` pipeline step.

    For an example of working with a DatabricksStep and specifying libraries,
    see the notebook https://aka.ms/pl-databricks.

    :param library: The URI of the JAR to be installed. Only DBFS and S3 URIs are supported.
    :type library: str
    """

    # This is used to deserialize.
    # This is also the order for serialization into a file.
    _field_to_info_dict = collections.OrderedDict([
        ("library", _FieldInfo(
            str, "URI of the jar to be installed. Only DBFS and S3 URIs are supported."))
    ])

    def __init__(self, library=None):
        """Initialize JarLibrary."""
        super(JarLibrary, self).__init__()
        if library is not None and isinstance(library, str):
            self.library = library
        else:
            self.library = ""
        self._initialized = True


class EggLibrary(_AbstractRunConfigElement):
    """Specifies an Egg library for use in a :class:`azureml.pipeline.steps.DatabricksStep` pipeline step.

    For an example of working with a DatabricksStep and specifying libraries,
    see the notebook https://aka.ms/pl-databricks.

    :param library: The URI of the Egg to be installed. Only DBFS and S3 URIs are supported.
    :type library: str
    """

    # This is used to deserialize.
    # This is also the order for serialization into a file.
    _field_to_info_dict = collections.OrderedDict([
        ("library", _FieldInfo(
            str, "URI of the egg to be installed. Only DBFS and S3 URIs are supported."))
    ])

    def __init__(self, library=None):
        """Initialize EggLibrary."""
        super(EggLibrary, self).__init__()
        if library is not None and isinstance(library, str):
            self.library = library
        else:
            self.library = ""

        self._initialized = True


class DatabricksCluster(_AbstractRunConfigElement):
    """Defines Databricks cluster information for use in a :class:`azureml.core.databricks.DatabricksSection`.

    :param existing_cluster_id: A cluster ID of an existing interactive cluster on the Databricks workspace. If this
                                parameter is specified, none of the other parameters should be specified.
    :type existing_cluster_id: str
    :param spark_version: The version of Spark for the Databricks run cluster. Example: "10.4.x-scala2.12".
    :type spark_version: str
    :param node_type: The Azure VM node types for the Databricks run cluster. Example: "Standard_D3_v2".
    :type node_type: str
    :param instance_pool_id: The instance pool ID to which the cluster needs to be attached to.
    :type instance_pool_id: str
    :param num_workers: The number of workers for a Databricks run cluster. If this parameter is specified, the
                        ``min_workers`` and ``max_workers`` parameters should not be specified.
    :type num_workers: int
    :param min_workers: The minimum number of workers for an autoscaled Databricks cluster.
    :type min_workers: int
    :param max_workers: The number of workers for an autoscaled Databricks run cluster.
    :type max_workers: int
    :param spark_env_variables: The Spark environment variables for the Databricks run cluster.
    :type spark_env_variables: dict({str:str})
    :param spark_conf: The Spark configuration for the Databricks run cluster.
    :type spark_conf: dict({str:str})
    :param init_scripts: A list of DBFS paths to init scripts for the Databricks run cluster.
    :type init_scripts: builtin.list[str]
    :param cluster_log_dbfs_path: The DBFS path to where clusters logs need to be delivered.
    :type cluster_log_dbfs_path: str
    :param permit_cluster_restart: if existing_cluster_id is specified, this parameter tells whether cluster
        can be restarted on behalf of user.
    :type permit_cluster_restart: bool
    """

    # This is used to deserialize.
    # This is also the order for serialization into a file.
    _field_to_info_dict = collections.OrderedDict([
        ("existing_cluster_id", _FieldInfo(str, "Cluster ID of an existing Interactive cluster on the"
                                                "Databricks workspace.")),
        ("spark_version", _FieldInfo(
            str, "The version of spark for the Databricks run cluster.")),
        ("node_type", _FieldInfo(
            str, "The Azure VM node types for the Databricks run cluster.")),
        ("instance_pool_id", _FieldInfo(
            str, "The instance pool Id to which the cluster needs to be attached to.")),
        ("num_workers", _FieldInfo(
            int, "The number of workers for a Databricks run cluster.")),
        ("min_workers", _FieldInfo(
            int, "The minimum number of workers for an auto scaled Databricks cluster.")),
        ("max_workers", _FieldInfo(
            int, "The number of workers for an auto scaled Databricks cluster.")),
        ("spark_env_variables", _FieldInfo(dict, "The spark environment variables for the Databricks"
                                                 "run cluster", user_keys=True)),
        ("spark_conf", _FieldInfo(
            dict, "The spark configuration for the Databricks run cluster", user_keys=True)),
        ("init_scripts", _FieldInfo(
            list, "DBFS paths to init scripts for the Databricks run cluster", user_keys=True)),
        ("cluster_log_dbfs_path", _FieldInfo(
            str, "DBFS paths to where clusters logs need to be delivered", user_keys=True)),
        ("permit_cluster_restart", _FieldInfo(bool, "Whether existing cluster can be restarted"))
    ])

    def __init__(self, existing_cluster_id=None, spark_version=None, node_type=None, instance_pool_id=None,
                 num_workers=None, min_workers=None, max_workers=None, spark_env_variables=None, spark_conf=None,
                 init_scripts=None, cluster_log_dbfs_path=None, permit_cluster_restart=None):
        """Initialize."""
        super(DatabricksCluster, self).__init__()
        self.existing_cluster_id = existing_cluster_id
        self.spark_version = spark_version
        self.node_type = node_type
        self.instance_pool_id = instance_pool_id

        self.num_workers = num_workers
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.spark_env_variables = spark_env_variables
        self.spark_conf = spark_conf
        self.init_scripts = init_scripts
        self.cluster_log_dbfs_path = cluster_log_dbfs_path
        self.permit_cluster_restart = permit_cluster_restart

        self._initialized = True

    def _validate(self):
        if self.existing_cluster_id is not None and (self.spark_version is not None
                                                     or self.node_type is not None
                                                     or self.instance_pool_id is not None
                                                     or self.num_workers is not None
                                                     or self.min_workers is not None
                                                     or self.max_workers is not None
                                                     or self.spark_env_variables is not None
                                                     or self.spark_conf is not None
                                                     or self.init_scripts is not None
                                                     or self.cluster_log_dbfs_path is not None):
            raise UserErrorException("If you specify existing_cluster_id, you can not specify any other "
                                     "cluster parameters")
        if (self.existing_cluster_id is None and self.spark_version is None and self.node_type is None
            and self.instance_pool_id is None and self.num_workers is None and self.min_workers is None
            and self.max_workers is None and self.spark_env_variables is None and self.spark_conf is None
                and self.init_scripts is None and self.cluster_log_dbfs_path is None):
            raise UserErrorException(
                "Either specify existing_cluster_id or specify the rest of the cluster parameters")

        if self.existing_cluster_id is None:
            if self.num_workers is None and (self.min_workers is None or self.max_workers is None):
                raise UserErrorException(
                    "You need to either specify num_workers or both min_workers and max_workers")
            if self.num_workers is not None and (self.min_workers is not None or self.max_workers is not None):
                raise UserErrorException(
                    "You need to either specify num_workers or both min_workers and max_workers")
            if (self.min_workers is None and self.max_workers is not None) or \
                    (self.min_workers is not None and self.max_workers is None):
                raise UserErrorException(
                    "Either specify both min_workers and max_workers or neither")

        if self.existing_cluster_id is not None and not isinstance(self.existing_cluster_id, str):
            raise UserErrorException("existing_cluster_id needs to be a str")
        if self.spark_version is not None and not isinstance(self.spark_version, str):
            raise UserErrorException("spark_version needs to be a str")
        if self.node_type is not None and not isinstance(self.node_type, str):
            raise UserErrorException("node_type needs to be a str")
        if self.instance_pool_id is not None and not isinstance(self.instance_pool_id, str):
            raise UserErrorException("instance_pool_id needs to be a str")
        if self.num_workers is not None and not isinstance(self.num_workers, int):
            raise UserErrorException("num_workers needs to be a int")
        if self.min_workers is not None and not isinstance(self.min_workers, int):
            raise UserErrorException("min_workers needs to be a int")
        if self.max_workers is not None and not isinstance(self.max_workers, int):
            raise UserErrorException("max_workers needs to be a int")
        if self.spark_env_variables is not None and not isinstance(self.spark_env_variables, dict):
            raise UserErrorException("spark_env_variables needs to be a dict")
        if self.spark_conf is not None and not isinstance(self.spark_conf, dict):
            raise UserErrorException("spark_conf needs to be a dict")
        if self.init_scripts is not None and not isinstance(self.init_scripts, list):
            raise UserErrorException("init_scripts needs to be a list of strings")
        if self.cluster_log_dbfs_path is not None and not isinstance(self.cluster_log_dbfs_path, str):
            raise UserErrorException("cluster_log_dbfs_path needs to be a string")

        if self.permit_cluster_restart is not None and self.existing_cluster_id is None:
            raise UserErrorException("If you specify permit_cluster_restart, you must specify existing_cluster_id")

    def validate(self):
        """Validate the specified Databricks cluster details.

        Validate checks the types of provided parameters as well as  whether the correct combination
        of parameters is provided. For example, you need to either specify the ``existing_cluster_id``
        or specify the rest of the cluster parameters. For more information see the constructor
        parameter definitions.

        :raises: class:azureml.exceptions.UserErrorException
        """
        self._validate()


class DatabricksSection(_AbstractRunConfigElement):
    """Defines library dependencies for a cluster in Databricks.

    Use the DatabricksSection object with the class attribute ``databricks`` of the
    :class:`azureml.core.environment.Environment` class.

    :param maven_libraries: A list of Maven libraries.
    :type maven_libraries: builtin.list[azureml.core.databricks.MavenLibrary]

    :param pypi_libraries: A list of PyPi libraries.
    :type pypi_libraries: builtin.list[azureml.core.databricks.PyPiLibrary]

    :param rcran_libraries: A list of RCran libraries.
    :type rcran_libraries: builtin.list[azureml.core.databricks.RCranLibrary]

    :param jar_libraries: A list of JAR libraries.
    :type jar_libraries: builtin.list[azureml.core.databricks.JarLibrary]

    :param egg_libraries: A list of Egg libraries.
    :type egg_libraries: builtin.list[azureml.core.databricks.EggLibrary]

    :param cluster: Databricks cluster information.
    :type cluster: azureml.core.databricks.DatabricksCluster
    """

    # This is used to deserialize.
    # This is also the order for serialization into a file.
    _field_to_info_dict = collections.OrderedDict([
        ("maven_libraries", _FieldInfo(
            list, "List of maven libraries.", list_element_type=MavenLibrary)),
        ("pypi_libraries", _FieldInfo(
            list, "List of PyPi libraries", list_element_type=PyPiLibrary)),
        ("rcran_libraries", _FieldInfo(
            list, "List of RCran libraries", list_element_type=RCranLibrary)),
        ("jar_libraries", _FieldInfo(
            list, "List of JAR libraries", list_element_type=JarLibrary)),
        ("egg_libraries", _FieldInfo(
            list, "List of Egg libraries", list_element_type=EggLibrary)),
        ("cluster", _FieldInfo(DatabricksCluster, "Databricks cluster information"))
    ])

    def __init__(self, _skip_defaults=False):
        """Class DatabricksSection constructor."""
        super(DatabricksSection, self).__init__()
        self.maven_libraries = []
        self.pypi_libraries = []
        self.rcran_libraries = []
        self.jar_libraries = []
        self.egg_libraries = []
        self._cluster = None
        self._initialized = True

    @property
    def cluster(self):
        """Databricks cluster."""
        return self._cluster

    @cluster.setter
    def cluster(self, cluster):
        cluster.validate()
        self._cluster = cluster


class DatabricksEnvironment(DatabricksSection):
    """DEPRECATED - A class for managing DatabricksEnvironment.

    Use the :class:`azureml.core.databricks.DatabricksSection`.
    """

    def __init__(self):
        """Class DatabricksEnvironment constructor."""
        super(DatabricksEnvironment, self).__init__()
        module_logger.warning(
            "'DatabricksEnvironment' will be deprecated soon."
            " Please use DatabricksSection from 'azureml.core.databricks'.")

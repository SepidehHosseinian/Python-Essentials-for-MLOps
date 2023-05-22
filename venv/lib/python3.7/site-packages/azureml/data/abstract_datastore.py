# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""Contains the base functionality for datastores that save connection information to Azure storage services."""
from abc import ABCMeta

import sys


class AbstractDatastore(object):
    """Represents the base class for all datastores.

    Datastores reference storage services in Azure and contain connection information to them.
    When working with Azure Machine Learning experiments, you can retrieve datastores already registered with your
    workspace or register new ones. To get started with datastores including create a new datastore, see the
    :class:`azureml.core.datastore.Datastore` class.

    Note: When using a datastore to access data, you must have permission to access the data, which depends on the
    credentials registered with the datastore.

    You should not work with this class directly. To create a datastore, use one of the ``register*`` methods
    of the Datastore class such as :meth:`azureml.core.datastore.Datastore.register_azure_blob_container`.

    :param workspace: The workspace this datastore belongs to.
    :type workspace: str
    :param name: The datastore name.
    :type name: str
    :param name: The datastore type, for example, "AzureBlob".
    :type name: str
    """

    __metaclass__ = ABCMeta

    def __init__(self, workspace, name, datastore_type):
        """Class AbstractDatastore constructor.

        This is a base class and users should not be creating this class using the constructor.

        :param workspace: The workspace this datastore belongs to.
        :type workspace: str
        :param name: The datastore name.
        :type name: str
        :param name: The datastore type, for example, "AzureBlob".
        :type name: str
        """
        self._workspace = workspace
        self._name = name
        self._datastore_type = datastore_type

    @property
    def workspace(self):
        """Return the workspace the datastore belows to.

        :return: The workspace.
        :rtype: azureml.core.Workspace
        """
        return self._workspace

    @workspace.setter
    def workspace(self, workspace):
        """Obsolete, will be removed in future releases."""
        self._workspace = workspace

    @property
    def name(self):
        """Return the name of the datastore.

        :return: The name of the datastore.
        :rtype: str
        """
        return self._name

    @name.setter
    def name(self, name):
        """Obsolete, will be removed in future releases."""
        self._name = name

    @property
    def datastore_type(self):
        """Return the type of the datastore.

        :return: The type of the datastore, for example, "AzureBlob".
        :rtype: str
        """
        return self._datastore_type

    @datastore_type.setter
    def datastore_type(self, datastore_type):
        """Obsolete, will be removed in future releases."""
        self._datastore_type = datastore_type

    def set_as_default(self):
        """Set the current datastore as the default datastore."""
        AbstractDatastore._client().set_default(self.workspace, self.name)

    def unregister(self):
        """Unregister the datastore, the underlying storage service will not be deleted or modified."""
        AbstractDatastore._client().delete(self.workspace, self.name)

    def _as_dict(self):
        return {
            "name": self.name,
            "datastore_type": self.datastore_type
        }

    def _get_console_logger(self):
        return sys.stdout

    @staticmethod
    def _client():
        from .datastore_client import _DatastoreClient
        return _DatastoreClient

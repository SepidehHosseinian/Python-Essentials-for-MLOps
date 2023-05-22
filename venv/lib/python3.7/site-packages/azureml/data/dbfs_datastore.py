# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""Contains functionality for datastores that save connection information to Databricks File Sytem (DBFS)."""

from .abstract_datastore import AbstractDatastore


class DBFSDatastore(AbstractDatastore):
    """Represents a datastore that saves connection information to Databricks File System (DBFS).

    You should not work with this class directly. To create a datastore that saves connection information
    to DBFS, use the :meth:`azureml.core.datastore.Datastore.register_dbfs` method of the
    :class:`azureml.core.datastore.Datastore` class.

    Note: DBFSDatastore is the only supported input/output for Databricks jobs on Databricks clusters.

    :param workspace: The workspace this datastore belongs to.
    :type workspace: str
    :param name: The datastore name.
    :type name: str
    """

    def __init__(self, workspace, name):
        """Initialize a new Databricks File System (DBFS) datastore.

        :param workspace: The workspace this datastore belongs to.
        :type workspace: str
        :param name: The datastore name.
        :type name: str
        """
        import azureml.data.constants as constants

        super(DBFSDatastore, self).__init__(workspace, name, constants.DBFS)

    def _as_dict(self, hide_secret=True):
        output = super(DBFSDatastore, self)._as_dict()

        return output

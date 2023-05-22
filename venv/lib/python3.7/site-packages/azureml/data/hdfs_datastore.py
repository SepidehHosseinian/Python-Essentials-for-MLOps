# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""Contains the base functionality for datastores that save connection information to an HDFS cluster."""

from .abstract_datastore import AbstractDatastore
from azureml._base_sdk_common._docstring_wrapper import experimental


@experimental
class HDFSDatastore(AbstractDatastore):
    """Represents a datastore that saves connection information to an HDFS cluster.

    You should not work with this class directly. To create a datastore that saves connection information
    to an HDFS cluster, use the :meth:`azureml.core.datastore.Datastore.register_hdfs`
    method of the :class:`azureml.core.datastore.Datastore` class.

    Note: When using a datastore to access data, you must have permission to access the data, which depends on the
    credentials registered with the datastore.

    :param workspace: The workspace this datastore belongs to.
    :type workspace: str
    :param name: The datastore name.
    :type name: str
    :param protocol: The protocol to use when communicating with the HDFS
        cluster. http or https. Possible values include: 'http', 'https'
    :type protocol: str or ~_restclient.models.enum
    :param namenode_address: The IP address or DNS hostname of the HDFS namenode. Optionally includes a port.
    :type namenode_address: str
    :param hdfs_server_certificate_data: The TLS signing certificate of the HDFS namenode, if using TLS
        with a self-signed cert.
    :type hdfs_server_certificate: bytes, optional
    :param kerberos_realm: The Kerberos realm.
    :type kerberos_realm: str
    :param kerberos_kdc_address: The IP address or DNS hostname of the Kerberos KDC.
    :type kerberos_kdc_address: str
    :param kerberos_principal: The Kerberos principal to use for authentication and authorization.
    :type kerberos_principal: str
    :param credential_value: Keytab file contents/Password for kerberos principal. Must supply either keytab
        or password, not both. Needs to be a base64 encoded string if keytab.
    :type credential_value: str
    :param credential_type: HDFS Authentication type. Possible values include: 'KerberosKeytab', 'KerberosPassword'
    :type credential_type: str or ~_restclient.models.HdfsCredentialType
    """

    def __init__(self, workspace, name, protocol, namenode_address, hdfs_server_certificate_data,
                 kerberos_realm, kerberos_kdc_address, kerberos_principal, credential_value, credential_type):
        """Initialize a new OnPremisesHDFSDatastore.

        :param workspace: The workspace this datastore belongs to.
        :type workspace: str
        :param name: The datastore name.
        :type name: str
        :param protocol: The protocol to use when communicating with the HDFS cluster.
            Possible values include: 'http', 'https'
        :type protocol: str or ~_restclient.models.enum
        :param namenode_address: The IP address or DNS hostname of the HDFS namenode.  Optionally includes a port.
        :type namenode_address: str
        :param hdfs_server_certificate_data: The base-64 encoded TLS signing certificate of the HDFS namenode,
            if using TLS with a self-signed cert.
        :type hdfs_server_certificate: str, optional
        :param kerberos_realm: The Kerberos realm.
        :type kerberos_realm: str
        :param kerberos_kdc_address: The IP address or DNS hostname of the Kerberos KDC.
        :type kerberos_kdc_address: str
        :param kerberos_principal: The Kerberos principal to use for authentication and authorization.
        :type kerberos_principal: str
        :param credential_value: Keytab file contents/Password for kerberos principal. Must supply either keytab
            or password, not both. Needs to be a base64 encoded string if keytab.
        :type credential_value: str
        :param credential_type: HDFS Authentication type.
            Possible values include: 'KerberosKeytab', 'KerberosPassword'
        :type credential_type: str or ~_restclient.models.HdfsCredentialType
        """
        import azureml.data.constants as constants

        super(HDFSDatastore, self).__init__(workspace, name, constants.HDFS)
        self.protocol = protocol
        self.namenode_address = namenode_address
        self.hdfs_server_certificate_data = hdfs_server_certificate_data
        self.kerberos_realm = kerberos_realm
        self.kerberos_kdc_address = kerberos_kdc_address
        self.kerberos_principal = kerberos_principal
        self.credential_value = credential_value
        self.credential_type = credential_type

    def _as_dict(self, hide_secret=True):
        output = super(HDFSDatastore, self)._as_dict()
        output["protocol"] = self.protocol
        output["namenode_address"] = self.namenode_address
        output["hdfs_server_certificate"] = self.hdfs_server_certificate_data
        output["kerberos_realm"] = self.kerberos_realm
        output["kerberos_kdc_address"] = self.kerberos_kdc_address

        if not hide_secret:
            output["credential_type"] = self.credential_type
            output["credential_value"] = self.credential_value

        return output

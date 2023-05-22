# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""Contains functionality that defines how to create references to data in datastores."""

import uuid


class DataReference(object):
    """Represents a reference to data in a datastore.

    A DataReference represents a path in a datastore and can be used to describe how and where data should be made
    available in a run. It is no longer the recommended approach for data access and delivery in Azure Machine
    Learning. [Dataset](https://aka.ms/azureml/howto/createdatasets) supports accessing data from Azure Blob
    storage, Azure Files, Azure Data Lake Storage Gen1, Azure Data Lake Storage Gen2, Azure SQL Database, and Azure
    Database for PostgreSQL through unified interface with added data management capabilities. It is recommended to
    use dataset for reading data in your machine learning projects.

    For more information on how to use Azure ML dataset in two common scenarios, see the articles:
    * `Create and run machine learning pipelines <https://aka.ms/pipeline-with-datastore>`_
    * `Create estimators in training <https://aka.ms/train-with-datastore>`_

    .. remarks::

        A DataReference defines both the data location and how the data is used on the target compute binding
        (mount or upload). The path to the data in the datastore can be the root /, a directory within the
        datastore, or a file in the datastore.

    :param datastore: The datastore to reference.
    :type datastore: typing.Union[azureml.data.azure_storage_datastore.AbstractAzureStorageDatastore,
                    azureml.data.azure_data_lake_datastore.AzureDataLakeDatastore]
    :param data_reference_name: The name of the data reference.
    :type data_reference_name: str
    :param path_on_datastore: The relative path in the backing storage for the data reference.
    :type path_on_datastore: str
    :param mode: The operation on the data reference. Supported values are 'mount' (the default) and 'download'.

        Use the 'download' mode when your script expects a specific (e.g., hard-coded) path for the input data.
        In this case, specify the path with the ``path_on_compute`` parameter when you declare the DataReference.
        Azure Machine Learning will download the data specified by that path before executing your script.

        With the 'mount' mode, a temporary directory is created with the mounted data and an environment variable
        $AZUREML_DATAREFERENCE_<data_reference_name> is set with the path to the temporary directory.
        If you pass a DataReference into the arguments list for a pipeline step (e.g. PythonScriptStep),
        then the reference will be expanded to the local data path at runtime.
    :type mode: str
    :param path_on_compute: The path on the compute target for the data reference.
    :type path_on_compute: str
    :param overwrite: Indicates whether to overwrite existing data.
    :type overwrite: bool
    """

    def __init__(self, datastore, data_reference_name=None,
                 path_on_datastore=None, mode="mount", path_on_compute=None, overwrite=False):
        """Class DataReference constructor.

        :param datastore: The datastore to reference.
        :type datastore: typing.Union[azureml.data.azure_storage_datastore.AbstractAzureStorageDatastore,
                        azureml.data.azure_data_lake_datastore.AzureDataLakeDatastore]
        :param data_reference_name: The name of the data reference.
        :type data_reference_name: str
        :param path_on_datastore: The relative path in the backing storage for the data reference.
        :type path_on_datastore: str
        :param mode: The operation on the data reference. Supported values 'mount' (the default) and 'download'.

            Use the 'download' mode when your script expects a specific (e.g., hard-coded) path for the input data.
            In this case, specify the path with the ``path_on_compute`` parameter when you declare the DataReference.
            Azure Machine Learning will download the data specified by that path before executing your script.

            With the 'mount' mode, a temporary directory is created with the mounted data and an environment variable
            $AZUREML_DATAREFERENCE_<data_reference_name> is set with the path to the temporary directory.
            If you pass a DataReference into the arguments list for a pipeline step (e.g. PythonScriptStep),
            then the reference will be expanded to the local data path at runtime.
        :type mode: str
        :param path_on_compute: The path on the compute target for the data reference.
        :type path_on_compute: str
        :param overwrite: Indicates whether to overwrite existing data.
        :type overwrite: bool
        """
        self.datastore = datastore
        self.data_reference_name = data_reference_name
        self.path_on_datastore = path_on_datastore
        self.mode = mode
        self.path_on_compute = path_on_compute
        self.overwrite = overwrite

        if not self.data_reference_name and (
                path_on_datastore and not self._is_current_path(path_on_datastore)):
            self.data_reference_name = str(uuid.uuid4().hex)
        elif not self.data_reference_name:
            self.data_reference_name = self.datastore.name
        self._validate_mode(mode)

    def path(self, path=None, data_reference_name=None):
        """Create a DataReference instance based on the given path.

        :param path: The path on the datastore.
        :type path: str
        :param data_reference_name: The name of the data reference.
        :type data_reference_name: str
        :return: The data reference object.
        :rtype: azureml.data.data_reference.DataReference
        """
        if not path or self._is_current_path(path):
            if data_reference_name:
                return DataReference(
                    datastore=self.datastore,
                    data_reference_name=data_reference_name)
            else:
                return self

        if not self.path_on_datastore or self._is_current_path(self.path_on_datastore):
            return DataReference(
                datastore=self.datastore,
                data_reference_name=data_reference_name,
                path_on_datastore=path)
        else:
            return DataReference(
                datastore=self.datastore,
                data_reference_name=data_reference_name,
                path_on_datastore="{0}/{1}".format(self.path_on_datastore, path)
            )

    def as_download(self, path_on_compute=None, overwrite=False):
        """Switch data reference operation to download.

        DataReference download only supports Azure Blob and Azure File Share. To download data from Azure Blob,
        Azure File Share, Azure Data Lake Gen1, and Azure Data Lake Gen2 we recommend using Azure Machine Learning
        Dataset. For more information on how to create and use Dataset, please visit
        https://docs.microsoft.com/en-us/azure/machine-learning/how-to-train-with-datasets.

        :param path_on_compute: The path on the compute for the data reference.
        :type path_on_compute: str
        :param overwrite: Indicates whether to overwrite existing data.
        :type overwrite: bool
        :return: A new data reference object.
        :rtype: azureml.data.data_reference.DataReference
        """
        dref = self._clone()
        dref.path_on_compute = path_on_compute
        dref.mode = "download"
        return dref

    def as_upload(self, path_on_compute=None, overwrite=False):
        """Switch data reference operation to upload.

        For more information on which computes and datastores support uploading of the data, see:
        https://aka.ms/datastore-matrix.

        :param path_on_compute: The path on the compute for the data reference.
        :type path_on_compute: str
        :param overwrite: Indicates whether to overwrite existing data.
        :type overwrite: bool
        :return: A new data reference object.
        :rtype: azureml.data.data_reference.DataReference
        """
        dref = self._clone()
        dref.path_on_compute = path_on_compute
        dref.mode = "upload"
        return dref

    def as_mount(self):
        """Switch data reference operation to mount.

        DataReference mount only supports Azure Blob. To mount data in Azure Blob, Azure File Share,
        Azure Data Lake Gen1, and Azure Data Lake Gen2 we recommend using Azure Machine Learning Dataset. For more
        information on how to create and use Dataset, please visit
        https://docs.microsoft.com/en-us/azure/machine-learning/how-to-train-with-datasets.

        :return: A new data reference object.
        :rtype: azureml.data.data_reference.DataReference
        """
        dref = self._clone()
        dref.mode = "mount"
        return dref

    def to_config(self):
        """Convert the DataReference object to DataReferenceConfiguration object.

        :return: A new DataReferenceConfiguration object.
        :rtype: azureml.core.runconfig.DataReferenceConfiguration
        """
        from azureml.core.runconfig import DataReferenceConfiguration
        return DataReferenceConfiguration(
            datastore_name=self.datastore.name,
            mode=self.mode,
            path_on_datastore=self._get_normalized_path(self.path_on_datastore),
            path_on_compute=self._get_normalized_path(self.path_on_compute),
            overwrite=self.overwrite)

    def __str__(self):
        """Return string representation of object.

        :return:
        """
        result = "$AZUREML_DATAREFERENCE_{0}".format(self.data_reference_name)
        return result

    def __repr__(self):
        """Return string representation of object.

        :return:
        """
        return self.__str__()

    def _is_current_path(self, path):
        return path == "/" or path == "." or path == "./"

    def _validate_mode(self, mode):
        from azureml.core.runconfig import SUPPORTED_DATAREF_MODES
        from azureml.exceptions import UserErrorException

        message = "Invalid mode {0}. Only mount, download, upload are supported"
        if mode not in SUPPORTED_DATAREF_MODES:
            raise UserErrorException(message.format(mode))

    def _clone(self):
        return DataReference(
            datastore=self.datastore,
            data_reference_name=self.data_reference_name,
            path_on_compute=self.path_on_compute,
            path_on_datastore=self.path_on_datastore,
            mode=self.mode,
            overwrite=self.overwrite
        )

    def _get_normalized_path(self, path):
        result = path
        if (self._is_current_path(path)):
            result = None
        return result

    @staticmethod
    def create(data_reference_name=None, datapath=None, datapath_compute_binding=None):
        """Create a DataReference using DataPath and DataPathComputeBinding.

        :param data_reference_name: The name for the data reference to create.
        :type data_reference_name: str
        :param datapath: [Required] The datapath to use.
        :type datapath: azureml.data.datapath.DataPath
        :param datapath_compute_binding: [Required] The datapath compute binding to use.
        :type datapath_compute_binding: azureml.data.datapath.DataPathComputeBinding
        :return: A DataReference object.
        :rtype: azureml.data.data_reference.DataReference
        """
        if None in [datapath, datapath_compute_binding]:
            raise ValueError('datapath and datapath_compute_binding are expected parameters')

        return DataReference(datastore=datapath._datastore, path_on_datastore=datapath._path_on_datastore,
                             data_reference_name=data_reference_name,
                             mode=datapath_compute_binding._mode,
                             path_on_compute=datapath_compute_binding._path_on_compute,
                             overwrite=datapath_compute_binding._overwrite)

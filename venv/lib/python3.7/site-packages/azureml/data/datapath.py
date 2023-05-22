# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""Contains functionality to create references to data in datastores.

This module contains the :class:`azureml.data.datapath.DataPath` class, which represents the location of data,
and the :class:`azureml.data.datapath.DataPathComputeBinding` class, which represents how the data is made
available on the compute targets.
"""
import uuid
from .data_reference import DataReference


class DataPath(object):
    """Represents a path to data in a datastore.

    The path represented by DataPath object can point to a directory or a data artifact (blob, file).
    DataPath is used in combination with the :class:`DataPathComputeBinding` class, which defines how
    the data is consumed during pipeline step execution. A DataPath can be modified at during pipeline
    submission with the :class:`azureml.pipeline.core.graph.PipelineParameter`.

    .. remarks::

        The following example shows how to work create a DataPath and pass in arguments to it using
        :class:`azureml.pipeline.core.graph.PipelineParameter`.

        .. code-block:: python

            def_blob_store = ws.get_default_datastore()
            print("Default datastore's name: {}".format(def_blob_store.name))

            data_path = DataPath(datastore=def_blob_store, path_on_datastore='sample_datapath1')
            datapath1_pipeline_param = PipelineParameter(name="input_datapath", default_value=data_path)
            datapath_input = (datapath1_pipeline_param, DataPathComputeBinding(mode='mount'))

            string_pipeline_param = PipelineParameter(name="input_string", default_value='sample_string1')

        Full sample is available from
        https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/machine-learning-pipelines/intro-to-pipelines/aml-pipelines-showcasing-datapath-and-pipelineparameter.ipynb


    :param datastore: [Required] The Datastore to reference.
    :type datastore: typing.Union[azureml.data.azure_storage_datastore.AbstractAzureStorageDatastore,
                    azureml.data.azure_data_lake_datastore.AzureDataLakeDatastore]
    :param path_on_datastore: The relative path in the backing storage for the data reference.
    :type path_on_datastore: str
    :param name: An optional name for the DataPath.
    :type name: str, optional
    """

    # TODO DPrep team will extend this implementation with filters
    def __init__(self, datastore=None, path_on_datastore=None, name=None):
        """Initialize DataPath.

        :param datastore: [Required] The Datastore to reference.
        :type datastore: typing.Union[azureml.data.azure_storage_datastore.AbstractAzureStorageDatastore,
                        azureml.data.azure_data_lake_datastore.AzureDataLakeDatastore]
        :param path_on_datastore: The relative path in the backing storage for the data reference.
        :type path_on_datastore: str
        :param name: An optional name for the DataPath.
        :type name: str, optional
        """
        if None in [datastore]:
            raise ValueError('datastore parameter is required.')

        self._datastore = datastore
        self._path_on_datastore = path_on_datastore
        if not name:
            self._name = '{0}_{1}'.format(self.datastore_name, str(uuid.uuid4().hex)[0:8])
        else:
            self._name = name

    def create_data_reference(self, data_reference_name=None, datapath_compute_binding=None):
        """Create a DataReference object using this DataPath and the given DataPathComputeBinding.

        :param data_reference_name: The name for the data reference to create.
        :type data_reference_name: str
        :param datapath_compute_binding: [Required] The data path compute binding to use to create the data reference.
        :type datapath_compute_binding: azureml.data.datapath.DataPathComputeBinding
        :return: A DataReference object.
        :rtype: azureml.data.data_reference.DataReference
        """
        if not datapath_compute_binding:
            raise ValueError('datapath_compute_binding is a required parameter')

        if not isinstance(datapath_compute_binding, DataPathComputeBinding):
            raise ValueError('Invalid type. Expected datapath_compute_binding, but type is {0}'.
                             format(type(datapath_compute_binding).__name__))

        return DataReference.create(data_reference_name=data_reference_name, datapath=self,
                                    datapath_compute_binding=datapath_compute_binding)

    @property
    def datastore_name(self):
        """Get the name of the datastore.

        :return: The name.
        :rtype: string
        """
        return self._datastore.name

    @property
    def path_on_datastore(self):
        """Get the path on datastore.

        :return: The path.
        :rtype: string
        """
        return self._path_on_datastore

    @staticmethod
    def create_from_data_reference(data_reference):
        """Create a DataPath from a DataReference.

        :param data_reference: [Required] The data reference to use to create data path.
        :type data_reference: azureml.data.data_reference.DataReference
        :return: A DataPath object.
        :rtype: azureml.data.datapath.DataPath
        """
        if not data_reference:
            raise ValueError('data_reference is a required parameter')

        if not isinstance(data_reference, DataReference):
            raise ValueError('Invalid type. Expected DataReference, but type is {0}'.
                             format(type(data_reference).__name__))

        return DataPath(datastore=data_reference.datastore, path_on_datastore=data_reference.path_on_datastore)

    def _serialize_to_dict(self):
        return {"name": self._name,
                "path_on_datastore": self.path_on_datastore,
                "datastore": self.datastore_name}


class DataPathComputeBinding(object):
    """Configure how data defined by :class:`DataPath` is made available on a compute target.

    DataPath configuration indicates how the data will be used on the compute target, that is uploaded or mounted,
    as well as if the data should be overwritten.

    :param mode: The operation on the data reference. "mount" and "download" are supported.
    :type mode: str
    :param path_on_compute: The path on the compute target for the data reference.
    :type path_on_compute: str
    :param overwrite: Indicates whether to overwrite existing data.
    :type overwrite: bool
    """

    def __init__(self, mode='mount', path_on_compute=None, overwrite=False):
        """Initialize DataPathComputeBinding.

        :param mode: The operation on the data reference. "mount" and "download" are supported.
        :type mode: str
        :param path_on_compute: The path on the compute target for the data reference.
        :type path_on_compute: str
        :param overwrite: Indicates whether to overwrite existing data.
        :type overwrite: bool
        """
        self._mode = mode
        self._path_on_compute = path_on_compute
        self._overwrite = overwrite

    def create_data_reference(self, data_reference_name=None, datapath=None):
        """Create a DataReference from a DataPath and this DataPathComputeBinding.

        :param data_reference_name: The name of the data reference to create.
        :type data_reference_name: str
        :param datapath: [Required] The data path to use to create the data reference.
        :type datapath: azureml.data.datapath.DataPath
        :return: A DataReference object.
        :rtype: azureml.data.data_reference.DataReference
        """
        if not datapath:
            raise ValueError('datapath is a required parameter')

        if not isinstance(datapath, DataPath):
            raise ValueError('Invalid type. Expected DataReference, but type is {0}'.
                             format(type(datapath).__name__))

        return DataReference.create(datapath=datapath, datapath_compute_binding=self,
                                    data_reference_name=data_reference_name)

    @staticmethod
    def create_from_data_reference(data_reference):
        """Create a DataPathComputeBinding from a DataReference.

        :param data_reference: [Required] The data reference to use to create the data path compute binding.
        :type data_reference: azureml.data.data_reference.DataReference
        :return: A DataPathComputeBinding object.
        :rtype: azureml.data.datapath.DataPathComputeBinding
        """
        if not data_reference:
            raise ValueError('data_reference is a required parameter')

        if not isinstance(data_reference, DataReference):
            raise ValueError('Invalid type. Expected DataReference, but type is {0}'.
                             format(type(data_reference).__name__))

        return DataPathComputeBinding(mode=data_reference.mode, path_on_compute=data_reference.path_on_compute,
                                      overwrite=data_reference.overwrite)

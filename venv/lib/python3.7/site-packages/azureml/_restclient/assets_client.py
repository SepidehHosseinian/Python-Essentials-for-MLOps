# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Access AssetsClient"""

from azureml._restclient.models import Asset

from .workspace_client import WorkspaceClient
ASSETS_SERVICE_VERSION = "2018-11-19"

RETRY_LIMIT = 3
BACKOFF_START = 2


class AssetsClient(WorkspaceClient):
    """Asset client class"""

    def get_rest_client(self, user_agent=None):
        """get service rest client"""
        return self._service_context._get_assets_restclient(user_agent=user_agent)

    def create_asset(self, model_name, artifact_values, metadata_dict,
                     project_id=None, run_id=None, tags=None, properties=None, asset_type=None):
        """
        :param model_name:
        :type model_name: str
        :param artifact_values:
        :type artifact_values: list
        :param metadata_dict:
        :type metadata_dict: dict
        :param project_id:
        :type project_id: str
        :param run_id:
        :type run_id: str
        :param tags:
        :type tags: dict
        :param properties:
        :type properties: dict
        :param asset_type:
        :type asset_type: str
        :return:
        """
        asset = Asset(name=model_name,
                      artifacts=artifact_values,
                      kv_tags=tags,
                      properties=properties,
                      runid=run_id,
                      meta=metadata_dict,
                      type=asset_type)

        return self._execute_with_workspace_arguments(self._client.assets.create, asset)

    def get_asset_by_id(self, asset_id):
        """
        Get events of a run by its run_id
        :rtype: ~_restclient.models.Asset or ~msrest.pipeline.ClientRawResponse
        """
        return self._execute_with_workspace_arguments(self._client.assets.query_by_id,
                                                      id=asset_id)

    def list_assets(self):
        """
        Get events of a run by its run_id
        :rtype: ~_restclient.models.Asset or ~msrest.pipeline.ClientRawResponse
        """
        return self._execute_with_workspace_arguments(self._client.assets.list_query)

    def _properties_dict_to_query_string(self, properties):
        """
        Convert a given properties dictionary into a string suitable for
        sending in a query.
        If None is passed, None is returned.
        :param properties: The properties whose conversion is considered desirable
        :type properties: dict(str,str)
        :rtype: string
        """
        prop_string = None
        if properties is not None:
            prop_string = ''
            for key in properties.keys():
                if properties[key] == '':
                    raise Exception('The empty string is not a valid property value.')
                else:
                    prop_string += '{}={},'.format(key, str(properties[key]))
            prop_string = prop_string.strip(',')
        return prop_string

    def list_assets_with_query(self, run_id=None, name=None, properties=None, asset_type=None):
        """
        Gets a list of Assets, with an optional query
        :param run_id: The run ID for which Assets are desired (optional)
        :type run_id: str
        :param name: The desired Asset name (optional)
        :type name: str
        :param properties: A dictionary of desired properties (optional)
        :type properties: dict(str,str)
        :param asset_type: The desired Asset type (optional)
        :type asset_type: str
        :return: A generator of Assets from the given run which match the query
        :rtype: generator[~_restclient.models.Asset]
        """
        prop_string = self._properties_dict_to_query_string(properties)
        return self._execute_with_workspace_arguments(self._client.assets.list_query,
                                                      run_id=run_id,
                                                      name=name,
                                                      properties=prop_string,
                                                      type=asset_type,
                                                      is_paginated=True)

    def list_assets_by_properties_run_id_name(self, run_id, name, properties):
        """
        Get assets filtered on run ID and asset name
        :param run_id: the run ID to filter on
        :type run_id: str
        :param name: the asset name to filter on
        :type name: str
        :param properties: A dictionary with the keys and values to filter on. Keys must be strings.
            Values may be strings, booleans, or integers.
        :type properties: dict
        :return: A generator of assets which have run IDs, names, and properties matching the params
        :rtype: generator[~_restclient.models.Asset]
        """
        prop_string = self._properties_dict_to_query_string(properties)
        return self._execute_with_workspace_arguments(self._client.assets.list_query,
                                                      run_id=run_id,
                                                      name=name,
                                                      properties=prop_string).value

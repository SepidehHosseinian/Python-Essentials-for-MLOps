# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Access EnvironmentClient"""

import requests
import urllib.parse
from .workspace_client import WorkspaceClient


class EnvironmentClient(WorkspaceClient):
    """Environment client class"""

    def get_rest_client(self, user_agent=None):
        """get service rest client"""
        return self._service_context._get_environment_restclient(user_agent=user_agent)

    def _get_environment_definition(self, name, version=None):
        """
        :param name:
        :type name: str
        :param version:
        :type version: str
        :return Returns the environment definition object:
        """

        cluster_address = self.get_cluster_url()
        headers = self.auth.get_authentication_header()
        workspace_address = self.get_workspace_uri_path()

        environment_url = cluster_address + "/environment/v1.0" + \
            workspace_address + self._format_env_for_path(name, version)
        # Else Retrieve the latest version of the environment definition

        response = requests.get(environment_url, headers=headers)

        if response.status_code == 200:
            return response.json()
        else:
            message = "Error retrieving the environment definition. Code: {}\n: {}".format(response.status_code,
                                                                                           response.text)
            raise Exception(message)

    def _get_environment_definition_by_label(self, name, label):
        """
        :return Returns True if succeeded
        """

        cluster_address = self.get_cluster_url()
        headers = self.auth.get_authentication_header()
        workspace_address = self.get_workspace_uri_path()
        environment_url = cluster_address + "/environment/v1.0" + \
            workspace_address + self._format_env_for_path(name) + \
            "/labels/" + label

        response = requests.get(environment_url, headers=headers)

        if response.status_code == 200:
            return response.json()
        else:
            message = "Error retrieving the environment definition {} with label {}. "\
                "Code: {}\n: {}".format(name, label, response.status_code, response.text)
            raise Exception(message)

    def _register_environment_definition(self, environment_dict):
        """
        :return Returns the environment definition dictionary from the response:
        """

        name = environment_dict["name"]
        environment_version = environment_dict["version"]
        cluster_address = self.get_cluster_url()
        headers = self.auth.get_authentication_header()
        workspace_address = self.get_workspace_uri_path()
        environment_url = cluster_address + "/environment/v1.0" + \
            workspace_address + self._format_env_for_path(name, environment_version)

        body = {'Name': name,
                'python': environment_dict["python"],
                "EnvironmentVariables": environment_dict["environmentVariables"],
                "Docker": environment_dict["docker"],
                "Spark": environment_dict["spark"],
                "InferencingStackVersion": environment_dict.get("inferencingStackVersion", None),
                "r": environment_dict.get("r", None)
                }

        response = requests.put(
            environment_url, headers=headers, json=body)

        if response.status_code == 200:
            return response.json()
        else:
            message = "Error registering the environment definition. Code: {}\n: {}".format(response.status_code,
                                                                                            response.text)
            raise Exception(message)

    def _set_envionment_definition_labels(self, name, version, labels):
        """
        :return Returns True if succeeded
        """

        cluster_address = self.get_cluster_url()
        headers = self.auth.get_authentication_header()
        workspace_address = self.get_workspace_uri_path()
        environment_url = cluster_address + "/environment/v1.0" + \
            workspace_address + self._format_env_for_path(name, version) + \
            "/metadata"

        body = {'Labels': labels}

        response = requests.put(
            environment_url, headers=headers, json=body)

        if response.status_code == 200:
            return True
        else:
            message = "Error setting labels on the environment definition. Code: {}\n: {}".format(response.status_code,
                                                                                                  response.text)
            raise Exception(message)

    def _get_image_details(self, name, version=None):
        """
        :param name:
        :type name: str
        :param version:
        :type version: str
        :return Returns the image details:
        """

        cluster_address = self.get_cluster_url()
        headers = self.auth.get_authentication_header()
        workspace_address = self.get_workspace_uri_path()

        environment_url = cluster_address + "/environment/v1.0" + \
            workspace_address + self._format_env_for_path(name, version) + \
            "/image"

        response = requests.get(environment_url, headers=headers)

        if response.status_code == 200:
            return response.json()
        else:
            message = "Error getting image details. Code: {}\n: {}".format(response.status_code,
                                                                           response.text)
            raise Exception(message)

    def _list_definitions(self):
        """
        :return Returns the list of environment definitions in the workspace:
        """

        cluster_address = self.get_cluster_url()
        headers = self.auth.get_authentication_header()
        workspace_address = self.get_workspace_uri_path()

        environment_url = cluster_address + "/environment/v1.0" + \
            workspace_address + "/environments/"

        response = requests.get(environment_url, headers=headers)

        error_msg_template = "Error listing environment definitions. Code: {}\n: {}"

        if response.status_code == 200:
            response_json = response.json()
            envs = response_json.get("value")
            continuation = response_json.get("continuationToken")

            while continuation:
                response = requests.get(environment_url, headers=headers, params={'continuation': continuation})
                if response.status_code != 200:
                    message = error_msg_template.format(response.status_code, response.text)
                    raise Exception(message)

                response_json = response.json()
                envs.extend(response_json.get("value"))
                continuation = response_json.get("continuationToken")
            return envs
        else:
            message = error_msg_template.format(response.status_code, response.text)
            raise Exception(message)

    def _get_recipe_for_build(self, name, version=None, **kwargs):
        """
        :param name:
        :type name: str
        :param version:
        :type version: str
        :return Returns the recipe details for image build:
        """

        cluster_address = self.get_cluster_url()
        headers = self.auth.get_authentication_header()
        workspace_address = self.get_workspace_uri_path()

        environment_url = cluster_address + "/environment/v1.0" + \
            workspace_address + self._format_env_for_path(name, version) + \
            "/recipe"

        response = requests.post(environment_url, headers=headers, json=kwargs)

        if response.status_code == 200:
            return response.json()
        else:
            message = "Error getting recipe specifications. Code: {}\n: {}".format(response.status_code,
                                                                                   response.text)
            raise Exception(message)

    def _start_cloud_image_build(self, name, version=None, image_build_compute=None):
        """
        :param name:
        :type name: str
        :param version:
        :type version: str
        :param image_build_compute:
        :type image_build_compute: str
        :return Returns the cloud image build details:
        """

        cluster_address = self.get_cluster_url()
        headers = self.auth.get_authentication_header()
        workspace_address = self.get_workspace_uri_path()

        environment_url = cluster_address + "/environment/v2.0" + \
            workspace_address + self._format_env_for_path(name, version)

        if image_build_compute:
            environment_url += "/imageoncompute/" + image_build_compute
        else:
            environment_url += "/image"

        # empty json to support FromBody
        response = requests.post(environment_url, headers=headers, json={})

        if response.status_code == 202:
            return response.json()
        else:
            message = "Error building image. Code: {}\n: {}".format(response.status_code,
                                                                    response.text)
            self._logger.error(message)
            raise Exception(message)

    def _get_lro_response(self, location):
        """
        :param location:
        :type location: str
        :return Returns the long running operation status:
        """
        headers = self.auth.get_authentication_header()
        response = requests.get(location, headers=headers)

        if response.status_code in (200, 202):
            return response.json()
        else:
            message = "Error getting image build status. Code: {}\n: {}".format(response.status_code,
                                                                                response.text)
            raise Exception(message)

    @staticmethod
    def _format_env_for_path(name, version=None):
        env_path = "/environments/" + urllib.parse.quote(name, safe='')

        if version is not None:
            env_path += "/versions/" + str(version)

        return env_path

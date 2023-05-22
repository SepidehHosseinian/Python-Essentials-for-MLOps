# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------


class PyTokenLibrary:

    # resource = """{"audience": "AzureManagement", "name": ""}"""
    # resource = """{"audience": "storage", "name": ""}"""

    _ARM_RESOURCE = '{"audience": "AzureManagement", "name": ""}'
    _STORAGE_RESOURCE = '{"audience": "storage", "name": ""}'
    # TODO: Not correct for now.
    _GRAPH_RESOURCE = '{"audience": "graph", "name": ""}'

    @staticmethod
    def get_access_token(resource):
        """
        Returns AccessToken(token, type, expireTime, Some(serverName))
        :param resource:
        :type resource: str
        :return:
        """
        from pyspark.sql import SparkSession
        sc = SparkSession.builder.getOrCreate()
        # leverage Py4J to access Scala TokenLibrary
        token_library_new = sc._jvm.com.microsoft.azure.synapse.tokenlibrary.TokenLibrary
        access_token = token_library_new.getAccessToken(resource)
        return access_token

    @staticmethod
    def get_AAD_token(resource):
        """
        Returns AAD token as a string
        :param resource:
        :type resource: str
        :return:
        :rtype: str
        """
        access_token = PyTokenLibrary.get_access_token(resource)
        return access_token.token()

    @staticmethod
    def get_expiry(resource):
        """
        Returns expiry time as a string (Unix epoch in seconds)
        :param resource:
        :type resource: str
        :return:
        """
        access_token = PyTokenLibrary.get_access_token(resource)
        return access_token.expireTime()

    @staticmethod
    def get_token_and_expiry(resource):
        """
        Returns a tuple (AAD token, expiry time)
        :param resource:
        :type resource: str
        :return:
        """
        access_token = PyTokenLibrary.get_access_token(resource)
        return access_token.token(), access_token.expireTime()

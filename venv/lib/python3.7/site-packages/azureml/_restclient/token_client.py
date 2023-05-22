# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""Access token client """
from .clientbase import ClientBase
from .rest_client import RestClient


class TokenClient(ClientBase):
    """Access token client used by AzureMLTokenAuthentication to refresh the token"""
    def __init__(self, auth, host, **kwargs):
        self.rest_client = RestClient(auth, base_url=host)
        super(TokenClient, self).__init__(**kwargs)

    def get_rest_client(self, user_agent=None):
        """get service rest client"""
        return self.rest_client

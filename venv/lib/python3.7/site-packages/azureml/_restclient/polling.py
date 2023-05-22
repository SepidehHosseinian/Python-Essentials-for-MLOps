# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""Access AzureML polling"""

from azureml._restclient.clientbase import ClientBase
from msrestazure.polling.arm_polling import ARMPolling


class AzureMLPolling(ARMPolling):

    def update_status(self):
        """
        Update the current status of the LRO
        This method will leverage ClientBase retry logic which already has a proper handing for 429,
        Specifically this will be used in waiting for metrics ingest long running operation
        """
        ClientBase._execute_func(super(AzureMLPolling, self).update_status)

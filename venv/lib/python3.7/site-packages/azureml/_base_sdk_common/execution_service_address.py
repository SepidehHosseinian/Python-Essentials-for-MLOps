# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import socket
import errno
import os


class ExecutionServiceAddress(object):
    """
    The class defines the address of the execution service to use.
    If $AZUREML_EXECUTION_SERVICE_ADDRESS is defined, then that is returned.
    (format: "http://127.0.0.1:33707")
    Otherwise, if a local execution service is running for debugging, then that is returned.
    Otherwise, the cloud service is returned.
    """
    def __init__(self, cloud_service_address):
        # Check for execution service override
        override_address = os.getenv("AZUREML_EXECUTION_SERVICE_ADDRESS")
        if override_address:
            self.address = override_address
        # Dev path for debugging against a locally-running service rather than the cloud.
        elif os.environ.get("AZUREML_EXECUTION_DEBUG") and self._port_in_use(33707):
            self.address = "http://localhost:33707"
        else:
            self.address = cloud_service_address

    def _port_in_use(self, port):
        probe_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            probe_socket.bind(('127.0.0.1', port))
        except socket.error as e:
            if e.errno == errno.EADDRINUSE:
                return True
        finally:
            probe_socket.close()

        return False

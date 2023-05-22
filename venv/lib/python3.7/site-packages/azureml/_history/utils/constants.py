# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

OUTPUTS_DIR = "./outputs"

# Logs_dir is the default dir to be upload in *realtime* in submitted experiments
LOGS_DIR = "./logs"
# LOGS_AZUREML_DIR is a subdirectory of logs where azureml logs go to
LOGS_AZUREML_DIR = "logs/azureml"

# User log filename
USER_LOG_FILE = "user_log.txt"

# Driver logs created by the user's program
DRIVER_LOG_NAME = "driver_log"

AZUREML_LOGS = "azureml-logs"
AZUREML_LOG_FILE_NAME = "azureml.log"

AZUREML_DRIVER_LOG = "azureml-logs/80_driver_log.txt"
AZUREML_CONTROL_LOG = "azureml-logs/60_control_log.txt"

# key for runstatus
LOG_FILES_FIELD = "logFiles"
RUN_ID_FIELD = "runId"
RUN_CONFIG_FIELD = "runConfiguration"
STATUS_FIELD = "status"

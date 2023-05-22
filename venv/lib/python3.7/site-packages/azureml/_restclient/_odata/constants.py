# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------


# TODO: Consume contracts from services common
PROP_EXISTS_FORMAT_STR = "(Properties/{0} ne null or Properties/{0} eq null)"
PROP_EQ_FORMAT_STR = "(Properties/{0} eq {1})"
TAG_EXISTS_FORMAT_STR = "(Tags/{0} ne null or Tags/{0} eq null)"
TAG_EQ_FORMAT_STR = "(Tags/{0} eq {1})"
STATUS_EQ_FORMAT_STR = "(Status eq {0})"
# TODO: Remove runsource filter after PuP
TYPE_EQ_FORMAT_STR = "(RunType eq {0} or Properties/azureml.runsource eq {0})"
RUNTYPEV2_ORCHESTRATOR_EQ_FORMAT_STR = "(RunTypeV2/Orchestrator eq {0})"
RUNTYPEV2_TRAITS_EQ_FORMAT_STR = "(RunTypeV2/Traits/{0} ne null) or (RunTypeV2/Traits/{0} eq null)"
CREATED_AFTER_FORMAT_STR = "(CreatedUtc ge {0})"
NO_CHILD_RUNS_QUERY = "(ParentRunId eq null)"

AND_OP = "and"
OR_OP = "or"
RUN_ID_EXPRESSION = "RunId eq "
NAME_EXPRESSION = "name eq "
METRIC_TYPE_EXPRESSION = "MetricType eq "
AFTER_TIMESTAMP_EXPRESSION = "CreatedUtc gt "
ORDER_BY_STARTTIME_EXPRESSION = "StartTimeUtc desc"
ORDER_BY_CREATEDTIME_EXPRESSION = "CreatedUtc desc"
ORDER_BY_RUNID_EXPRESSION = "RunId desc"
TARGET_EQ_FORMAT_STR = "(Target eq {0})"

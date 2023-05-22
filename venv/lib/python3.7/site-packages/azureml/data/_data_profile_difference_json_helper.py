# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""A class for helping with DataProfileDifference Json serialization."""

import json


class _DataProfileDifferenceJsonHelper(object):
    """A class for helping with DataProfileDifference Json serialization."""

    @staticmethod
    def to_json(data_profile_difference):
        pod = data_profile_difference.to_pod()
        return json.dumps(pod)

    @staticmethod
    def from_json(json_str):
        import azureml.dataprep as dprep
        dp = dprep.DataProfileDifference()
        dp.column_profile_difference = list()
        pr_list = json.loads(json_str)
        for col in pr_list['columnProfileDifference']:
            dp.column_profile_difference.append(dprep.ColumnProfileDifference.from_pod(col))
        dp.unmatched_column_profiles = pr_list['unmatchedColumnProfiles']
        return dp

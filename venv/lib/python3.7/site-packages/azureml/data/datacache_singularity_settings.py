# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""Contains objects needed for Datacache Singularity settings representation."""

from enum import Enum


class SingularitySlaTier(Enum):
    """Defines Singularity SLA tiers."""

    basic = "Basic"
    standard = "Standard"
    premium = "Premium"


class SingularityPriority(Enum):
    """Defines Singularity job priorities."""

    low = "Low"
    medium = "Medium"
    high = "High"


class DatacacheSingularitySettings(object):
    """Represents a holder for Datacache Singularity settings.

    Use this class when registering or updating DatacacheStore that configures usage of Singularity platform
    for running Datacache hydration jobs.
    """

    _attribute_map = {
        'sla_tier': {'key': 'slaTier', 'type': 'SingularitySlaTier'},
        'priority': {'key': 'priority', 'type': 'SingularityPriority'},
        'instance_type': {'key': 'instanceType', 'type': 'str'},
        'virtual_cluster_arm_id': {'key': 'virtualClusterArmId', 'type': 'str'},
        'locations': {'key': 'locations', 'type': '[str]'},
    }

    def __init__(self, sla_tier=None, priority=None, instance_type=None, virtual_cluster_arm_id=None, locations=None):
        """Create DatacacheSingularitySettings object.

        :param sla_tier: Possible values include: 'Basic', 'Standard', 'Premium'
        :type sla_tier: str or SingularitySlaTier
        :param priority: Possible values include: 'Low', 'Medium', 'High'
        :type priority: str or SingularityPriority
        :param instance_type: Singularity instance type
        :type instance_type: str
        :param virtual_cluster_arm_id: Singularity virtual cluster ARM ID
        :type virtual_cluster_arm_id: str
        :param locations: Locations (region names) where Singularity should run jobs
        :type locations: list[str]
        """
        super(DatacacheSingularitySettings, self).__init__()
        self.sla_tier = sla_tier
        self.priority = priority
        self.instance_type = instance_type
        self.virtual_cluster_arm_id = virtual_cluster_arm_id
        self.locations = locations

# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

from azureml._cli import abstract_subgroup


class DatastoreSubGroup(abstract_subgroup.AbstractSubGroup):
    """This class defines the run sub group."""

    def get_subgroup_name(self):
        """Returns the name of the subgroup.
        This name will be used in the cli command."""
        return "datastore"

    def get_subgroup_title(self):
        """Returns the subgroup title as string. Title is just for informative purposes, not related
        to the command syntax or options. This is used in the help option for the subgroup."""
        return "Commands for managing and using datastores with the Azure ML Workspace"

    def get_nested_subgroups(self):
        """Returns sub-groups of this sub-group."""
        return super(DatastoreSubGroup, self).compute_nested_subgroups(__package__)

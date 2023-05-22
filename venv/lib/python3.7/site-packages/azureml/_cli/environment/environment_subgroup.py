# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------


from azureml._cli import abstract_subgroup


class EnvironmentSubGroup(abstract_subgroup.AbstractSubGroup):
    """This class defines the environment subgroup."""

    def get_subgroup_name(self):
        """Returns the name of the subgroup.
        This name will be used in the cli command."""
        return "environment"

    def get_subgroup_title(self):
        """Returns the subgroup title as string. Title is just for informative purposes, not related
        to the command syntax or options. This is used in the help option for the subgroup."""
        return "Commands to manage environments"

    def get_nested_subgroups(self):
        """Returns subgroups of this subgroup."""
        return super(EnvironmentSubGroup, self).compute_nested_subgroups(__package__)

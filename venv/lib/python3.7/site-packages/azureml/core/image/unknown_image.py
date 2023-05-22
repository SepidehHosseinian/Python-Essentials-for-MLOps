# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""Contains functionality for managing images that are not imported into Azure Machine Learning."""
from .image import Image
from azureml._model_management._constants import UNKNOWN_IMAGE_TYPE, UNKNOWN_IMAGE_FLAVOR


class UnknownImage(Image):
    """Used internally to represent an unknown Image type.

    This class is DEPRECATED.

    This class is used by :class:`azureml.core.image.image.Image` during a list operation to represent
    images that were created by classes that are not imported at the time the list operation is executed.
    """

    _image_type = UNKNOWN_IMAGE_TYPE
    _image_flavor = UNKNOWN_IMAGE_FLAVOR
    # _expected_payload_keys is inherited from the parent class Image

    def _initialize(self, workspace, obj_dict):
        """Initialize the UnknownImage object.

        :param workspace: The workspace object containing the image to retrieve.
        :type workspace: azureml.core.Workspace
        :param obj_dict:
        :type obj_dict: dict
        :return:
        :rtype: None
        :raises: None
        """
        super(UnknownImage, self)._initialize(workspace, obj_dict)
        self.image_flavor = UnknownImage._image_flavor

    @staticmethod
    def image_configuration():
        """Cannot return image configuration since image type is unknown.

        .. note::
            Not implemented.

        :raises: NotImplementedError
        """
        raise NotImplementedError("Cannot create image configuration because specific Image type not imported. \
                                  ie. ContainerImage, AccelContainerImage, IotContainerImage")

    def run(self):
        """Test an image locally. This does not apply to unknown image types.

        .. note::
            Not implemented.

        :raises: NotImplementedError
        """
        raise NotImplementedError("Cannot run image locally because specific Image type not imported. \
                                  ie. ContainerImage, IotContainerImage (AccelContainerImage cannot be run locally)")

from pathvision.core.base import CorePathvision
from pathvision.core.types import Gradient, Segmentation, Trajectory, Dataset, Models

INCORRECT_VALUE = 'INCORRECT_VALUE'

PARAMETER_ERROR_MESSAGE = {
    INCORRECT_VALUE: (
        'Expected {} to one of the following: {}'
    )
}

class ObjectDetection(CorePathvision):
    def ProcessFrames(self,
                      frames,
                      gradient_technique,
                      dataset,
                      trajectory_technique=None,
                      segmentation_technique=None,
                      pre_trained_model=None):

        # Check our parameters are valid types using Enums

        if not (gradient_technique in iter(Gradient)):
            raise ValueError(PARAMETER_ERROR_MESSAGE['INCORRECT_VALUE'].format(gradient_technique, [e for e in Gradient]))

        if not (dataset in iter(Dataset)):
            raise ValueError(PARAMETER_ERROR_MESSAGE['INCORRECT_VALUE'].format(dataset, [e for e in Dataset]))

        if not (trajectory_technique in iter(Trajectory)) and trajectory_technique is not None:
            raise ValueError(PARAMETER_ERROR_MESSAGE['INCORRECT_VALUE'].format(trajectory_technique, [e for e in Trajectory]))

        if not (segmentation_technique in iter(Segmentation)) and segmentation_technique is not None:
            raise ValueError(PARAMETER_ERROR_MESSAGE['INCORRECT_VALUE'].format(segmentation_technique, [e for e in Segmentation]))

        if not (pre_trained_model in iter(Models)) and pre_trained_model is not None:
            raise ValueError(PARAMETER_ERROR_MESSAGE['INCORRECT_VALUE'].format(pre_trained_model, [e for e in Models]))









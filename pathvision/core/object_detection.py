from pathvision.core.base import CorePathvision
from pathvision.core.types import Gradients, Segmentation, Trajectory

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
                      segmentation_technique=None):

        if not (gradient_technique in iter(Gradients)):
            raise ValueError(PARAMETER_ERROR_MESSAGE['INCORRECT_VALUE'].format(gradient_technique, [e for e in Gradients]))

        if not (gradient_technique in iter(Gradients)):
            raise ValueError(PARAMETER_ERROR_MESSAGE['INCORRECT_VALUE'].format(gradient_technique, [e for e in Gradients]))

        if not (gradient_technique in iter(Segmentation)) and segmentation_technique is not None:
            raise ValueError(PARAMETER_ERROR_MESSAGE['INCORRECT_VALUE'].format(gradient_technique, [e for e in Segmentation]))

        if not (gradient_technique in iter(Trajectory)) and trajectory_technique is not None:
            raise ValueError(PARAMETER_ERROR_MESSAGE['INCORRECT_VALUE'].format(gradient_technique, [e for e in Trajectory]))




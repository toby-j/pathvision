import sys
from enum import StrEnum

# TODO: StrEnum is part of python 3.11. Make alternative for older versions.

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    class StrEnum(str):
        pass


class Segmentation(StrEnum):
    PANOPTIC_DEEPLAB = "Panoptic Deeplab"


class Gradients(StrEnum):
    INTEGRATED_GRADIENTS = "Integrated Gradients"
    SMOOTHGRAD = "Smoothgrad"
    VANILLA_GRADIENTS = "Vanilla Gradients"


class Trajectory(StrEnum):
    KALMAN_FILTER = "Kalman Filter"


class Datasets(StrEnum):
    COCO = "COCO"

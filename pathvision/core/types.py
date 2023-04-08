import sys
from enum import StrEnum

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    # As natve StrEnum is only supported in 3.11, so we import a seperate package that handles it if client is on a lesser version
    from strenum import StrEnum


class Segmentation(StrEnum):
    PANOPTIC_DEEPLAB = "Panoptic Deeplab"


class Gradient(StrEnum):
    INTEGRATED_GRADIENTS = "Integrated Gradients"
    SMOOTHGRAD = "Smoothgrad"
    VANILLA_GRADIENTS = "Vanilla Gradients"


class Trajectory(StrEnum):
    KALMAN_FILTER = "Kalman Filter"

class Models(StrEnum):
    FASTERCNN_RESNET50_FPN = "fasterrcnn_resnet50_fpn"

class Dataset(StrEnum):
    COCO = "COCO"

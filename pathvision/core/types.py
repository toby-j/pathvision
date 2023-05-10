import sys

if sys.version_info >= (3, 11):
    print(sys.version_info)
    from enum import StrEnum
else:
    # As natve StrEnum is only supported in 3.11, so we import a seperate package that handles it if client is on a lesser version
    from strenum import StrEnum

class Segmentation(StrEnum):
    PANOPTIC_DEEPLAB = "Panoptic Deeplab"

class Gradient(StrEnum):
    INTEGRATED_GRADIENTS = "IntegratedGradients"
    SMOOTHGRAD = "Smoothgrad"
    VANILLA_GRADIENTS = "VanillaGradients"


class Trajectory(StrEnum):
    KALMAN_FILTER = "KalmanFilter"

class Models(StrEnum):
    FASTERCNN_RESNET50_FPN = "fasterrcnn_resnet50_fpn"

class Labels(StrEnum):
    COCO = "COCO"

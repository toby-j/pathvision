import sys
from PIL import Image
import os
import pathvision.core as pathvision
from pathvision.core.logger import logger as LOGGER

def tester():

    if sys.version_info[0] != 3:
        LOGGER.critical("This module requires Python 3")
        sys.exit(1)

    frame_list = []
    frame_list_dir = os.listdir("debug/test_data/paris_cars/")
    for frame in frame_list_dir:
        frame_list.append(Image.open("debug/test_data/paris_cars/" + frame))

    od = pathvision.ObjectDetection()

    # Two available models, replace pre_trained_model parameter with either:
    # fastercnn_resnet50_fpn
    # maskrcnn_resnet50_fpn
    od.ProcessFrames(frames=frame_list, labels="COCO", gradient_technique="Smoothgrad",
                             trajectory_technique="KalmanFilter", segmentation_technique="Panoptic Deeplab",
                             pre_trained_model="fasterrcnn_resnet50_fpn", model=None, threshold=None,
                             LoadFromDisk=False, log=True, debug=True)

    LOGGER.info("Execution finished: processed {} frames".format(len(frame_list)))

if __name__ == "__main__":
    tester()
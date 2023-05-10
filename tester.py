import math
import time
from collections import Counter

import PIL
from PIL import Image
from matplotlib import pylab as P
import torch
import os
import numpy as np

import pathvision.core as pathvision
from pathvision.core.logger import logger as LOGGER
from pathvision.core.kalman import KalmanBB
from pathvision.core.tracking import calculate_euclidean_distance
import json


def tester():
    frames = []
    frame2 = []
    frames.append(Image.open("pathvision/test/bike_test/image_0.jpg"))
    frames.append(Image.open("pathvision/test/bike_test/image_1.jpg"))
    frames.append(Image.open("pathvision/test/bike_test/image_2.jpg"))
    frames.append(Image.open("pathvision/test/bike_test/image_3.jpg"))
    frames.append(Image.open("pathvision/test/bike_test/image_4.jpg"))
    frames.append(Image.open("pathvision/test/bike_test/image_5.jpg"))
    frames.append(Image.open("pathvision/test/bike_test/image_6.jpg"))
    frames.append(Image.open("pathvision/test/bike_test/image_7.jpg"))
    frames.append(Image.open("pathvision/test/bike_test/image_8.jpg"))
    frames.append(Image.open("pathvision/test/bike_test/image_9.jpg"))

    frame_list = []
    frame_list_dir = os.listdir("debug/test_data/motorway2/")
    for frame in frame_list_dir:
        frame_list.append(Image.open("debug/test_data/motorway2/" + frame))

    frame2.append(Image.open("pathvision/test/frame2.jpg"))
    # frames.append(Image.open("pathvision/test/frame.png"))
    od = pathvision.ObjectDetection()
    kalman_results = od.ProcessFrames(frames=frame_list, labels="COCO", gradient_technique="VanillaGradients",
                             trajectory_technique="KalmanFilter", segmentation_technique="Panoptic Deeplab",
                             pre_trained_model="fasterrcnn_resnet50_fpn", model=None, threshold=None,
                             LoadFromDisk=False, log=True, debug=True)
    print("done")

if __name__ == "__main__":
    tester()
    #
    # results = {
    #     "frame_data": [],
    #     "errors": {}
    # }
    #
    # if not results["errors"].keys() == 5:
    #     results["errors"].setdefault(5, {"kalman": []})
    #
    # print(results)
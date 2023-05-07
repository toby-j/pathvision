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

    frame2.append(Image.open("pathvision/test/frame2.jpg"))
    # frames.append(Image.open("pathvision/test/frame.png"))
    od = pathvision.ObjectDetection()
    image = od.ProcessFrames(frames=frame2, labels="COCO", gradient_technique="VanillaGradients",
                             trajectory_technique="KalmanFilter", segmentation_technique="Panoptic Deeplab",
                             pre_trained_model="fasterrcnn_resnet50_fpn", model=None, threshold=None,
                             LoadFromDisk=False, log=True, debug=True)

# TODO: Kalman filter prediction
if __name__ == "__main__":
    # Able to put bounding box in to class id

    class_idxs1 = [1, 2, 3, 3, 3, 3, 2]
    bb_boxes1 = [[100, 100, 100, 100], [200, 200, 200, 200], [300, 300, 300, 300], [400, 400, 400, 400],
                 [500, 500, 500, 500], [600, 600, 600, 600], [700, 700, 700, 700]]

    # class_idxs2 = [1, 2, 3, 3, 3, 3, 2]
    # bb_boxes2 = [[100, 100, 100, 100], [200, 200, 200, 200], [300, 300, 300, 300], [400, 400, 400, 400],
    #              [500, 500, 500, 500], [600, 600, 600, 600], [700, 700, 700, 700]]
    #
    class_idxs3 = [1, 1, 1, 1, 1, 1, 1]
    bb_boxes3 = [[200, 200, 200, 200], [300, 300, 300, 300], [400, 400, 400, 400],
                 [500, 500, 500, 500], [600, 600, 600, 600], [100, 100, 100, 100], [700, 700, 700, 700]]

    kalman_tracker = {}

    def _euclidean_distance(x1, y1, x2, y2):
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


    def _rank_boxes(bboxes, target_bbox):
        target_x, target_y = (target_bbox[0] + target_bbox[2]) / 2, (target_bbox[1] + target_bbox[3]) / 2
        distances = []
        for idx, bbox in enumerate(bboxes):
            bbox_x, bbox_y = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
            distance = _euclidean_distance(target_x, target_y, bbox_x, bbox_y)
            distances.append((bbox, idx, distance))
        distances.sort(key=lambda x: x[2])
        ranked_bboxes = [(dist[0], dist[1], rank + 1) for rank, dist in enumerate(distances)]
        return ranked_bboxes


    def test_kalman_tracker(class_idxs, bb_boxes):
        # For each unique class that our model has seen
        for class_idx in list(set(class_idxs)):
            # For this class, collect the indexes of where this class is in class_idxs
            class_count = [i for i in range(len(class_idxs)) if class_idxs[i] == class_idx]
            class_bb_boxes = []
            # Collect all boxes for each class
            for box_idx in class_count:
                class_bb_boxes.append(bb_boxes[box_idx])
            # For each class there's already being tracked
            if class_idx in kalman_tracker:
                # We create a copy because we'll remove boxes that have been appended to a tracked object.
                # Therefore, any left over in this variable is our new objects that the model is not yet tracking.
                boxes_to_place = class_bb_boxes.copy()
                object_bbs = []
                # For how many objects of this class we're already tracking, we collect the end box of each. We then
                # have the latest position of these tracked objects. We use these to decide where to place our new
                # boxes.
                for key in kalman_tracker[class_idx].keys():
                    # Get the last box of each object
                    object_bbs.append(kalman_tracker[class_idx][key][0][:1][0])

                # For each existing end box, see which of the new boxes is most appropriate to append
                for i, box in enumerate(object_bbs):
                    ranked_bboxes = _rank_boxes(boxes_to_place, box)
                    # Append new location to object

                    kalman_tracker[class_idx][str(i)].append(ranked_bboxes[0][0])
                    boxes_to_place.remove(ranked_bboxes[0][0])
                # For remaining boxes, initialise a new object inside the class to begin tracking
                for box in boxes_to_place:
                    kalman_tracker.setdefault(class_idx, {})[len(kalman_tracker[class_idx].keys())+1] = [[box]]

            else:
                # We're not yet tracking this class.
                # For our box we have for this class, initialise a new tracking entry.
                for i, bb_box in enumerate(class_bb_boxes):
                    kalman_tracker.setdefault(class_idx, {})[str(i)] = [[bb_box]]

    print("-----ITERATION 1-----")
    test_kalman_tracker(class_idxs1, bb_boxes1)

    print("-----ITERATION 2-----")
    test_kalman_tracker(class_idxs3, bb_boxes3)

    print(json.dumps(kalman_tracker, indent=4, separators=(',', ': ')).replace('[\n    [',
                                                                               '[[[').replace(
        ']\n    ]', ']]]').replace(',\n        ', ', '))

    # tester()

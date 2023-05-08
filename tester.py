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
from pathvision.core.tracking import calculate_error
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

    class_idxs2 = [1]
    bb_boxes2 = [[110, 110, 110, 110]]

    class_idxs3 = [1]
    bb_boxes3 = [[120, 120, 120, 120]]

    class_idxs4 = [1]
    bb_boxes4 = [[130, 130, 130, 130]]

    class_idxs5 = [1]
    bb_boxes5 = [[140, 140, 140, 140]]


    tracking = {}
    kalman_tracker = {}
    fps = 300.
    dT = (1 / fps)

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

    def _create_tracking_entry(class_idx, box):
        if len(kalman_tracker[class_idx].keys()) == 0:
            obj_dict = kalman_tracker[class_idx].setdefault("0", {})
        else:
            obj_dict = kalman_tracker[class_idx].setdefault(str(len(kalman_tracker[class_idx].keys()) + 1), {})
        obj_dict['boxes'] = [[box]]
        obj_dict.setdefault('kalman', KalmanBB()).init(np.float32(box))


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
                    object_bbs.append(kalman_tracker[class_idx][key]['boxes'][0][:1][0])

                # For each existing end box, see which of the new boxes is most appropriate to append
                for i, box in enumerate(object_bbs):
                    ranked_bboxes = _rank_boxes(boxes_to_place, box)
                    # Append new location to object
                    kalman = kalman_tracker[class_idx][str(i)]["kalman"]
                    # We initialise with previously known boxes
                    # kalman.iterate(np.float32(box))

                    # Kalman test
                    x, y, w, h = ranked_bboxes[0][0]
                    tracked = kalman.track(x, y, w, h, dT)
                    print("tracked position: {}".format(tracked))
                    pred = kalman.predict()
                    print("predicted position: {}".format(pred))
                    print("Models bb: {}".format(ranked_bboxes[0][0]))

                    error = calculate_error(pred, ranked_bboxes[0][0])

                    print(error)

                    kalman_tracker[class_idx][str(i)]['boxes'].append(ranked_bboxes[0][0])

                    boxes_to_place.remove(ranked_bboxes[0][0])

                    # If we have less new boxes than total tracked objects, we'll run out of new boxes.
                    if len(boxes_to_place) == 0:
                        break

                # For remaining boxes, initialise a new object inside the class to begin tracking
                for box in boxes_to_place:
                    _create_tracking_entry(class_idx, box)

            else:
                # We're not yet tracking this class.
                # For our box we have for this class, initialise a new tracking entry.
                kalman_tracker.setdefault(class_idx, {})
                for box in class_bb_boxes:
                    _create_tracking_entry(class_idx, box)


    # bb_boxes = [100, 100, 100, 100]
    # class_idxs = [1]
    # for i in range(1, 150):
    #     bb = [[]]
    #     for v in bb_boxes:
    #         bb[0].append(v*i)
    #     test_kalman_tracker(class_idxs, bb)

    tester()
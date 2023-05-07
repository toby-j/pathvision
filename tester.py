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


# Kalman object tracker algorithm
# One iteration is working. Next is to implement multi-iteration approach.
if __name__ == "__main__":
    # Able to put bounding box in to class id

    class_idxs1 = [1, 2, 3, 3, 3, 3, 2]
    bb_boxes1 = [[100, 100, 100, 100], [200, 200, 200, 200], [300, 300, 300, 300], [400, 400, 400, 400],
                 [500, 500, 500, 500], [600, 600, 600, 600], [700, 700, 700, 700]]

    # class_idxs2 = [1, 2, 3, 3, 3, 3, 2]
    # bb_boxes2 = [[100, 100, 100, 100], [200, 200, 200, 200], [300, 300, 300, 300], [400, 400, 400, 400],
    #              [500, 500, 500, 500], [600, 600, 600, 600], [700, 700, 700, 700]]
    #
    class_idxs3 = [1, 1, 1, 1, 1]
    bb_boxes3 = [[100, 100, 100, 100], [100, 100, 100, 100], [100, 100, 100, 100], [100, 100, 100, 100], [100, 100, 100, 100]]

    kalman_tracker = {}


    def _euclidean_distance(x1, y1, x2, y2):
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


    def _rank_boxes(bboxes, target_bbox):
        print("TARGET BOX: {}".format(target_bbox))
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
        # Finding duplicates in the model's found classes
        counter = Counter(class_idxs)
        duplicates = [{"class_idx": idx, "count": count} for idx, count in counter.items() if count > 1]
        # Have we seen multiple of the same class?
        for class_idx in list(set(class_idxs)):
            class_count = [i for i in range(len(class_idxs)) if class_idxs[i] == class_idx]
            class_bb_boxes = []
            # Collect all boxes for each class
            for box_idx in class_count:
                class_bb_boxes.append(bb_boxes[box_idx])
            if class_idx in kalman_tracker:
                # New objects in the frame
                number_of_new_boxes = len(kalman_tracker.get(class_idx).keys()) - len(class_bb_boxes)
                # We're going to iterate the new boxes
                boxes_to_place = class_bb_boxes.copy()
                object_bbs = []

                for key in kalman_tracker[class_idx].keys():
                    # Get the last box of each object
                    object_bbs.append(kalman_tracker[class_idx][key][0][:1][0])

                # Go through existing boxes and see what of the new boxes match
                for i, box in enumerate(object_bbs):
                    ranked_bboxes = _rank_boxes(boxes_to_place, box)

                    kalman_tracker[class_idx][str(i)].append(ranked_bboxes[0][0])
                    boxes_to_place.remove(ranked_bboxes[0][0])

                for box in boxes_to_place:
                    kalman_tracker.setdefault(class_idx, {})[len(kalman_tracker[class_idx].keys())+1] = [[box]]

            else:
                for i, bb_box in enumerate(class_bb_boxes):
                    kalman_tracker.setdefault(class_idx, {})[str(i)] = [[bb_box]]

        # Now we've got the bounding boxes for the class
        # We've got all the bounding boxes per class

        # We need to check if there's multiple of the same label in pre[0]['lebels']. if there is, we know we're looking at two objects.
        # Their prediction scores might change, so we can't use index. Best approach is to use distance between bounding boxes to decide which one it belongs to.

        # if class_idx in kalman_tracker:
        #
        #     # Collect known end bounding boxes for this object type
        #     object_bbs = []
        #     # Are we already tracking more than one of these objects?
        #
        #
        #
        #     if len(kalman_tracker.get(class_idx).keys()) > 1:
        #         # Collect up all the end boxes, so we can decide where to append our new box
        #         for key in kalman_tracker[class_idx]:
        #             object_bbs.append(kalman_tracker[class_idx][key][0][:1][0])
        #         ranked_boxes = _rank_boxes(object_bbs, bb_box)
        #         new_objects = len(kalman_tracker.get(class_idx).keys()) - len(ranked_boxes)
        #         # Is there any new objects on screen?
        #         if new_objects > 0:
        #             kalman_tracker[class_idx][len(kalman_tracker.get(class_idx).keys()) + 1] = [
        #                 [bb_box]]
        #         best_box = ranked_boxes[0]
        #         print("best box: {}".format(best_box))
        #     else:
        #         kalman_tracker[class_idx]["1"] = [[bb_box]]
        # else:
        #     kalman_tracker[class_idx] = {"0": [[bb_box]]}


    print("-----ITERATION 1-----")
    test_kalman_tracker(class_idxs1, bb_boxes1)

    print("-----ITERATION 2-----")
    test_kalman_tracker(class_idxs3, bb_boxes3)

    print(json.dumps(kalman_tracker, indent=4, separators=(',', ': ')).replace('[\n    [',
                                                                               '[[[').replace(
        ']\n    ]', ']]]').replace(',\n        ', ', '))

    # tester()

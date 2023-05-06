import math
import time
from collections import Counter

import PIL
from PIL import Image
from matplotlib import pylab as P
import torch
import os
import numpy as np
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

    class_idxs = [1, 2, 3, 3, 3, 3, 2]
    bb_boxes = [[349, 230, 203, 323], [432, 424, 645, 323], [243, 112, 462, 323], [243, 112, 462, 323], [243, 112, 462, 323], [243, 112, 462, 323], [243, 112, 462, 323]]

    kalman_tracker = {}

    # Finding duplicates in the model's found classes
    counter = Counter(class_idxs)
    duplicates = [{"class_idx": idx, "count": count} for idx, count in counter.items() if count > 1]
    # Have we seen multiple of the same class?
    print("duplicates: {}".format(duplicates))

    for i, class_idx in enumerate(class_idxs):
        bb_box = bb_boxes[i]

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

        # We need to check if there's multiple of the same label in pre[0]['lebels']. if there is, we know we're looking at two objects.
        # Their prediction scores might change, so we can't use index. Best approach is to use distance between bounding boxes to decide which one it belongs to.

        if class_idx in kalman_tracker:
            if len(duplicates) > 0:
                for x, dup in enumerate(duplicates):
                    # Collect known end bounding boxes for this object type
                    object_bbs = []
                    print("dup: {}".format(dup))
                    print("x: {}".format(x))
                    # Are we already tracking more than one of these objects?
                    if len(kalman_tracker.get(dup['class_idx']).keys()) > 1:
                        for key in kalman_tracker[dup['class_idx']]:
                            # Collect up all the end boxes, so we can decide where to append our new box
                            object_bbs.append(kalman_tracker[dup['class_idx']][key][0][:1][0])
                        ranked_boxes = _rank_boxes(object_bbs, bb_box)
                        print(dup['count'])
                        print(len(ranked_boxes))
                        new_objects = dup['count'] - len(ranked_boxes)
                        # Is there any new objects on screen?
                        if new_objects > 0:
                            kalman_tracker[class_idx][len(kalman_tracker.get(dup['class_idx']).keys()) + 1] = [[bb_box]]
                    else:
                        # ranked_boxes = _rank_boxes(kalman_tracker.get(dup_idx).get("0")[0][:1],
                        #                            bb_box)
                        # # Update the object with new box
                        # print(ranked_boxes)
                        # closest_bbox, closest_idx, closest_rank = ranked_boxes[0]
                        kalman_tracker[dup['class_idx']][int(max(kalman_tracker.get(dup['class_idx']).keys())) + 1] = [[bb_box]]
            else:
                closest_bbox, idx = _rank_boxes(kalman_tracker[class_idx], bb_box)
                kalman_tracker[class_idx][idx] = closest_bbox
        else:
            kalman_tracker[class_idx] = {"0": [[bb_box]]}
    print(json.dumps(kalman_tracker, indent=4, separators=(',', ': ')).replace('[\n    [', '[[[').replace(
        ']\n    ]', ']]]').replace(',\n        ', ', '))



# Copyright 2023 Toby Johnson. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from pathvision.core import kalman
from pathvision.core.utils import write_to_csv
import numpy as np
from pathvision.core.logger import logger as LOGGER
import json
import pprint

def _euclidean_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def calculate_euclidean_distance(kalman_prediction, model_bbox) -> float:
    """
    Calculates the error between the predicted bounding box from the Kalman filter
    and the actual bounding box from the object detection model.

    Args:
        kalman_prediction (tuple): Predicted bounding box from the Kalman filter.
            The bounding box is represented as a tuple (x, y, w, h).
        model_bbox (tuple): Actual bounding box from the object detection model.
            The bounding box is represented as a tuple (x, y, w, h).

    Returns:
        bool: If the difference in the boxes meets our threshold
    """

    # Calculate the center coordinates of the predicted and actual bounding boxes
    kalman_center = (kalman_prediction[0] + kalman_prediction[2] / 2, kalman_prediction[1] + kalman_prediction[3] / 2)
    model_center = (model_bbox[0] + model_bbox[2] / 2, model_bbox[1] + model_bbox[3] / 2)

    # Calculate the Euclidean distance between the center coordinates
    return math.sqrt((kalman_center[0] - model_center[0]) ** 2 + (kalman_center[1] - model_center[1]) ** 2)


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

def getpos(i, x,y,w,h):
    return(np.float32([x[i],y[i],w[i],h[i]]))

def _create_tracking_entry(class_idx, box, kalman_tracker):
    if len(kalman_tracker[class_idx].keys()) == 0:
        obj_dict = kalman_tracker[class_idx].setdefault("0", {})
    else:
        obj_dict = kalman_tracker[class_idx].setdefault(str(len(kalman_tracker[class_idx].keys())), {})
    if type(box) == "list":
        box = box.tolist()
    else:
        obj_dict['boxes'] = [[box]]
    obj_dict.setdefault('kalman', kalman.KalmanBB()).init(box)

def iterate_kalman_tracker(class_idxs, bb_boxes, kalman_tracker):
    # We manually assume the framerate
    fps = 24.
    dT = (1 / fps)
    class_errors = []
    # For each unique class that our model has seen
    for class_idx in set(class_idxs.tolist()):
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
            # Convert our box tensor to lists, so we can safety remove top rank boxes once they've been appended
            boxes_to_place = [box.tolist() for box in boxes_to_place]
            bbs_and_key = []
            # For how many objects of this class we're already tracking, we collect the end box of each. We then
            # have the latest position of these tracked objects. We use these to decide where to place our new
            # boxes.

            for key in kalman_tracker[class_idx].keys():
                # Get the last box of each object
                bbs_and_key.append([kalman_tracker[class_idx][key]['boxes'][0][:1][0], key])

            # For each existing end box, see which of the new boxes is most appropriate to append
            for box_and_key in bbs_and_key:
                ranked_bboxes = _rank_boxes(boxes_to_place,box_and_key[0])

                kalman = kalman_tracker[class_idx][box_and_key[1]]["kalman"]
                x, y, w, h = ranked_bboxes[0][0]

                tracked = kalman.track(x, y, w, h, dT)

                LOGGER.debug("tracked position: {}".format(tracked))
                LOGGER.debug("Models bounding boxes: {}".format(ranked_bboxes[0][0]))

                distance = calculate_euclidean_distance(tracked, ranked_bboxes[0][0])

                LOGGER.debug("Distance between boxes: {}".format(distance))

                if distance > 1500:
                    LOGGER.debug("Distance was over 1000: {} for class".format(distance, class_idx))
                    write_to_csv(0, 2, distance, "kalman_tracker_log")
                    class_errors.append([class_idx, tracked, ranked_bboxes[0][0], distance])

                kalman_tracker[class_idx][box_and_key[1]]['boxes'].append(ranked_bboxes[0][0])

                boxes_to_place.remove(ranked_bboxes[0][0])

                # If we have less new boxes than total tracked objects, we'll run out of new boxes.
                if len(boxes_to_place) == 0:
                    break
            # pprint.pprint(kalman_tracker)

            # For remaining boxes, initialise a new object inside the class to begin tracking
            for box in boxes_to_place:
                # Are any of these new objects very close to any previously tracked?
                _create_tracking_entry(class_idx, box, kalman_tracker)

        else:
            # We're not yet tracking this class.
            # For our box we have for this class, initialise a new tracking entry.
            kalman_tracker.setdefault(class_idx, {})
            for box in class_bb_boxes:
                _create_tracking_entry(class_idx, box, kalman_tracker)

    return kalman_tracker, class_errors

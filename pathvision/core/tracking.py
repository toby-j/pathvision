import math
from pathvision.core import kalman
import numpy as np

def _euclidean_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def calculate_error(kalman_prediction, model_bbox) -> bool:
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
    inaccuracy = math.sqrt((kalman_center[0] - model_center[0])**2 + (kalman_center[1] - model_center[1])**2)
    print("inaccuracy: {}".format(inaccuracy))
    if inaccuracy > 100:
        return True
    else:
        return False

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


def _create_tracking_entry(class_idx, box, kalman_tracker):
    if len(kalman_tracker[class_idx].keys()) == 0:
        obj_dict = kalman_tracker[class_idx].setdefault("0", {})
    else:
        obj_dict = kalman_tracker[class_idx].setdefault(str(len(kalman_tracker[class_idx].keys())), {})
    obj_dict['boxes'] = [[box]]
    obj_dict.setdefault('kalman', kalman.KalmanBB()).init(np.float32(box))


def iterate_kalman_tracker(class_idxs, bb_boxes, kalman_tracker) -> dict:
    fps = 300.
    dT = (1 / fps)
    # For each unique class that our model has seen
    print(set(class_idxs.tolist()))
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

                pred = kalman.predict()
                # Kalman test
                x, y, w, h = ranked_bboxes[0][0]
                print("box locations: {} {} {} {}".format(x, y, w, h,))
                tracked = kalman.track(x, y, w, h)
                print("tracked position: {}".format(tracked))
                print("predicted position: {}".format(pred))
                print("Models bb: {}".format(ranked_bboxes[0][0]))

                error = calculate_error(pred, ranked_bboxes[0][0])

                kalman_tracker[class_idx][str(i)]['boxes'].append(ranked_bboxes[0][0])

                print(type(ranked_bboxes[0][0]))

                boxes_to_place.remove(ranked_bboxes[0][0])

                # If we have less new boxes than total tracked objects, we'll run out of new boxes.
                if len(boxes_to_place) == 0:
                    break

            # For remaining boxes, initialise a new object inside the class to begin tracking
            for box in boxes_to_place:
                _create_tracking_entry(class_idx, box, kalman_tracker)

        else:
            # We're not yet tracking this class.
            # For our box we have for this class, initialise a new tracking entry.
            kalman_tracker.setdefault(class_idx, {})
            for box in class_bb_boxes:
                _create_tracking_entry(class_idx, box, kalman_tracker)

    return kalman_tracker
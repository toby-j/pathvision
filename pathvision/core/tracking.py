import math


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


def iterate_kalman_tracker(class_idxs, bb_boxes, kalman_tracker, tracking):
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
                # If we have less new boxes than total tracked objects, we'll run out of new boxes.
                if len(boxes_to_place) == 0:
                    break
            # For remaining boxes, initialise a new object inside the class to begin tracking
            for box in boxes_to_place:
                kalman_tracker.setdefault(class_idx, {})[len(kalman_tracker[class_idx].keys()) + 1] = [[box]]

        else:
            # We're not yet tracking this class.
            # For our box we have for this class, initialise a new tracking entry.
            for i, bb_box in enumerate(class_bb_boxes):
                kalman_tracker.setdefault(class_idx, {})[str(i)] = [[bb_box]]
        return kalman_tracker

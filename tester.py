from PIL import Image
import os

import pathvision.core as pathvision


def tester():
    frame_list = []
    frame_list_dir = os.listdir("debug/test_data/motorbikes/")
    for frame in frame_list_dir:
        frame_list.append(Image.open("debug/test_data/motorbikes/" + frame))

    od = pathvision.ObjectDetection()
    kalman_results = od.ProcessFrames(frames=frame_list, labels="COCO", gradient_technique="VanillaGradients",
                             trajectory_technique="KalmanFilter", segmentation_technique="Panoptic Deeplab",
                             pre_trained_model="fasterrcnn_resnet50_fpn", model=None, threshold=None,
                             LoadFromDisk=False, log=True, debug=True)
    print("done")

if __name__ == "__main__":
    tester()
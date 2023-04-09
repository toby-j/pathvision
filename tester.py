import PIL
from PIL import Image
import pathvision.core as pathvision

def tester():
    frames = []
    frames.append(Image.open("pathvision/test/frame2.jpg"))
    # frames.append(Image.open("pathvision/test/frame.png"))
    od = pathvision.ObjectDetection()
    image = od.ProcessFrames(frames=frames, labels="COCO", gradient_technique="Vanilla Gradients",
                                             trajectory_technique="Kalman Filter", segmentation_technique="Panoptic Deeplab",
                                             pre_trained_model="fasterrcnn_resnet50_fpn", model=None, threshold=None, LoadFromDisk=True)
    image.save('pathvision/test/result_images/output.jpg')


if __name__ == "__main__":
    tester()
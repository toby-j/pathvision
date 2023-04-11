import PIL
from PIL import Image
import pathvision.core as pathvision
from matplotlib import pylab as P
import torch
import os
import numpy as np

def tester():
    frames = []
    frames.append(Image.open("pathvision/test/frame2.jpg"))
    # frames.append(Image.open("pathvision/test/frame.png"))
    od = pathvision.ObjectDetection()
    image = od.ProcessFrames(frames=frames, labels="COCO", gradient_technique="Smoothgrad",
                                             trajectory_technique="KalmanFilter", segmentation_technique="Panoptic Deeplab",
                                             pre_trained_model="fasterrcnn_resnet50_fpn", model=None, threshold=None, LoadFromDisk=True, log=True)
    image.save('pathvision/test/result_images/output4.png')


if __name__ == "__main__":
    # def ShowImage(im, title='', ax=None):
    #     if ax is None:
    #         fig, ax = P.subplots()
    #     if isinstance(im, Image.Image):  # check if image is PIL image
    #         im = np.asarray(im)
    #     ax.imshow(im.squeeze())
    #     ax.set_title(title)
    #     ax.axis('off')
    #     return ax

    #
    # smoothgrad_folder = 'pathvision/test/outs/smoothgrad/'
    # smoothgrad_disk = []
    # for filename in os.listdir(smoothgrad_folder):
    #     if filename.endswith('.csv'):
    #         smoothgrad_arr = np.loadtxt(os.path.join(smoothgrad_folder, filename), delimiter=',')
    #         smoothgrad_disk.append(smoothgrad_arr)
    #
    # ShowImage(smoothgrad_disk[0])
    tester()



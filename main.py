import numpy as np

import pathvision.core as pathvision
import boto3
import botocore
import os
import io
import zipfile
from PIL import Image
s3_url = 's3://pvworkspacedata180802-staging/public/frames/{}.zip'.format(os.environ.get("WORKSPACE_UUID"))


# Extract bucket and key from S3 URL
bucket, key = s3_url.split('/')[2], '/'.join(s3_url.split('/')[3:])

s3 = boto3.resource("s3")
bucket_obj = s3.Bucket(bucket)
obj = bucket_obj.Object(key)

frames = []  # Array to store PIL images

with io.BytesIO(obj.get()["Body"].read()) as tf:
    # rewind the file
    tf.seek(0)

    # Read the file as a zipfile and process the members
    print(tf)
    with zipfile.ZipFile(tf, mode='r') as zipf:
        for subfile in zipf.namelist():
            with zipf.open(subfile) as image_file:
                image = Image.open(image_file)
                frames.append(np.array(image))

od = pathvision.ObjectDetection()
pil_frames = []
for frame in frames:
    pil_frames.append(Image.fromarray(frame))



kalman_results = od.ProcessFrames(frames=pil_frames, labels="COCO",
                                  gradient_technique="VanillaGradients",
                                  trajectory_technique="KalmanFilter", segmentation_technique="Panoptic Deeplab",
                                  pre_trained_model="fasterrcnn_resnet50_fpn", model=None, threshold=None,
                                  LoadFromDisk=False, log=True, debug=True)

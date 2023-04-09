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
import sys

import PIL
import cv2
from PIL import Image, ImageOps, ImageEnhance
import numpy as np
import os

from numpy import savetxt
from pycocotools.coco import COCO
from torchvision.transforms import ToPILImage
from torchvision.utils import save_image

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
import pathvision.core as pathvision
from pathvision.core.visualisation import VisualizeImageGrayscale

to_pil = ToPILImage()
import torchvision.transforms.functional as TF
from pathvision.core.base import CorePathvision, INPUT_OUTPUT_GRADIENTS
from pathvision.core.types import Gradient, Segmentation, Trajectory, Labels, Models
from pathvision.core.logger import logger as LOGGER
import torchvision
import torch

'''
TODO: Run the object detection model on the images.
TODO: Run selected techniques on the images
TODO: Send back processed images, with result dict to analyse
'''

INCORRECT_VALUE = 'INCORRECT_VALUE'
EMPTY_FRAMES = 'EMPTY_FRAMES'
INCORRECT_CONFIG = 'INCORRECT_CONFIG'
NO_MODEL = 'NO_MODEL'

PARAMETER_ERROR_MESSAGE = {
    INCORRECT_VALUE: (
        'Expected {} to one of the following: {}'
    ),
    EMPTY_FRAMES: (
        'frames must contain a list of images'
    ),
    INCORRECT_CONFIG: (
        'Please select either {} or {}'
    ),
    NO_MODEL: (
        'Model not loaded correctly, please refer to the spesification for uploading your own model, '
        'or use a pre-trained model'
    )
}


# Utility Functions

def _preprocess_image(im):
    # assumes input is 4-D, with range [0,255]
    #
    # torchvision have color channel as first dimension
    # with normalization relative to mean/std of ImageNet:
    #    https://pytorch.org/vision/stable/models.html

    im = np.array(im)
    im = im / 255
    im = np.transpose(im, (0, 3, 1, 2))
    im = torch.tensor(im, dtype=torch.float32)
    # images = transformer.forward(images)
    return im.requires_grad_(True)


def _pre_process_model_output(preds, threshold=0.8, coco=None):
    # Get the scores as a NumPy array
    scores = preds[0]['scores'].detach().numpy()

    # Filter the boxes, labels, and scores based on the threshold and indices to remove
    idx_to_keep = np.where(scores > threshold)[0]
    boxes = preds[0]['boxes'][idx_to_keep]
    # For some reason, the labels are float. So we're converting them to ints.
    labels = torch.tensor([int(label) for label in preds[0]['labels'][idx_to_keep]])
    scores = preds[0]['scores'][idx_to_keep]

    # Convert the labels to strings and add the scores
    annot_labels = ["{:.1f} - {}".format(scores[i], coco.loadCats(int(labels[i]))[0]['name']) for i in
                    range(len(labels))]

    # Replace the boxes, labels, and scores with the filtered ones
    preds[0]['boxes'] = boxes
    preds[0]['labels'] = labels
    preds[0]['scores'] = scores

    return preds, annot_labels


def _crop_frame(frame, box):
    x1, y1, x2, y2 = map(int, box)
    cropped_tensor = TF.crop(frame, y1, x1, y2 - y1, x2 - x1)
    cropped_tensor = cropped_tensor.squeeze(0)
    cropped_image = to_pil(cropped_tensor)
    return cropped_image, (x1, y1, x2, y2)


# Return PIL image as np array, or a image provided a path.
def _load_image_arr(file_path='', pil_img=None):
    if file_path != '':
        im = Image.open(file_path)
    elif pil_img:
        im = pil_img
    else:
        raise Exception("Unable to convert image to array")
    im = np.asarray(im)
    return im


def _load_image(im):
    im = im.convert("RGB")
    im = TF.pil_to_tensor(im)
    frame_int = im.unsqueeze(dim=0)
    return frame_int

def _reduce_opacity(im, opacity):
    """
    Returns an image with reduced opacity.
    Taken from http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/362879
    """
    assert opacity >= 0 and opacity <= 1
    if im.mode != 'RGBA':
        im = im.convert('RGBA')
    else:
        im = im.copy()
    alpha = im.split()[3]
    alpha = ImageEnhance.Brightness(alpha).enhance(opacity)
    im.putalpha(alpha)
    return im


class ObjectDetection(CorePathvision):
    r"""Object Detection supported features."""

    def ProcessFrames(self,
                      frames,
                      labels,
                      gradient_technique,
                      trajectory_technique,
                      segmentation_technique=None,
                      pre_trained_model=None
                      , model=None,
                      threshold=None,LoadFromDisk=False):

        """Detects the objects in the frames. Uses trajectory prediction to find frames with errors. If errors are found,
        segmentation technique and the gradient technique are used on the frames to attempt to uncover the reason for the error.

        Args:
          frames: List of frames to be processed
          labels: dataset
          gradient_technique: Gradient technique to be applied to each frame
          trajectory_technique: Trajectory prediction to highlight core inaccuracies
          segmentation_technique: optional: Segmentation algorithm for the debug process with gradient techniques
          pre_trained_model: optional: a pre-trained model from tensorflow's torchvision, model must be false
          model: optional: File path to user's model, pre_trained_model must be false.
          threshold: optional: percentage of what predictions the user is interested in. Objects classified with less than the threshold will be ignored
          LoadFromDisk: DEBUG: Loading the processed gradient numpys from disk to avoid calculating them from scratch again
        Raises:
            ValueError: Parameter sanitisation"""

        # Pre-flight checks

        if len(frames) == 0:
            raise TypeError(PARAMETER_ERROR_MESSAGE['EMPTY_FRAMES'])

        if not (labels in iter(Labels)):
            raise TypeError(PARAMETER_ERROR_MESSAGE['INCORRECT_VALUE'].format(labels, [e for e in Labels]))

        if not (gradient_technique in iter(Gradient)):
            raise TypeError(
                PARAMETER_ERROR_MESSAGE['INCORRECT_VALUE'].format(gradient_technique, [e for e in Gradient]))

        if not (trajectory_technique in iter(Trajectory)) and trajectory_technique is not None:
            raise TypeError(
                PARAMETER_ERROR_MESSAGE['INCORRECT_VALUE'].format(trajectory_technique, [e for e in Trajectory]))

        if not (segmentation_technique in iter(Segmentation)) and segmentation_technique is not None:
            raise TypeError(
                PARAMETER_ERROR_MESSAGE['INCORRECT_VALUE'].format(segmentation_technique, [e for e in Segmentation]))

        if not (pre_trained_model in iter(Models)) and pre_trained_model is not None:
            raise TypeError(PARAMETER_ERROR_MESSAGE['INCORRECT_VALUE'].format(pre_trained_model, [e for e in Models]))

        if model and pre_trained_model:
            raise ValueError(PARAMETER_ERROR_MESSAGE['INCORRECT_CONFIG'].format(model, pre_trained_model))

        conv_layer_outputs = {}
        class_idx_str = 'class_idx_str'

        vanilla_vision = pathvision.Vanilla()

        def _call_model_function(image, call_model_args=None, expected_keys=None):
            images = _preprocess_image(image)

            target_class = call_model_args[class_idx_str]
            preds = model(images)
            LOGGER.info("Classes in the prediction: {}".format([cat['name'] for cat in coco.loadCats(preds[0]["labels"].numpy()[:10])]))
            LOGGER.info("IDs in the prediction: {}".format(
                [cat['id'] for cat in coco.loadCats(preds[0]["labels"].numpy()[:10])]))
            out = preds[0]['scores'].unsqueeze(0)
            # Check if there's the class we're looking is in the current interpolated image
            if target_class not in preds[0]['labels']:
                # It's not, so let's return an empty gradient. Which is essentially skipping this image.
                LOGGER.info("{} is not in {}".format(target_class, preds[0]['labels']))

                empty_gradients = torch.zeros_like(images.permute(0, 2, 3, 1)).detach().numpy()
                return {INPUT_OUTPUT_GRADIENTS: empty_gradients}
            else:
                target_classes = torch.tensor(target_class)
                # Extract the class that we've confirmed is in the labels
                target_class_idx = torch.where(torch.isin(preds[0]['labels'], target_classes))[0][0]
                label = coco.loadCats(target_class)[0]['name']

            if INPUT_OUTPUT_GRADIENTS in expected_keys:
                outs = out[:, target_class_idx]
                LOGGER.info("Detected accuracy {} for class {}".format(outs, label))
                grads = torch.autograd.grad(outs, images, grad_outputs=torch.ones_like(outs))
                grads = torch.movedim(grads[0], 1, 3)
                gradients = grads.detach().numpy()
                return {INPUT_OUTPUT_GRADIENTS: gradients}
            else:
                LOGGER.info("---INPUT_OUTPUT_GRADIENTS is NOT EXPECTED KEYS---")
                one_hot = torch.zeros_like(out)
                one_hot[:, target_class_idx] = 1
                model.zero_grad()
                out.backward(gradient=one_hot, retain_graph=True)
                return conv_layer_outputs

        '''
        Initial loading
        '''
        # Load the model
        if pre_trained_model:
            # If it's not None, then we know it's going to be in the ENUM already as we checked above
            if pre_trained_model == Models.FASTERCNN_RESNET50_FPN:
                model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, progress=True,
                                                                             num_classes=91, weights_backbone=True)
        else:
            print("We need to load their model and their dataset")

        # Load annotations into memory
        coco_annotations = 'pathvision/data/instances_val2017.json'
        coco = COCO(coco_annotations)

        if model:
            model.eval()
            for frame in frames:
                LOGGER.info("Processing frame {} of {} frames".format(frames.index(frame)+1, len(frames)+1))
                im_pil = frame
                im_arr = _load_image_arr(pil_img=im_pil)
                im_for_od = _preprocess_image([im_arr])
                im_tensor = _load_image(im_pil)
                od_preds = model(im_for_od)
                pre, annot_labels = _pre_process_model_output(preds=od_preds, coco=coco)

                results = {
                    "origin": [],
                    "crops": [],
                    "crops_on_origin": [],
                    "coords": [],
                    "size": (),
                    "vanilla": [],
                    "smoothgrad": [],
                }

                '''
                Let's crop and organise every bounding box image
                '''
                # Get the size of the original image as our background canvas
                results['size'] = tuple(reversed(im_tensor[0].shape[-2:]))
                results['origin'].append(im_tensor[0])
                for i in range(len(pre[0]['labels'])):
                    cropped_image, bb_coords = _crop_frame(im_tensor, pre[0]['boxes'][i].tolist())
                    results['crops'].append(_load_image_arr(pil_img=cropped_image))
                    results['coords'].append(bb_coords)
                    original_size_image = Image.new("RGB", results['size'])
                    original_size_image.paste(cropped_image, bb_coords)
                    results['crops_on_origin'].append(original_size_image)

                class_idxs = pre[0]['labels']
                if LoadFromDisk == False:
                    for i, tensor in enumerate(class_idxs):
                        call_model_args = {class_idx_str: tensor.item()}
                        cats = coco.loadCats(coco.getCatIds())
                        cat_id = call_model_args[class_idx_str]
                        cat_name = next(cat['name'] for cat in cats if cat['id'] == cat_id)
                        LOGGER.info("Category name for index value {}: {}".format(cat_id, cat_name))
                        vanilla_mask_3d = vanilla_vision.GetMask(results['crops'][i], _call_model_function,
                                                                 call_model_args)
                        smoothgrad_mask_3d = vanilla_vision.GetSmoothedMask(results['crops'][i], _call_model_function,
                                                                            call_model_args)

                        results['vanilla'].append(pathvision.VisualizeImageGrayscale(vanilla_mask_3d))
                        results['smoothgrad'].append(pathvision.VisualizeImageGrayscale(smoothgrad_mask_3d))
                        LOGGER.info("Completed image {} of {}".format(frames.index(frame)+1, len(frames)+1))
                        LOGGER.info("Saving to disk")
                        # Saving to disk
                        for i in range(len(results['smoothgrad'])):
                            savetxt('pathvision/test/numpys/vanilla/vanilla-{}.csv'.format(i), results['vanilla'][i],
                                    delimiter=',')
                            savetxt('pathvision/test/numpys/smoothgrad/smoothgrad-{}.csv'.format(i),
                                    results['smoothgrad'][i], delimiter=',')
                else:
                    LOGGER.info("Loading gradients from disk")
                    # Define the folder paths
                    vanilla_folder = 'pathvision/test/numpys/vanilla/'
                    smoothgrad_folder = 'pathvision/test/numpys/smoothgrad/'

                    # Loop through the files in the folder and load the numpy arrays
                    for filename in os.listdir(vanilla_folder):
                        if filename.endswith('.csv'):
                            vanilla_arr = np.loadtxt(os.path.join(vanilla_folder, filename), delimiter=',')
                            results['smoothgrad'].append(vanilla_arr)

                    for filename in os.listdir(smoothgrad_folder):
                        if filename.endswith('.csv'):
                            smoothgrad_arr = np.loadtxt(os.path.join(smoothgrad_folder, filename), delimiter=',')
                            results['smoothgrad'].append(smoothgrad_arr)

                if segmentation_technique == "Panoptic Deeplab":
                    cfg = get_cfg()
                    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
                    cfg.merge_from_file(
                        model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml"))
                    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
                    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
                    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
                        "COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml")
                    predictor = DefaultPredictor(cfg)
                    outputs = predictor(im_arr)

                    instances = outputs["instances"].to("cpu")
                    # Get the binary masks for each instance
                    masks = instances.pred_masks.numpy()

                    masked_gradients_list = []
                    overlapping_pixels_list = []
                    masked_regions = []

                    original_image = to_pil(results['origin'][0])

                    # Loop through the masks and save each one as a separate image
                    for i, mask in enumerate(masks):
                        # Convert the binary mask to a uint8 image (0s and 255s)
                        mask = np.uint8(mask * 255)
                        # Apply a bitwise-and operation to the original image to extract the masked region
                        original_base_image = Image.new("L", results['size'], 0)

                        smoothgrad_arr = results['smoothgrad'][i] * 10.0

                        original_base_image.paste(TF.to_pil_image(smoothgrad_arr), results['coords'][i])

                        # Resize the smoothgrad gradients to match the size of the mask
                        # smoothed_gradients = cv2.resize(results['smoothgrad'][i], (results['size']))

                        masked_region = cv2.bitwise_and(im_arr[:, :, ::-1], im_arr[:, :, ::-1], mask=mask)

                        # Apply the binary mask to the resized gradients to keep only the gradients that overlap with the segment
                        masked_gradients = cv2.bitwise_and(_load_image_arr(pil_img=original_base_image),
                                                           _load_image_arr(pil_img=original_base_image), mask=mask)

                        # Create a copy of the masked region to show the overlapping pixels in red
                        overlapping_pixels = masked_region.copy()

                        # Set the values of the overlapping pixels to red
                        overlapping_pixels[..., 0] += masked_gradients.astype(overlapping_pixels.dtype)

                        # Save the masked region and the masked gradients as separate images
                        masked_regions.append(masked_region)
                        masked_gradients_list.append(masked_gradients)
                        overlapping_pixels_list.append(overlapping_pixels)

                    for i in masked_gradients_list:
                        smoothgrad_array = (i - i.min()) / (i.max() - i.min())

                        smoothgrad_array = (smoothgrad_array * 255).astype(np.uint8)

                        smoothgrad_pil = Image.fromarray(smoothgrad_array, mode="L")

                        smoothgrad_with_alpha = Image.new("RGBA", smoothgrad_pil.size, (0, 0, 0, 0))

                        smoothgrad_with_alpha.paste(smoothgrad_pil, (0, 0))

                        smoothgrad_inverted = ImageOps.invert(smoothgrad_pil)

                        threshold = 200

                        alpha = smoothgrad_inverted.point(lambda x: 255 if x < threshold else 0)

                        smoothgrad_with_alpha.putalpha(alpha)

                        original_image.paste(im=_reduce_opacity(smoothgrad_with_alpha, 0.5),
                                             mask=_reduce_opacity(smoothgrad_with_alpha, 0.5))
                    return original_image


        else:
            raise ValueError(PARAMETER_ERROR_MESSAGE['NO_MODEL'])

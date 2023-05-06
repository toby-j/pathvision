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
import time
import cv2
from PIL import Image, ImageEnhance
import numpy as np
import os
import logging
from pycocotools.coco import COCO
from torchvision.transforms import ToPILImage
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
import pathvision.core as pathvision
from pathvision.core.meaurements import calculate_overlap
import torchvision.transforms.functional as TF
from pathvision.core.base import CorePathvision, INPUT_OUTPUT_GRADIENTS
from pathvision.core.types import Gradient, Segmentation, Trajectory, Labels, Models
from pathvision.core.logger import logger as LOGGER
import torchvision
import torch

"""
Finds the prime factors for a given integer RSA modulus n, where the range
between the two prime factors is less than (64n)^1/4.

:param n: The modulus to factorize.
:param range_limit: Optionally, a range limit that can be specified to
show if the assumption will hold.
:return: Either the integers p and q in a Tuple, or None.
"""

to_pil = ToPILImage()

INCORRECT_VALUE = 'INCORRECT_VALUE'
EMPTY_FRAMES = 'EMPTY_FRAMES'
INCORRECT_CONFIG = 'INCORRECT_CONFIG'
NO_MODEL = 'NO_MODEL'
UNEQUAL_GRADIENT_COUNT = 'UNEQUAL_GRADIENT_COUNT'

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
    ),
    UNEQUAL_GRADIENT_COUNT: (
        'Missing gradients'
    )
}


# Utility Functions
# Public
# Return PIL image as np array, or a image provided a path.
def load_image_arr(file_path='', pil_img=None):
    if file_path != '':
        im = Image.open(file_path)
    elif pil_img:
        im = pil_img
    else:
        LOGGER.critical("Unable to convert image to array")
        raise Exception
    im = np.asarray(im)
    return im


# Utility Functions
# Private

def _preprocess_image(im):
    ## If it's a png, we need to squeeze it. But we also check it's at least a .jpg (3 channels)
    im_arr = np.array(im)
    print(im_arr.shape)
    im_arr = im_arr / 255
    im_arr = np.transpose(im_arr, (0, 3, 1, 2))
    im_tensor = torch.tensor(im_arr, dtype=torch.float32)
    im_tensor = im_tensor[:, :3, :, :]
    # images = transformer.forward(images)
    return im_tensor.requires_grad_(True)


"""Here we clean the preds object returned from the model. There's likely many classes that has too low of a 
percentage to bare importance. This is defined by the threshold amount.

Pathvision supports multi-label objects. For example, an image of two for the same dog. We assume that the 
predictions returned by the model would contain two high accuracies for dog. We can then assume there's two dogs in 
the image.

:param preds: The prediction object returned by the model
:param threshold: Will ignore objects with prediction scores below this threshold percentage
:param coco: If we're using COCO labels
:return:
    preds: a cleaned preds object
    annot_labels: the labels of the objects as text
"""


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
                      threshold=None, LoadFromDisk=False, log=False, debug=False):

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
          LoadFromDisk: DEBUG: Loading the processed gradient outs from disk to avoid calculating them from scratch again
          LOGGER: OPTIONAL: Defaults to just show the ERROR and WARNING messages, but can be switched to DEBUG mode.
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

        if torch.cuda.is_available():
            device = torch.device('cuda')  # Assign a CUDA device
        else:
            device = torch.device('cpu')  # Assign a CPU device

        if log:
            LOGGER.setLevel(logging.DEBUG)
        else:
            LOGGER.setLevel(logging.WARNING)

        for handler in LOGGER.handlers:
            handler.setLevel(LOGGER.level)

        conv_layer_outputs = {}
        class_idx_str = 'class_idx_str'

        vanilla_vision = pathvision.Vanilla()

        def _call_model_function(image, call_model_args=None, expected_keys=None):
            images = _preprocess_image(image)

            target_class = call_model_args[class_idx_str]
            preds = model(images)
            LOGGER.info("Classes in the interpolation: {}".format(
                [cat['name'] for cat in coco.loadCats(preds[0]["labels"].numpy()[:10])]))
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

        results = {
            "frame_data": [],
            "errors": []
        }

        kalman_tracker = {}


        if model:
            model.eval()
            for frame in frames:
                LOGGER.info("Processing frame {} of {} frames".format(frames.index(frame) + 1, len(frames) + 1))
                im_pil = frame
                im_arr = load_image_arr(pil_img=im_pil)
                im_for_od = _preprocess_image([im_arr])
                im_tensor = _load_image(im_pil)

                od_preds = model(im_for_od)
                pre, annot_labels = _pre_process_model_output(preds=od_preds, coco=coco)

                LOGGER.info("Model preds: {}".format(od_preds))
                # After we've processed the predictions, we're left with high accuracy predictions. There's a chance the model could predict the same object with high accuracy twice.
                LOGGER.info("Model preds processed: {}".format(pre))

                """
                origin: original image as a PIL
                crops: cropped out objects as their original size as a numpy array
                crops_on_origin: cropped out objects pasted over a black image of the origin.
                coords: Bounding box coordinates as x1, x2, y1, y2
                size: Tuple of the resolution of the original image
                gradients:
                    heatmap_3d: RGB heatmap image from vanilla gradients
                result_images:
                    full: All techniques applied
                    overlap: Only gradients that are not within the segment but are within the bounding box
                    internal: Only gradients within the segment
                metrics:
                    overlap percentage: overlap percentage as a decimal
                """

                frame_data = {
                    "origin": im_pil,
                    "crops": [],
                    "crops_on_origin": [],
                    "coords": [],
                    "size": (),
                    "vanilla": {
                        "gradients": {
                            "heatmap_3d": [],
                            "raw": []
                        },
                        "result_images": {
                            "full": [],
                            "overlap": [],
                            "internal": [],
                        },
                        "metrics": {
                            "overlap_ps": 0.
                        }
                    },
                    "smoothgrad": {
                        "gradients": {
                            "heatmap_3d": [],
                        },
                        "result_images": {
                            "full": [],
                            "overlap": [],
                            "internal": [],
                        },
                        "metrics": {
                            "overlap_ps": 0.
                        }
                    },
                }

                folder_dict = {"VanillaGradients": "pathvision/test/outs/vanilla/",
                               "Smoothgrad": "pathvision/test/outs/smoothgrad/"}

                technique_dict = {"VanillaGradients": "vanilla",
                                  "Smoothgrad": "smoothgrad"}

                '''
                Let's crop and organise every bounding box image
                '''
                # Get the size of the original image as our background canvas
                frame_data['size'] = tuple(reversed(im_tensor[0].shape[-2:]))
                frame_data['origin'] = im_pil
                # For the number of label's we
                for i in range(len(pre[0]['labels'])):
                    cropped_image, bb_coords = _crop_frame(im_tensor, pre[0]['boxes'][i].tolist())
                    # Store the cropped object as a numpy array
                    frame_data['crops'].append(np.array(cropped_image))
                    frame_data['coords'].append(bb_coords)
                    crop_object = Image.new("RGB", frame_data['size'])
                    # Paste the crops over the origin image. Now we've only got the objects we're interested in in the image.
                    crop_object.paste(cropped_image, bb_coords)
                    frame_data['crops_on_origin'].append(crop_object)

                technique_key = technique_dict.get(gradient_technique)

                '''
                Kalman Tracking
                '''
                #
                # class_idxs = pre[0]['labels']
                # # For each label
                # for i, class_idx in enumerate(len(class_idxs)):
                #     bb_box = pre[0]['boxes'][i]
                #
                #     def euclidean_distance(x1, y1, x2, y2):
                #         return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                #
                #     def find_closest_bbox(bboxes, target_bbox):
                #         target_x, target_y = (target_bbox[0] + target_bbox[2]) / 2, (
                #                     target_bbox[1] + target_bbox[3]) / 2
                #         closest_bbox = None
                #         idx = int()
                #         closest_distance = float('inf')
                #         for x, bbox in enumerate(bboxes):
                #             bbox_x, bbox_y = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
                #             distance = euclidean_distance(target_x, target_y, bbox_x, bbox_y)
                #             if distance < closest_distance:
                #                 closest_bbox = bbox
                #                 closest_distance = distance
                #                 idx = x
                #         return closest_bbox, idx
                #
                #     # We need to check if there's multiple of the same label in pre[0]['lebels']. if there is, we know we're looking at two objects.
                #     # Their prediction scores might change, so we can't use index. Best approach is to use distance between bounding boxes to decide which one it belongs to.
                #
                #     if class_idx in kalman_tracker:
                #         # Finding duplicates in the model's found classes
                #         counter = Counter(class_idxs)
                #         duplicates = [idx for idx, count in counter.items() if count > 1]
                #         # Have we seen multiple of the same class?
                #         if duplicates:
                #             for x, dup_idx in enumerate(duplicates):
                #                 # Collect known end bounding boxes for this object type
                #                 object_bbs = []
                #                 for object_list in kalman_tracker[dup_idx].values():
                #                     if len(object_list) > 0:
                #                         # Get the latest bounding box
                #                         last_bb = object_list[-1]
                #                         object_bbs.append(last_bb)
                #                     else:
                #                         LOGGER.debug("There wasn't an bounding box even though there was a key..")
                #                 closest_bbox, idx = find_closest_bbox(object_bbs, bb_box)
                #                 # Update the object with new box
                #                 kalman_tracker[dup_idx][idx] = closest_bbox
                #         else:
                #             closest_bbox, idx = find_closest_bbox(kalman_tracker[class_idx].values()[-1], bb_box)
                #             kalman_tracker[class_idx][idx] = closest_bbox
                #
                #     else:
                #         kalman_tracker[class_idx] = [bb_box]
                #
                #
                #     # Get the bounding box coordinates and calculate its center
                #     bb_coords = pre[0]['boxes'][i].tolist()
                #     bb_center = [(bb_coords[0] + bb_coords[2]) / 2, (bb_coords[1] + bb_coords[3]) / 2]
                #
                #     if class_idx in kalman_tracker.keys():
                #         # Is there already more than one object in the dict
                #         if isinstance(kalman_tracker.get(class_idx), dict) and len(kalman_tracker[class_idx]) > 1:
                #             # There's multiple of the same class, so we need to find which one the bounding box belongs to
                #             # We'll use Euclidean distance to find which object this box is closest to and we'll append to that one.
                #             # This will get more accurate the higher the framerate
                #         else:
                #             # There's an object in the dict but there's only one
                #             kalman_tracker[class_idx].append(bb_coords)
                #     else:
                #         kalman_tracker[class_idx] = bb_coords

                '''
                Gradient calculation
                '''

                if LoadFromDisk == False:
                    for i, tensor in enumerate(class_idxs):
                        call_model_args = {class_idx_str: tensor.item()}
                        cats = coco.loadCats(coco.getCatIds())
                        cat_id = call_model_args[class_idx_str]
                        cat_name = next(cat['name'] for cat in cats if cat['id'] == cat_id)
                        LOGGER.debug("Category name for index value {}: {}".format(cat_id, cat_name))

                        if technique_key == "vanilla":
                            vanilla_mask_3d = vanilla_vision.GetMask(frame_data['crops'][i], _call_model_function,
                                                                     call_model_args)
                            heatmap_img, raw_gradients = pathvision.visualiseImageToHeatmap(image_3d=vanilla_mask_3d)
                            frame_data['vanilla']['gradients']['heatmap_3d'].append(heatmap_img)
                            frame_data['vanilla']['gradients']['raw'].append(raw_gradients)

                        elif technique_key == "smoothgrad":
                            smoothgrad_mask_3d = vanilla_vision.GetSmoothedMask(frame_data['crops'][i],
                                                                                _call_model_function,
                                                                                call_model_args)
                            frame_data['smoothgrad']['gradients']['heatmap_3d'].append(
                                pathvision.visualiseImageToHeatmap(image_3d=smoothgrad_mask_3d))

                        LOGGER.debug("Completed image {} of {}".format(frames.index(frame) + 1, len(frames) + 1))
                        LOGGER.debug("Saving to disk")

                        '''
                        Checking that we have the same number of gradients in each category.
                        '''
                        vanilla_lengths = [len(frame_data['vanilla']['gradients'][key]) for key in
                                           ['heatmap_3d']]
                        smoothgrad_lengths = [len(frame_data['smoothgrad']['gradients'][key]) for key in
                                              ['heatmap_3d']]

                        if not all(len_list == vanilla_lengths[0] for len_list in vanilla_lengths) and \
                                all(len_list == smoothgrad_lengths[0] for len_list in smoothgrad_lengths):
                            raise RuntimeError(PARAMETER_ERROR_MESSAGE['UNEQUAL_GRADIENT_COUNT'])
                    '''
                    DEBUG ONLY
                    - Once we've calculated the gradients and added them to the dict, we can save them to disk for convenience
                    '''
                    if debug:
                        if technique_key == "vanilla":
                            for i in range(len(frame_data['crops'])):
                                np.save('pathvision/test/outs/vanilla/heatmap/heatmap_image{}.npy'.format(i),
                                        frame_data['vanilla']['gradients']['heatmap_3d'][i])
                                np.save('pathvision/test/outs/vanilla/raw/raw_grad{}.npy'.format(i),
                                        vanilla_mask_3d)

                        elif technique_key == "smoothgrad":
                            for i in range(len(frame_data['crops'])):
                                np.save('pathvision/test/outs/smoothgrad/heatmap/heatmap_image{}.npy'.format(i),
                                        frame_data['smoothgrad']['gradients']['heatmap_3d'][i])
                                np.save('pathvision/test/outs/smoothgrad/raw/raw_grad{}.npy'.format(i),
                                        smoothgrad_mask_3d)
                else:
                    LOGGER.debug("Loading gradients from disk")
                    folder = folder_dict.get(gradient_technique)
                    # Loop through the files in the folder and load the numpy arrays
                    for i, filename in enumerate([f for f in os.listdir(folder + "heatmap/") if f.endswith('.npy')]):
                        np_arr = np.load(os.path.join(folder, 'heatmap/heatmap_image{}.npy'.format(i)))
                        frame_data[technique_key]['gradients']['heatmap_3d'].append(np_arr)

                if segmentation_technique == "Panoptic Deeplab":
                    cfg = get_cfg()
                    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
                    cfg.merge_from_file(
                        model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml"))
                    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
                    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
                    cfg.MODEL.DEVICE = 'cuda' if device.type == 'cuda' else 'cpu'
                    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
                        "COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml")
                    predictor = DefaultPredictor(cfg)

                    masks = []

                    for i, crop_object in enumerate(frame_data['crops_on_origin']):
                        outputs = predictor(np.array(crop_object))
                        instances = outputs["instances"].to("cpu")
                        # We can be pretty certain the bounding box only has one object in it for the segmenter, so we use [0] to pick the first set of masks.
                        # If there's multiple objects, pred_masks would contain multiple arrays of mask arrays.
                        masks.append(instances.pred_masks[0].numpy().squeeze())

                    masked_gradients_list = []
                    masked_regions = []

                    # Loop through the masks and save each one as a separate image
                    for i, raw_mask in enumerate(masks):
                        # Convert the binary mask to a uint8 image (0s and 255s) for visualisation
                        mask = np.uint8(raw_mask * 255)
                        # Apply a bitwise-and operation to the original image to extract the masked region
                        original_base_image = Image.new("RGBA", frame_data['size'], 0)

                        percentage_overlap = calculate_overlap(predictor,
                                                               frame_data[technique_key]['gradients']['raw'][i],
                                                               frame_data['crops'][i])

                        frame_data[technique_key]['metrics']['overlap_ps'] = percentage_overlap

                        original_base_image.paste(frame_data[technique_key]['gradients']['heatmap_3d'][i],
                                                  frame_data['coords'][i])

                        # Extract the masked region from the main image.
                        masked_region = cv2.bitwise_and(im_arr[:, :, ::-1], im_arr[:, :, ::-1], mask=mask)

                        im_bgr = cv2.cvtColor(np.asarray(original_base_image), cv2.COLOR_BGR2BGRA)

                        # Apply the binary mask to the resized gradients to keep only the gradients that are within
                        # the segment
                        masked_gradients = cv2.bitwise_and(im_bgr, im_bgr, mask=mask)

                        if percentage_overlap > 50:
                            # We log an error. The array reads as [the frame, [the index of the error type]). We can then inspect the data on the front-end.
                            LOGGER.debug("Overlapping pixels is over 50%, writing to error JSON")
                            results['errors'].append([frames.index(frame), [1]])

                        if debug:
                            LOGGER.debug("Percentage of overlap: {}".format(percentage_overlap))
                            # LOGGER.debug("Total gradient sum {}".format(
                            #     np.sum(load_image_arr(pil_img=original_base_image), axis=2)))
                            LOGGER.debug("Writing debug images")
                            cv2.imwrite("debug/im_bgr/im_bgr_{}.png".format(time.time()), im_bgr)
                            cv2.imwrite("debug/masked/masked_{}.png".format(time.time()), masked_gradients)

                        # Save the masked region and the masked gradients as separate images

                        masked_regions.append(masked_region)
                        masked_gradients_list.append(masked_gradients)

                        # Now that we've made the crops, we can reduce the noise for the output image.

                    # Create a new transparent image with the size of the background image
                    output_image = Image.new('RGBA', im_pil.size, (0, 0, 0, 0))

                    # Loop through each gradient mask
                    for grad_mask in masked_gradients_list:
                        # Overlay the gradient mask onto the transparent image
                        grad_mask = to_pil(grad_mask).convert('RGBA')
                        output_image.alpha_composite(grad_mask)

                    # Paste the transparent image onto the background image
                    im_pil = im_pil.convert('RGBA')
                    im_pil.putalpha(255)
                    result_image = im_pil.copy()
                    result_image.paste(_reduce_opacity(output_image, 0.7), (0, 0), _reduce_opacity(output_image, 0.7))

                    if debug:
                        output_image.save("debug/output_image/output_image {}.png".format(time.time()))
                        result_image.save("debug/final_output/final_output {}.png".format(time.time()))
        else:
            raise ValueError(PARAMETER_ERROR_MESSAGE['NO_MODEL'])

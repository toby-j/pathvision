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
import PIL
from PIL import Image
import numpy as np
from pycocotools.coco import COCO
from torchvision.transforms import ToPILImage

from pathvision.core import VisualizeImageGrayscale

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
                      threshold=None):

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

        def _call_model_function(image, call_model_args=None, expected_keys=None):
            images = _preprocess_image(image)

            target_classes = call_model_args['class_idx_str']
            preds = model(images)
            print("Classes in the prediction {} ".format(coco.loadCats(preds[0]["labels"].numpy()[:10])))
            output = preds[0]['scores'].unsqueeze(0)
            # Check if there's the class we're looking is in the current interpolated image
            if target_classes not in preds[0]['labels']:
                # It's not, so let's return an empty gradient. Which is essentially skipping this image.
                print("{} is not in {}".format(target_classes, preds[0]['labels']))
                empty_gradients = torch.zeros_like(images.permute(0, 2, 3, 1)).detach().numpy()
                return {INPUT_OUTPUT_GRADIENTS: empty_gradients}
            else:
                target_classes = torch.tensor(target_classes)
                # Extract the class that we've confirmed is in the labels
                target_class_idx = torch.where(torch.isin(preds[0]['labels'], target_classes))[0]
                target_class_idx = target_class_idx[0]
                print("ID: {}".format(target_class_idx))
            if INPUT_OUTPUT_GRADIENTS in expected_keys:
                print("---INPUT_OUTPUT_GRADIENTS is IN EXPECTED KEYS---")
                print("Outputs {}".format(output))
                outputs = output[:, target_class_idx]
                print("Outputs target_class_idx{}".format(outputs))
                grads = torch.autograd.grad(outputs, images, grad_outputs=torch.ones_like(outputs))
                grads = torch.movedim(grads[0], 1, 3)
                gradients = grads.detach().numpy()
                return {INPUT_OUTPUT_GRADIENTS: gradients}
            else:
                print("---INPUT_OUTPUT_GRADIENTS is NOT EXPECTED KEYS---")
                one_hot = torch.zeros_like(output)
                one_hot[:, target_class_idx] = 1
                model.zero_grad()
                output.backward(gradient=one_hot, retain_graph=True)
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
        coco_annotations = '../data/instances_val2017.json'
        coco = COCO(coco_annotations)

        if model:
            model.eval()
            for frame in frames:
                im_pil = to_pil(frame)
                im_for_od = _preprocess_image(im_pil)
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
                for i, tensor in enumerate(class_idxs):
                    call_model_args = {'class_idx_str': tensor.item()}
                    cats = coco.loadCats(coco.getCatIds())
                    cat_id = call_model_args['class_idx_str']
                    cat_name = next(cat['name'] for cat in cats if cat['id'] == cat_id)
                    print("Category name for index value {}: {}".format(cat_id, cat_name))

                    vanilla_mask_3d = CorePathvision.GetMask(results['crops'][i], _call_model_function,
                                                                call_model_args)
                    smoothgrad_mask_3d = CorePathvision.GetSmoothedMask(results['crops'][i], _call_model_function,
                                                                           call_model_args)

                    results['vanilla'].append(VisualizeImageGrayscale(vanilla_mask_3d))
                    results['smoothgrad'].append(VisualizeImageGrayscale(smoothgrad_mask_3d))


        else:
            raise ValueError(PARAMETER_ERROR_MESSAGE['NO_MODEL'])

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

import numpy as np
from matplotlib import pyplot as plt, cm
from PIL import Image

'''
Convert 3D tensor to 2D vector and normalise.
Used for calculating percentage overlap and visualisation techniques
'''


def normaliseGradients(image_3d, percentile=99):
    image_2d = np.sum(np.abs(image_3d), axis=2)

    # Get max pixel value in the image
    vmax = np.percentile(image_2d, percentile)
    # Get minimum pixel value in the image
    vmin = np.min(image_2d)

    # Normalise the values. We clip intensities so values lower than 0 are equal 0.
    return np.clip((image_2d - vmin) / (vmax - vmin), 0, 1)


def visualiseImageToHeatmap(image_3d, percentile=99):
    r"""Returns a 3D tensor as RGB 3D heatmap
    Pixels with higher weightage in sailiency heatmap will most saturated and will correspond to high RGB values in output heatmap_rgb
  """
    image_2d = normaliseGradients(image_3d)
    # Create heatmap using "jet" colormap, which returns an RGBA image
    heatmap = plt.get_cmap('jet')(image_2d) * 255

    # Normalise to 0,255 so it's visible when pasted
    return Image.fromarray(heatmap.astype(np.uint8), mode='RGBA'), image_2d


def VisualizeImageDiverging(image_3d, percentile=99):
    r"""Returns a 3D tensor as a 2D tensor with positive and negative values.
  """
    image_2d = np.sum(image_3d, axis=2)

    span = abs(np.percentile(image_2d, percentile))
    vmin = -span
    vmax = span

    return np.clip((image_2d - vmin) / (vmax - vmin), -1, 1)

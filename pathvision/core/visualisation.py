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
from matplotlib import pyplot as plt
from PIL import Image

def VisualizeImage(image_3d, percentile=99, heatmap=False):
    r"""Returns a 3D tensor as a grayscale 2D tensor.

  This method sums a 3D tensor across the absolute value of axis=2, and then
  clips values at a given percentile.
  """
    image_2d = np.sum(np.abs(image_3d), axis=2)

    # Get max pixel value in the image
    vmax = np.percentile(image_2d, percentile)
    # Get minimum pixel value in the image
    vmin = np.min(image_2d)
    # Normalise the values. We clip intensities so values lower than 0 are equal 0.
    image_2d = np.clip((image_2d - vmin) / (vmax - vmin), 0, 1)
    if heatmap:

        # Create heatmap using "jet" colormap, which returns an RGBA image
        heatmap = plt.get_cmap('jet')(image_2d)

        # Convert heatmap to RGB image to support PIL .paste() as it doesn't support RGBA
        heatmap_rgb = np.delete(heatmap, 3, 2)

        return heatmap_rgb.astype(np.uint8)
    else:
        return image_2d

def VisualizeImageDiverging(image_3d, percentile=99):
    r"""Returns a 3D tensor as a 2D tensor with positive and negative values.
  """
    image_2d = np.sum(image_3d, axis=2)

    span = abs(np.percentile(image_2d, percentile))
    vmin = -span
    vmax = span

    return np.clip((image_2d - vmin) / (vmax - vmin), -1, 1)
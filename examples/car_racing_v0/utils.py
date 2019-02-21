# -*- coding: utf-8 -*-
"""Common util functions for CarRacing-v0.

- Author: Curt Park
- Contact: curt.park@medipixel.io
"""

import numpy as np
from skimage import color


def process_image(obs: np.ndarray):
    """Preprocessing function."""
    return (2 * color.rgb2gray(obs) - 1.0).reshape(-1, 96, 96)

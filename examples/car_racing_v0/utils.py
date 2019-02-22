# -*- coding: utf-8 -*-
"""Common util functions for CarRacing-v0.

- Author: Curt Park
- Contact: curt.park@medipixel.io
"""

import cv2
import numpy as np


def process_image(obs: np.ndarray):
    """Preprocessing function."""
    gray_obs = 2 * cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY) - 1.0
    gray_obs_resized = cv2.resize(gray_obs, dsize=(64, 64))
    return gray_obs_resized.reshape(-1, 64, 64)

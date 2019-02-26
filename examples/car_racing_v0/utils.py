# -*- coding: utf-8 -*-
"""Common util functions for CarRacing-v0.

- Author: Curt Park
- Contact: curt.park@medipixel.io
"""

import cv2
import numpy as np


def process_image(obs: np.ndarray):
    """Preprocessing function."""
    gray_obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
    mean, std = cv2.meanStdDev(gray_obs)
    std_obs = (gray_obs - mean) / (std + 1e-7)
    std_obs = np.expand_dims(std_obs, axis=0)
    return std_obs

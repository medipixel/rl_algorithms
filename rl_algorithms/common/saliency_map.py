# -*- coding: utf-8 -*-
"""Functions for Saliency map.

- Author: Euijin Jeong
- Contact: euijin.jeong@medipixel.io
"""

from datetime import datetime
import os
import pickle

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch

plt.rcParams["figure.figsize"] = (10.0, 8.0)  # set default size of plots
plt.rcParams["image.interpolation"] = "nearest"
plt.rcParams["image.cmap"] = "gray"


def make_saliency_dir():
    """Make directories for saving saliency map result."""

    date_time = datetime.now().strftime("%Y%m%d%H%M%S")
    os.makedirs(f"./data/saliency_map/{date_time}")
    os.makedirs(f"./data/saliency_map/{date_time}/input_image")
    os.makedirs(f"./data/saliency_map/{date_time}/state")
    os.makedirs(f"./data/saliency_map/{date_time}/saliency")
    os.makedirs(f"./data/saliency_map/{date_time}/overlay")
    saliency_map_dir = f"./data/saliency_map/{date_time}/"
    return saliency_map_dir


def compute_saliency_maps(X, y, model, device):
    """Compute a class saliency map using the model for images X and labels y."""

    model.eval()

    # Make input tensor require gradient
    X.requires_grad_()

    saliency = None

    # forward pass
    scores = model(X)
    scores = (scores.gather(1, y.unsqueeze(0))).squeeze(0)

    # backward pass
    scores.backward(torch.FloatTensor([1.0]).to(device))

    # saliency
    saliency, _ = torch.max(X.grad.data.abs(), dim=1)

    return saliency


def save_saliency_maps(i, X, y, model, device, saliency_map_dir):
    """Make and save saliency maps in directory."""

    # Convert X and y from numpy arrays to Torch Tensors
    X_tensor = torch.Tensor(X).float().to(device).unsqueeze(0)
    y = int(y)
    y_tensor = torch.LongTensor([y]).to(device)

    # Compute saliency maps for images in X
    saliency = compute_saliency_maps(X_tensor, y_tensor, model, device)

    # image
    saliency = saliency.cpu().numpy()
    input_image = np.rot90(X[-1], 3)
    input_image = Image.fromarray(np.uint8(input_image * 255.0))
    input_image.save(saliency_map_dir + "/input_image/{}.png".format(i))

    # numpy array
    with open(saliency_map_dir + "/state/{}.pkl".format(i), "wb") as f:
        pickle.dump(X, f)

    cmap = plt.cm.hot
    norm = plt.Normalize(saliency.min(), saliency.max())
    saliency = cmap(norm(saliency[0]))
    saliency = np.rot90(saliency, 3)
    saliency = Image.fromarray(np.uint8(saliency * 255.0))
    saliency.save(saliency_map_dir + "/saliency/{}.png".format(i))

    overlay = Image.blend(input_image.convert("RGBA"), saliency, alpha=0.5)
    overlay.save(saliency_map_dir + "/overlay/{}.png".format(i))

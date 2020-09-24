# -*- coding: utf-8 -*-
"""Abstract Agent used for all agents.

- Author: Curt Park
- Contact: curt.park@medipixel.io
"""

import pickle

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch

SQUEEZENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
SQUEEZENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

plt.rcParams["figure.figsize"] = (10.0, 8.0)  # set default size of plots
plt.rcParams["image.interpolation"] = "nearest"
plt.rcParams["image.cmap"] = "gray"


def compute_saliency_maps(X, y, model, device):
    """
    Compute a class saliency map using the model for images X and labels y.

    Input:
    - X: Input images; Tensor of shape (N, 3, H, W)
    - y: Labels for X; LongTensor of shape (N,)
    - model: A pretrained CNN that will be used to compute the saliency map.

    Returns:
    - saliency: A Tensor of shape (N, H, W) giving the saliency maps for the input
    images.
    """
    # Make sure the model is in "test" mode
    model.eval()

    # Make input tensor require gradient
    X.requires_grad_()

    saliency = None

    # forward pass
    scores = model(X)
    scores = (scores.gather(1, y.unsqueeze(0))).squeeze(0)

    # backward pass
    scores.backward(torch.FloatTensor([1.0] * 1).to(device))

    # saliency
    saliency, _ = torch.max(X.grad.data.abs(), dim=1)

    return saliency


def save_saliency_maps(i, X, y, model, device, saliency_map_dir):

    # Convert X and y from numpy arrays to Torch Tensors
    X_tensor = torch.Tensor(X).float().to(device).unsqueeze(0)
    y = int(y)
    y_tensor = torch.LongTensor([y]).to(device)

    # Compute saliency maps for images in X
    saliency = compute_saliency_maps(X_tensor, y_tensor, model, device)

    # and saliency maps together.

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

    print(i)

# -*- coding: utf-8 -*-
"""Abstract Agent used for all agents.

- Author: Curt Park
- Contact: curt.park@medipixel.io
"""

from abc import ABC, abstractmethod
import argparse
from datetime import datetime
import os
import pickle
import shutil
from typing import Tuple, Union

from PIL import Image
import cv2
import gym
from gym.spaces import Discrete
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
import torch
import torchvision.transforms as T
import wandb

from rl_algorithms.common.grad_cam import GradCAM
from rl_algorithms.utils.config import ConfigDict


SQUEEZENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
SQUEEZENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

plt.rcParams["figure.figsize"] = (10.0, 8.0)  # set default size of plots
plt.rcParams["image.interpolation"] = "nearest"
plt.rcParams["image.cmap"] = "gray"


def preprocess(img, size=224):
    transform = T.Compose(
        [
            T.Resize(size),
            T.ToTensor(),
            T.Normalize(mean=SQUEEZENET_MEAN.tolist(), std=SQUEEZENET_STD.tolist()),
            T.Lambda(lambda x: x[None]),
        ]
    )
    return transform(img)


def deprocess(img, should_rescale=True):
    transform = T.Compose(
        [
            T.Lambda(lambda x: x[0]),
            T.Normalize(mean=[0, 0, 0], std=(1.0 / SQUEEZENET_STD).tolist()),
            T.Normalize(mean=(-SQUEEZENET_MEAN).tolist(), std=[1, 1, 1]),
            T.Lambda(rescale) if should_rescale else T.Lambda(lambda x: x),
            T.ToPILImage(),
        ]
    )
    return transform(img)


def rescale(x):
    low, high = x.min(), x.max()
    x_rescaled = (x - low) / (high - low)
    return x_rescaled


def blur_image(X, sigma=1):
    X_np = X.cpu().clone().numpy()
    X_np = gaussian_filter1d(X_np, sigma, axis=2)
    X_np = gaussian_filter1d(X_np, sigma, axis=3)
    X.copy_(torch.Tensor(X_np).type_as(X))
    return X


def load_imagenet_val(num=None):
    """Load a handful of validation images from ImageNet.

    Inputs:
    - num: Number of images to load (max of 25)

    Returns:
    - X: numpy array with shape [num, 224, 224, 3]
    - y: numpy array of integer image labels, shape [num]
    - class_names: dict mapping integer label to class name
    """
    imagenet_fn = "./datasets/imagenet_val_25.npz"
    if not os.path.isfile(imagenet_fn):
        print("file %s not found" % imagenet_fn)
        print("Run the following:")
        print("cd cs231n/datasets")
        print("bash get_imagenet_val.sh")
        assert False, "Need to download imagenet_val_25.npz"
    print("is working")
    # modify the default parameters of np.load
    np_load_old = np.load

    # pylint: disable=unnecessary-lambda
    np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
    f = np.load(imagenet_fn)
    X = f["X"]
    y = f["y"]
    class_names = f["label_map"].item()
    if num is not None:
        X = X[:num]
        y = y[:num]
    return X, y, class_names


# Example of using gather to select one entry from each row in PyTorch
def gather_example():
    N, C = 4, 5
    s = torch.randn(N, C)
    y = torch.LongTensor([1, 2, 1, 3])
    print(s)
    print(y)
    print(s.gather(1, y.view(-1, 1)).squeeze())


def compute_saliency_maps(X, y, model):
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
    ##############################################################################
    # TODO: Implement this function. Perform a forward and backward pass through #
    # the model to compute the gradient of the correct class score with respect  #
    # to each input image. You first want to compute the loss over the correct   #
    # scores (we'll combine losses across a batch by summing), and then compute  #
    # the gradients with a backward pass.                                        #
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # forward pass
    scores = model(X)
    # print(scores.shape) # torch.Size([5, 1000]) since 5 images, 1000 classes
    scores = (scores.gather(1, y.unsqueeze(0))).squeeze(0)
    # print(scores.shape) # torch.Size([5])

    # backward pass
    scores.backward(torch.FloatTensor([1.0] * 1).to(device))

    # saliency
    saliency, _ = torch.max(X.grad.data.abs(), dim=1)

    return saliency


def show_saliency_maps(i, X, y, model, pixel_grad_dir):

    # Convert X and y from numpy arrays to Torch Tensors
    X_tensor = torch.Tensor(X).float().to(device).unsqueeze(0)
    y = int(y)
    y_tensor = torch.LongTensor([y]).to(device)

    # Compute saliency maps for images in X
    saliency = compute_saliency_maps(X_tensor, y_tensor, model)

    # and saliency maps together.

    # image
    saliency = saliency.cpu().numpy()
    input_image = np.rot90(X[-1], 3)
    input_image = Image.fromarray(np.uint8(input_image * 255.0))
    input_image.save(pixel_grad_dir + "/input_image/{}.png".format(i))

    # numpy array
    with open(pixel_grad_dir + "/state/{}.pkl".format(i), "wb") as f:
        pickle.dump(X, f)

    cmap = plt.cm.hot
    norm = plt.Normalize(saliency.min(), saliency.max())
    saliency = cmap(norm(saliency[0]))
    saliency = np.rot90(saliency, 3)
    saliency = Image.fromarray(np.uint8(saliency * 255.0))
    saliency.save(pixel_grad_dir + "/saliency/{}.png".format(i))

    overlay = Image.blend(input_image.convert("RGBA"), saliency, alpha=0.5)
    overlay.save(pixel_grad_dir + "/overlay/{}.png".format(i))

    print(i)


class Agent(ABC):
    """Abstract Agent used for all agents.

    Attributes:
        env (gym.Env): openAI Gym environment
        args (argparse.Namespace): arguments including hyperparameters and training settings
        log_cfg (ConfigDict): configuration for saving log
        state_dim (int): dimension of states
        action_dim (int): dimension of actions
        is_discrete (bool): shows whether the action is discrete

    """

    def __init__(
        self,
        env: gym.Env,
        env_info: ConfigDict,
        args: argparse.Namespace,
        log_cfg: ConfigDict,
    ):
        """Initialize."""
        self.args = args
        self.env = env
        self.env_info = env_info
        self.log_cfg = log_cfg
        self.log_cfg.env_name = env.spec.id if env.spec is not None else env.name

        self.total_step = 0
        self.learner = None

        if isinstance(env.action_space, Discrete):
            self.is_discrete = True
        else:
            self.is_discrete = False

    @abstractmethod
    def select_action(self, state: np.ndarray) -> Union[torch.Tensor, np.ndarray]:
        pass

    @abstractmethod
    def step(
        self, action: Union[torch.Tensor, np.ndarray]
    ) -> Tuple[np.ndarray, np.float64, bool, dict]:
        pass

    @abstractmethod
    def write_log(self, log_value: tuple):  # type: ignore
        pass

    @abstractmethod
    def train(self):
        pass

    def set_wandb(self):
        """Set configuration for wandb logging."""
        wandb.init(
            project=self.log_cfg.env_name,
            name=f"{self.log_cfg.agent}/{self.log_cfg.curr_time}",
        )
        wandb.config.update(vars(self.args))
        wandb.config.update(self.hyper_params)
        shutil.copy(self.args.cfg_path, os.path.join(wandb.run.dir, "config.py"))

    def interim_test(self):
        """Test in the middle of training."""
        self.args.test = True

        print()
        print("===========")
        print("Start Test!")
        print("===========")

        self._test(interim_test=True)

        print("===========")
        print("Test done!")
        print("===========")
        print()

        self.args.test = False

    def test(self):
        """Test the agent."""
        # logger
        if self.args.log:
            self.set_wandb()

        self._test()

        # termination
        self.env.close()

    def _test(self, interim_test: bool = False):
        """Common test routine."""

        if interim_test:
            test_num = self.args.interim_test_num
        else:
            test_num = self.args.episode_num

        if self.args.save_pixel_gradient:
            date_time = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
            os.mkdir("./rl_algorithms/pixel_grad/{}".format(date_time))
            os.mkdir("./rl_algorithms/pixel_grad/{}/input_image".format(date_time))
            os.mkdir("./rl_algorithms/pixel_grad/{}/state".format(date_time))
            os.mkdir("./rl_algorithms/pixel_grad/{}/saliency".format(date_time))
            os.mkdir("./rl_algorithms/pixel_grad/{}/overlay".format(date_time))
            pixel_grad_dir = "./rl_algorithms/pixel_grad/{}/".format(date_time)
            i = 0
        score_list = []
        for i_episode in range(test_num):
            state = self.env.reset()
            done = False
            score = 0
            step = 0

            while not done:
                if self.args.render:
                    self.env.render()

                action = self.select_action(state)
                if self.args.save_pixel_gradient:
                    for param in self.learner.dqn.parameters():
                        param.requires_grad = False
                    show_saliency_maps(
                        i, state, action, self.learner.dqn, pixel_grad_dir
                    )
                    i += 1
                next_state, reward, done, _ = self.step(action)

                state = next_state
                score += reward
                step += 1

            print(
                "[INFO] test %d\tstep: %d\ttotal score: %d" % (i_episode, step, score)
            )
            score_list.append(score)

        if self.args.log:
            wandb.log(
                {
                    "avg test score": round(sum(score_list) / len(score_list), 2),
                    "test total step": self.total_step,
                }
            )

    def test_with_gradcam(self):
        """Test agent with Grad-CAM."""
        policy = self.learner.get_policy()
        gcam = GradCAM(model=policy.eval())

        for i_episode in range(self.args.episode_num):
            state = self.env.reset()
            done = False
            score = 0
            step = 0

            key = 0
            print("\nPress Any Key to move to next step... (quit: ESC key)")
            while not done:
                action = self.select_action(state)
                next_state, reward, done, _ = self.step(action)

                state = self._preprocess_state(state)
                _ = gcam.forward(state)
                ids = torch.LongTensor([[int(action)]]).cuda()
                gcam.backward(ids=ids)

                state = state[-1].detach().cpu().numpy().astype(np.uint8)
                state = np.transpose(state)
                state = cv2.cvtColor(state, cv2.COLOR_GRAY2BGR)
                state = cv2.resize(state, (150, 150), interpolation=cv2.INTER_LINEAR)

                # Get Grad-CAM image
                result_images = None
                for target_layer in self.hyper_params.grad_cam_layer_list:
                    regions = gcam.generate(target_layer)
                    regions = regions.detach().cpu().numpy()
                    regions = np.squeeze(regions) * 255
                    regions = np.transpose(regions)
                    regions = cv2.applyColorMap(
                        regions.astype(np.uint8), cv2.COLORMAP_JET
                    )
                    regions = cv2.resize(
                        regions, (150, 150), interpolation=cv2.INTER_LINEAR
                    )
                    overlay = cv2.addWeighted(state, 1.0, regions, 0.5, 0)
                    result = np.hstack([state, regions, overlay])
                    result_images = (
                        result
                        if result_images is None
                        else np.vstack([result_images, result])
                    )
                # Show action on result image
                cv2.putText(
                    img=result_images,
                    text=f"action: {action}",
                    org=(50, 50),
                    fontFace=cv2.FONT_HERSHEY_PLAIN,
                    fontScale=1,
                    color=(0, 0, 255),
                    thickness=2,
                )

                cv2.imshow("result", result_images)
                key = cv2.waitKey(0)
                if key == 27 & 0xFF:  # ESC key
                    cv2.destroyAllWindows()
                    break

                state = next_state
                score += reward
                step += 1

            print(
                "[INFO] test %d\tstep: %d\ttotal score: %d" % (i_episode, step, score)
            )
            if key == 27 & 0xFF:  # ESC key
                break

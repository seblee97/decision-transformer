"""
The MIT License (MIT) Copyright (c) 2020 Andrej Karpathy

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

import matplotlib.pyplot as plt

from PIL import Image
import collections

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out

@torch.no_grad()
def sample(model, x, steps, temperature=1.0, sample=False, top_k=None, actions=None, rtgs=None, timesteps=None):
    """
    take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
    the sequence, feeding the predictions back into the model each time. Clearly the sampling
    has quadratic complexity unlike an RNN that is only linear, and has a finite context window
    of block_size, unlike an RNN that has an infinite context window.
    """
    block_size = model.get_block_size()
    model.eval()
    for k in range(steps):
        # x_cond = x if x.size(1) <= block_size else x[:, -block_size:] # crop context if needed
        x_cond = x if x.size(1) <= block_size//3 else x[:, -block_size//3:] # crop context if needed
        if actions is not None:
            actions = actions if actions.size(1) <= block_size//3 else actions[:, -block_size//3:] # crop context if needed
        rtgs = rtgs if rtgs.size(1) <= block_size//3 else rtgs[:, -block_size//3:] # crop context if needed
        logits, _ = model(x_cond, actions=actions, targets=None, rtgs=rtgs, timesteps=timesteps)
        # pluck the logits at the final step and scale by temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop probabilities to only the top k options
        if top_k is not None:
            logits = top_k_logits(logits, top_k)
        # apply softmax to convert to probabilities
        probs = F.softmax(logits, dim=-1)
        # sample from the distribution or take the most likely
        if sample:
            ix = torch.multinomial(probs, num_samples=1)
        else:
            _, ix = torch.topk(probs, k=1, dim=-1)
        # append to the sequence and continue
        # x = torch.cat((x, ix), dim=1)
        x = ix

    return x


def resize(shape):
    """Resizes array to the given shape."""
    if len(shape) != 2:
        raise ValueError("Resize shape has to be 2D, given: %s." % str(shape))
    # Image.resize takes (width, height) as output_shape argument.
    image_shape = (shape[1], shape[0])

    def resize_fn(array):
        if len(array.shape) == 3:
            pil_image = Image.fromarray((array * 255).astype(np.uint8), "RGB")
        else:
            pil_image = Image.fromarray(array)
        image = pil_image.resize(image_shape, Image.BILINEAR)
        image = np.array(image)
        return image

    return resize_fn


class Deque:
    """Double ended queue with a maximum length and initial values."""

    def __init__(self, max_length: int, initial_values=None):
        self._deque = collections.deque(maxlen=max_length)
        self._initial_values = initial_values or []

    def reset(self) -> None:
        self._deque.clear()
        self._deque.extend(self._initial_values)

    def __call__(self, value) -> collections.deque:
        self._deque.append(value)
        return self._deque


def trailing_zero_pad(length: int):
    """Adds trailing zero padding to array lists to ensure a minimum length."""

    def trailing_zero_pad_fn(arrays):
        padding_length = length - len(arrays)
        if padding_length <= 0:
            return arrays
        zero = np.zeros_like(arrays[0])
        return arrays + [zero] * padding_length

    return trailing_zero_pad_fn


class StatePreprocessor:
    """Preprocesses state observations for training."""
    
    def __init__(self, env_shape=(84, 84), num_stacked_frames=4):
        # preprocessor functions (taken from dqn_zoo to match)
        # could be optimized, but would prefer to keep code consistent with dqn_zoo
        self._resize_fn = resize(shape=env_shape)
        self._deque_fn = Deque(max_length=num_stacked_frames)
        self._trailing_zeros = trailing_zero_pad(length=num_stacked_frames)

    def preprocess(self, state):
        """
        Preprocesses a single state observation.
        """
        rgb_state = 255 * np.tensordot(state, [0.299, 0.587, 1 - (0.299 + 0.587)], (-1, 0))
        state = self._resize_fn(rgb_state)
        state = self._deque_fn(state)
        state = list(state)
        state = self._trailing_zeros(state)
        state = torch.from_numpy(np.stack(state, axis=-1))

        return state

def render_frame(x, save_path):
    fig = plt.figure()
    plt.imshow(x)
    fig.savefig(save_path, dpi=20)
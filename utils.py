from PIL import Image
import numpy as np
import torch

def check_shape(x):
    if isinstance(x, Image.Image):
        return x.size
    elif isinstance(x, np.ndarray):
        return x.shape
    elif isinstance(x, list):
        return np.array(x).shape
    elif isinstance(x, torch.Tensor):
        return x.shape

def tensor_to_PIL(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if x.shape[0] == 1:
        x = x[0]
    x = np.transpose(x, (1, 2, 0))
    return Image.fromarray((x*255).astype(np.uint8))

                
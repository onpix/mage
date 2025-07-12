import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from PIL import Image

def apply_turbo_colormap(input_tensor):
    batchsize, channels, h, w = input_tensor.shape
    assert channels == 1, "Input tensor must have 1 channel"

    # Convert to numpy array for matplotlib
    input_np = input_tensor.squeeze(1).cpu().numpy()  # Shape: [batchsize, h, w]

    # Prepare the colormap
    cmap = plt.get_cmap('turbo')

    # Prepare output tensor
    output_tensor = torch.zeros((batchsize, 3, h, w))

    for i in range(batchsize):
        # Normalize the input to 0-1 range
        img = input_np[i]
        img_min, img_max = img.min(), img.max()
        if img_max > img_min:
            img = (img - img_min) / (img_max - img_min)
        else:
            img = img - img_min  # Avoid division by zero

        # Apply colormap
        colored_img = cmap(img)  # Shape: [h, w, 4] RGBA

        # Discard the alpha channel
        colored_img = colored_img[..., :3]  # Shape: [h, w, 3]

        # Convert back to torch tensor
        colored_img_torch = torch.from_numpy(colored_img).permute(2, 0, 1)  # Shape: [3, h, w]

        output_tensor[i] = colored_img_torch

    return output_tensor
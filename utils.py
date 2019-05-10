import torch
import numpy as np

channels = [
    (0, 0, 0),  # Ground, black
    (0, 1, 0),  # Trees, green
    (0, 0, 1),  # Towers, red
    (1, 0, 0),  # Wires, blue
    (1, 1, 1),  # Copter, white
    (0, 1, 1)   # Grass, yellow
]


def probs_to_image(probs: torch.Tensor):    # probs: (channels, height, width) [0, 1]
    with torch.no_grad():
        while len(probs.shape) > 3:
            probs = probs.squeeze(0)
        image = torch.zeros(probs.shape[-2:] + (3,), dtype=torch.uint8)
        vals, indices = torch.max(probs, dim=-3)
        for c in range(min(len(channels), probs.shape[0])):
            color = torch.tensor(channels[c], dtype=torch.uint8) * 255
            image[indices == c] = color
        return image.cpu().numpy()  # image (height, width, 3) [0, 255]


def image_to_probs(image: np.ndarray):
    return ...

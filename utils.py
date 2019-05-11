import torch
import numpy as np

channels = [
    (0, 0, 0),  # Ground, black
    (0, 1, 0),  # Trees, green
    (0, 0, 1),  # Towers, red
    (1, 0, 0),  # Wires, blue
    (1, 1, 1),  # Copter, white
    (0, 1, 1)   # Bush, yellow
]


def visualize(source: np.ndarray, result: np.ndarray): # (height, width, 3)
    return source | result


def probs_to_image(probs: torch.Tensor):    # probs: (channels, height, width) [0, 1]
    with torch.no_grad():
        image = torch.zeros(probs.shape[:-3] + probs.shape[-2:] + (3,), dtype=torch.uint8)
        vals, indices = torch.max(probs, dim=-3)
        for c in range(min(len(channels), probs.shape[0])):
            color = torch.tensor(channels[c], dtype=torch.uint8) * 255
            image[indices == c] = color
        return image.cpu().numpy()  # image (height, width, 3) [0, 255]


def image_to_probs(image: np.ndarray, device=None):
    result = torch.zeros((len(channels),) + image.shape[-3:-1], device=device, dtype=torch.uint8)
    image = torch.tensor(np.moveaxis(image, -1, -3), device=device) > 128
    for channel, color in enumerate(channels):
        color = torch.tensor(color, dtype=torch.uint8, device=device).unsqueeze(-1).unsqueeze(-1)
        mask = (image == color).all(dim=-3)
        result[..., channel, :, :] = mask
    result[0, result.any(dim=-3) == 0] = 1
    return result.to(torch.float32)

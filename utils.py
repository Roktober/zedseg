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
    return source | (result & 192)


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


def image_to_tensor(image: np.ndarray, device=None):
    data = np.moveaxis(image, -1, -3).astype(np.float32) / 255
    data = torch.tensor(data, device=device)
    if len(data.shape) < 4:
        data = data.unsqueeze(0)
    return data


def calc_accuracy(result: torch.Tensor, target: torch.Tensor):
    ri = torch.argmax(result, dim=-3)
    rt = torch.argmax(target, dim=-3)


def get_device():
    print('PyTorch version:', torch.__version__)
    USE_CUDA = torch.cuda.is_available()
    print('Use CUDA:', torch.version.cuda if USE_CUDA else False)
    return torch.device('cuda' if USE_CUDA else 'cpu')

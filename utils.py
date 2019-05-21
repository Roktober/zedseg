import json
import torch
import numpy as np
from os.path import join
from pyzed import sl
from collections import namedtuple

sequences = {'sar': join('sar', 'rec2018_07_21-%s.svo')}
views = {'l': sl.VIEW.VIEW_LEFT, 'r': sl.VIEW.VIEW_RIGHT}


def decode_name(name: str, svo_dir: str):
    seq_name, seq_idx, view = name.split('-')
    return join(svo_dir, sequences[seq_name] % seq_idx), views[view]


channels = [
    (0, 0, 0),  # Ground, black
    (0, 1, 0),  # Trees, green
    (0, 1, 1),  # Bush, yellow
    (0, 0, 1),  # Towers, red
    (1, 0, 0),  # Wires, blue
    (1, 1, 1),  # Copter, white
    (1, 0, 1),  # Cars, magenta
    (1, 1, 0),  # Buildings, cyan
]

channel_names = ['ground', 'tree', 'bush', 'tower', 'wire', 'copter', 'car', 'build']


def visualize(source: np.ndarray, result: np.ndarray):  # (height, width, 3)
    return source | (result & 192)


def probs_to_image(probs: torch.Tensor, mask: torch.Tensor = None):    # probs: (channels, height, width) [0, 1]
    with torch.no_grad():
        image = torch.zeros(probs.shape[:-3] + probs.shape[-2:] + (3,), dtype=torch.uint8)
        vals, indices = torch.max(probs, dim=-3)
        for c in range(min(len(channels), probs.shape[-3])):
            color = torch.tensor(channels[c], dtype=torch.uint8) * 255
            image[indices == c] = color
        if mask is not None:
            image[mask.squeeze(-3) == 0] = torch.tensor((128, 128, 128), dtype=torch.uint8)
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


def check_accuracy(result: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        values, indexes = result.max(dim=-3, keepdim=True)
        result = result == values
        target = target > 0
        target_size = target.shape[-3]
        mat = torch.empty((target_size, result.shape[-3]), dtype=torch.int64)
        for i in range(target_size):
            row = (result * target[..., i:i + 1, :, :]).sum((-2, -1))
            while len(row.shape) > 1:
                row = row.sum(0)
            mat[i] = row
        return mat


def acc_to_str(acc: torch.Tensor, names=None) -> str:
    result = []
    if names is None:
        names = channel_names
    for i, name in enumerate(names):
        item = acc[i, i].item()
        target_sum = acc[i].sum().item()
        result_sum = acc[:, i].sum().item()
        t, r = tuple('%.1f%%' % (item / s * 100) if s > 0 else '???' for s in [target_sum, result_sum])
        result.append('%s %s, %s' % (name, t, r))
    return '; '.join(result)


def acc_to_details(acc: torch.Tensor) -> str:
    result = [' t \\ r  ' + ' '.join(map(lambda n: '%6s' % n, channel_names))]
    acc = acc.float() * (100 / acc.sum().item())
    for i, name in enumerate(channel_names):
        result.append('%6s: %s' % (name, ' '.join(map(lambda a: '%5.2f%%' % a, acc[i].tolist()))))
    return '\n'.join(result)


def get_device():
    print('PyTorch version:', torch.__version__)
    USE_CUDA = torch.cuda.is_available()
    print('Use CUDA:', torch.version.cuda if USE_CUDA else False)
    return torch.device('cuda' if USE_CUDA else 'cpu')


Config = namedtuple('Config', 'device model generator optimizer train')


def load_config() -> Config:
    with open('config.json', 'r') as f:
        config = json.load(f)
    return Config(*map(config.get, Config._fields))

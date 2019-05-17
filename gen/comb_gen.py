import torch
import numpy as np
from utils import channels, image_to_probs
from albumentations import (
    HorizontalFlip,
    VerticalFlip,
    CenterCrop,
    Crop,
    Compose,
    Transpose,
    RandomRotate90,
    ElasticTransform,
    GridDistortion,
    OpticalDistortion,
    RandomSizedCrop,
    OneOf,
    CLAHE,
    RandomBrightnessContrast,
    RandomGamma
)


def comb_generate(gens, resolution=(320, 320), device=None, input_channels=3):
    aug = Compose([
        RandomSizedCrop(min_max_height=(200, 320), height=resolution[0], width=resolution[1], p=1.0),
        VerticalFlip(p=0.5),
        RandomRotate90(p=0.5),
        OneOf([
            # ElasticTransform(p=0.5, alpha=12, sigma=12 * 0.05, alpha_affine=12 * 0.03),
            GridDistortion(p=0.5, distort_limit=0.1),
            OpticalDistortion(p=1, distort_limit=0.2, shift_limit=0.2)
        ], p=0.8),
        CLAHE(p=0.8),
        RandomBrightnessContrast(p=0.8),
        RandomGamma(p=0.8)
    ])

    xr = None
    yr = None
    for samples in zip(*gens):
        xs, ys, classes = zip(*samples)
        xs = sum([list(a) for a in xs], [])
        ys = sum([list(a) for a in ys], [])
        classes = sum(classes, [])
        if xr is None:
            xr = torch.empty((len(xs), input_channels) + resolution, dtype=torch.float32, device=device)
        if yr is None:
            yr = torch.empty((len(ys), len(channels)) + resolution, dtype=torch.float32, device=device)
        # assert shape[0] == len(xs) == len(ys)
        for b, (x, y) in enumerate(zip(xs, ys)):
            augmented = aug(image=x, mask=y)
            x, y = map(augmented.get, ['image', 'mask'])
            xr[b] = torch.tensor(np.moveaxis(x, -1, 0), dtype=torch.float32, device=device) / 255
            yr[b] = image_to_probs(y, device=device)
        yield xr, yr, classes

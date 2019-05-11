import torch
import numpy as np
from utils import image_to_probs
from os import listdir
from os.path import join, isdir
from random import choice
from cv2 import imread


def png_generate(batch=4, classes=None, root_dir='images', make_tensor=False, device=None):
    if classes is None:
        classes = [
            name[:-3]
            for name in listdir(root_dir)
            if name.endswith('-in') and isdir(join(root_dir, name)) and name[:-3] not in ['saved']
        ]
    files = sum([
        [
            (join(root_dir, c + '-in', fn), join(root_dir, c + '-out', fn))
            for fn in listdir(join(root_dir, c + '-in'))
            if fn.endswith('.png')
        ]
        for c in classes
    ], [])
    while True:
        inputs, outputs = [], []
        for _ in range(batch):
            f_in, f_out = choice(files)
            if make_tensor:
                inputs.append(torch.tensor(np.moveaxis(imread(f_in), -1, -3), dtype=torch.float32, device=device) / 255)
                outputs.append(image_to_probs(imread(f_out)))
            else:
                inputs.append(imread(f_in))
                outputs.append(imread(f_out))
        yield inputs, outputs

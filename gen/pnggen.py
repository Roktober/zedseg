import torch
import numpy as np
from utils import image_to_probs
from os import listdir
from os.path import join, isdir, isfile
from random import choice
from cv2 import imread


def png_generate(batch=4, classes=None, root_dir='images', make_tensor=False, device=None):
    """
    Подготовка png изображений к обучению
    """
    if classes is None:
        classes = [ # list с папками кончающимися на -in, берем название папки без -in
            name[:-3]
            for name in listdir(root_dir)
            if name.endswith('-in') and isdir(join(root_dir, name)) and name[:-3] not in ['saved']
        ] 
    files = sum([ # соответствие файлов из папок ин и аут
        [
            (join(root_dir, c + '-in', fn), join(root_dir, c + '-out', fn), c)
            for fn in listdir(join(root_dir, c + '-in'))
            if fn.endswith('.png') and isfile(join(root_dir, c + '-in', fn)) and isfile(join(root_dir, c + '-out', fn))
        ]
        for c in classes
    ], [])

    while True:
        inputs, outputs = [], []
        for _ in range(batch):
            f_in, f_out, f_class = choice(files)
            out = image_to_probs(imread(f_out))
            if f_class == 'part':
                out[:3] = 0 # убираем 3 первых класса
            if make_tensor:
                inputs.append(torch.tensor(np.moveaxis(imread(f_in), -1, -3), dtype=torch.float32, device=device) / 255)
                outputs.append(out)
            else:
                inputs.append(imread(f_in))
                outputs.append(np.moveaxis(out.numpy(), 0, -1))
        yield inputs, outputs

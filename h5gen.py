from h5py import File
import torch
import numpy as np
from random import choice, randrange
from scipy.ndimage.interpolation import rotate, geometric_transform, affine_transform


def augment(image, resolution):
    # affine_transform()
    # geometric_transform()
    return image


def h5generate(batch=4, classes=None, channels=3, source='cuts.h5', resolution=(320, 320), device=None):
    with File(source, mode='r') as file:
        groups = [
            file[group_name]
            for group_name in file
            if classes is None or int(file[group_name].attrs['type']) in classes
        ]
        if classes is None:
            classes = list(map(lambda g: g.attrs['type'], groups))
        output_batch = torch.empty((batch * len(groups), len(classes)) + resolution, device=device)
        input_batch = torch.empty((batch * len(groups), channels) + resolution, device=device)
        for gi, group in enumerate(groups):
            gt = group.attrs['type']
            scalar = [1 if i == gt else 0 for i in classes]
            output_batch[gi * batch:(gi + 1) * batch] = torch.tensor(
                scalar, dtype=torch.float32, device=device
            ).unsqueeze(-1).unsqueeze(-1)
        while True:
            for gi, group in enumerate(groups):
                for o in range(gi * batch, (gi + 1) * batch):
                    cut = group[choice(list(group))]
                    image = cut[randrange(cut.shape[0])].astype(np.float32) / 255
                    result = torch.tensor(augment(image, resolution), device=device)
                    input_batch[o] = result
            yield input_batch, output_batch


if __name__ == "__main__":
    h5generate()

from h5py import File
import torch
import numpy as np
from random import choice, randrange
from utils import channels


def augment(image, resolution):
    # affine_transform()
    # geometric_transform()
    return image


def h5_generate(batch=4, classes=None, channel_count=3, source='cuts.h5',
                resolution=(320, 320), device=None, make_tensor=False):
    with File(source, mode='r') as file:
        groups = [
            file[group_name]
            for group_name in file
            if classes is None or int(file[group_name].attrs['type']) in classes
        ]
        if classes is None:
            classes = list(map(lambda g: g.attrs['type'], groups))
        if make_tensor:
            output_batch = torch.empty((batch * len(groups), len(classes)) + resolution, device=device)
            input_batch = torch.empty((batch * len(groups), channel_count) + resolution, device=device)
        else:
            output_batch = np.empty((batch * len(groups),) + resolution + (channel_count,), dtype=np.uint8)
            input_batch = np.empty((batch * len(groups),) + resolution + (channel_count,), dtype=np.uint8)
        for gi, group in enumerate(groups):
            gt = group.attrs['type']
            scalar = [1 if i == gt else 0 for i in classes]
            if make_tensor:
                output_batch[gi * batch:(gi + 1) * batch] = torch.tensor(
                    scalar, dtype=torch.float32, device=device
                ).unsqueeze(-1).unsqueeze(-1)
            else:
                color = np.array(channels[gt], dtype=np.uint8) * 255
                output_batch[gi * batch:(gi + 1) * batch] = color
        while True:
            for gi, group in enumerate(groups):
                for o in range(gi * batch, (gi + 1) * batch):
                    cut = group[choice(list(group))]
                    if make_tensor:
                        image = cut[randrange(cut.shape[0])].astype(np.float32) / 255
                        result = torch.tensor(augment(image, resolution), device=device)
                    else:
                        result = np.moveaxis(cut[randrange(cut.shape[0])], 0, -1)
                    input_batch[o] = result
            yield input_batch, output_batch


if __name__ == "__main__":
    h5_generate()

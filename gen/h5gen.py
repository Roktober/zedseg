from h5py import File
import torch
import numpy as np
from random import choice, randrange
from utils import channels, channel_names


def augment(image, resolution):
    # affine_transform()
    # geometric_transform()
    return image


def h5_generate(batch=4, classes=None, input_channels=3, source='cuts.h5',
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
            output_batch = torch.empty((batch * len(groups), len(channels), 1, 1), device=device)
            input_batch = torch.empty((batch * len(groups), input_channels) + resolution, device=device)
        else:
            output_batch = np.empty((batch * len(groups),) + (1, 1, len(channels)), dtype=np.uint8)
            input_batch = np.empty((batch * len(groups),) + resolution + (input_channels,), dtype=np.uint8)
        # output_classes = []
        for gi, group in enumerate(groups):
            gt = group.attrs['type']
            channel_name = group.attrs['class_name']
            scalar = [1 if cn == channel_name else 0 for cn in channel_names]
            if make_tensor:
                output_batch[gi * batch:(gi + 1) * batch] = torch.tensor(
                    scalar, dtype=torch.float32, device=device
                ).unsqueeze(-1).unsqueeze(-1)
            else:
                # color = np.array(channels[gt], dtype=np.uint8) * 255
                output_batch[gi * batch:(gi + 1) * batch] = np.array(scalar, dtype=np.uint8)
            # output_classes += [channel_names[gt]] * batch
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

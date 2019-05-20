from h5py import File
import numpy as np
from random import choice, shuffle


def cvat_generate(batch: int = 4, source='cvat.h5', with_shuffle=True, infinite=True):
    batch_data = None
    batch_idx = 0
    with File(source, 'r') as file:
        groups = list(file)
        if with_shuffle:
            shuffle(groups)
        while True:
            for group_name in groups:
                group = file[group_name]
                images_ds, targets_ds = group['images'], group['targets']
                frames = list(range(images_ds.shape[0]))
                if with_shuffle:
                    shuffle(frames)
                for frame in frames:
                    image, target = map(lambda i: np.moveaxis(i[frame], 0, -1), [images_ds, targets_ds])
                    mask = target.any(axis=-1)
                    if batch_data is None:
                        batch_data = [
                            np.empty((batch,) + mask.shape + w, dtype=np.uint8)
                            for w in [(3,), (target.shape[-1],), ()]
                        ]
                    for d, s in zip(batch_data, [image, target, mask]):
                        d[batch_idx] = s
                    batch_idx += 1
                    if batch_idx >= batch:
                        batch_idx = 0
                        yield batch_data
            if not infinite:
                break

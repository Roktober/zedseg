import csv
import torch
import numpy as np
from collections import namedtuple
from os.path import isfile, join
import pyzed.sl as sl
from random import choice
import cv2


Cut = namedtuple('Cut', 'file left top start stop type')


def save(cuts: [Cut], name='cuts.csv'):
    with open(name, 'w') as f:
        w = csv.writer(f)
        w.writerow(Cut._fields)
        for cs in cuts.values():
            for cut in cs:
                w.writerow(list(cut))


def load(cuts: {Cut}):
    if isfile('cuts.csv'):
        with open('cuts.csv', 'r') as f:
            reader = csv.reader(f)
            is_title = True
            for row in reader:
                if is_title:
                    is_title = False
                    continue
                cut = Cut(*[
                    t if i == 0 else (None if t == '' else int(t))
                    for i, t in enumerate(row)
                ])
                cs = cuts.get(cut.file, None)
                if cs is None:
                    cuts[cut.file] = [cut]
                else:
                    cs.append(cut)


def open_cut(cuts: {Cut}, class_num, base_dir):
    cut = choice([c for c in sum(cuts.values(), []) if c.type == class_num])
    init = sl.InitParameters(svo_input_filename=join(base_dir, cut.file), svo_real_time_mode=False)
    cam = sl.Camera()
    status = cam.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit()
    cam.set_svo_position(cut.start)
    return cut, cam


def get_image(sources, i, size=320):
    cut, cam = sources[i]
    runtime = sl.RuntimeParameters()
    err = cam.grab(runtime)
    if err in [sl.ERROR_CODE.SUCCESS, sl.ERROR_CODE.ERROR_CODE_NOT_A_NEW_FRAME]:
        mat = sl.Mat()
        cam.retrieve_image(mat)
        position = cam.get_svo_position()
        if (cut.stop is not None and position >= cut.stop) or (err == sl.ERROR_CODE.ERROR_CODE_NOT_A_NEW_FRAME):
            sources[i] = None
        return np.moveaxis(mat.get_data()[cut.top:cut.top + size, cut.left:cut.left + size, :3], -1, 0)
    else:
        raise RuntimeError('Grab error')


def generate(cuts: {Cut}, widths, size=320, base_dir='/media/igor/Terra/svo', device=None):
    classes = len(widths)
    sources = [None] * classes
    input_batch = torch.empty((sum(widths.values()), 3, size, size))
    output_batch = torch.empty((sum(widths.values()), classes, size, size), device=device)
    o = 0
    for c, w in widths.items():
        scalar = [1 if i == c else 0 for i in range(classes)]
        output_batch[o: o + w] = torch.tensor(scalar, dtype=torch.float32, device=device).unsqueeze(-1).unsqueeze(-1)
        o += w
    while True:
        o = 0
        for i, (c, w) in enumerate(widths.items()):
            for t in range(o, o + w):
                if sources[i] is None:
                    sources[i] = open_cut(cuts, c, base_dir)
                image = get_image(sources, i)
                input_batch[t] = torch.tensor(image).float() / 255

            o += w
        yield input_batch.to(device), output_batch

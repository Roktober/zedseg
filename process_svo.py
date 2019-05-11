import json
import torch
from unet import UNet
from os import listdir
from os.path import join, isfile
import pyzed.sl as sl
import numpy as np
import cv2
from read_svo import read_svo
from os.path import join, isfile, isdir
from os import mkdir
from utils import probs_to_image, visualize


print('PyTorch version:', torch.__version__)
USE_CUDA = torch.cuda.is_available()
print('Use CUDA:', torch.version.cuda if USE_CUDA else False)
device = torch.device('cuda' if USE_CUDA else 'cpu')


def get_save_idx(images_dir='images', image_fmt='%3d.png'):
    in_dir, out_dir = join(images_dir, 'saved-in'), join(images_dir, 'saved-out')
    if not isdir(in_dir):
        mkdir(in_dir)
    if not isdir(out_dir):
        mkdir(out_dir)
    n = 1
    while isfile(join(in_dir, image_fmt % n)) or isfile(join(out_dir, image_fmt % n)):
        n += 1
    return n


def main(files=None, show=True, images_dir='images', image_fmt='%.3d.png'):
    with open('config.json', 'r') as f:
        config = json.load(f)
    base_dir = config['svo_dir']
    model = UNet(**config['unet'])
    model.load_state_dict(torch.load('models/unet2.pt'))
    model.eval()
    model = model.to(device)
    if files is None:
        files = [
            join(base_dir, 'sar', fn)
            for fn in listdir(join(base_dir, 'sar'))
            if isfile(join(base_dir, 'sar', fn)) and fn.endswith('.svo')
        ]

    runtime = sl.RuntimeParameters()
    mat = sl.Mat()
    pause = False
    save_idx = get_save_idx(images_dir, image_fmt)
    with torch.no_grad():
        for fn in files:
            for source in read_svo(fn):
                # if show:
                #    cv2.imshow('input', source)
                data = np.moveaxis(source, -1, 0).astype(np.float32) / 255
                data = model(torch.tensor(data, device=device).unsqueeze(0))  # [:, :2]
                result = probs_to_image(data)
                # data = np.moveaxis((data.detach().squeeze(0).cpu().numpy() * 255).astype(np.uint8), 0, -1)
                # data = np.concatenate((data, np.zeros(data.shape[:2] + (1,), dtype=np.uint8)), axis=2)
                if show:
                    # mix = source
                    # mix[data[..., 0] < data[..., 1], 1] = 200
                    cv2.imshow('output', visualize(source, result))
                    while True:
                        key = cv2.waitKey(1)
                        if key == ord('p'):
                            pause = not pause
                        if key == ord('s'):
                            result = probs_to_image(data)
                            cv2.imwrite(join('images', 'saved-in', image_fmt % save_idx), source)
                            cv2.imwrite(join('images', 'saved-out', image_fmt % save_idx), result)
                            save_idx += 1
                        if not pause:
                            break


if __name__ == "__main__":
    main()  # files=['svo/sar/rec2018_07_21-33.svo'])

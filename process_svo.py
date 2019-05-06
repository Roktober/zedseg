import json
import torch
from unet import UNet
from os import listdir
from os.path import join, isfile
import pyzed.sl as sl
import numpy as np
import cv2
from read_svo import read_svo


print('PyTorch version:', torch.__version__)
USE_CUDA = torch.cuda.is_available()
print('Use CUDA:', torch.version.cuda if USE_CUDA else False)
device = torch.device('cuda' if USE_CUDA else 'cpu')


def main(files=None, show=True):
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
    with torch.no_grad():
        for fn in files:
            for source in read_svo(fn):
                # if show:
                #    cv2.imshow('input', source)
                data = np.moveaxis(source, -1, 0).astype(np.float32) / 255
                data = model(torch.tensor(data, device=device).unsqueeze(0))  # [:, :2]
                data = np.moveaxis((data.detach().squeeze(0).cpu().numpy() * 255).astype(np.uint8), 0, -1)
                data = np.concatenate((data, np.zeros(data.shape[:2] + (1,), dtype=np.uint8)), axis=2)
                if show:
                    mix = source
                    mix[data[..., 0] < data[..., 1], 1] = 200
                    cv2.imshow('output', mix)
                    cv2.waitKey(1)


if __name__ == "__main__":
    main()  # files=['svo/sar/rec2018_07_21-33.svo'])

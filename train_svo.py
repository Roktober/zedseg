from unet import UNet, UNetLayerInfo
from csv_cut import load, generate
import numpy as np
import json
import cv2
import torch


print('PyTorch version: ', torch.__version__)
USE_CUDA = torch.cuda.is_available()
print('Use CUDA: ', torch.version.cuda if USE_CUDA else False)
device = torch.device('cuda' if USE_CUDA else 'cpu')


def show_images(image_torch, name):
    try:
        image = (image_torch.detach().cpu().numpy() * 255).astype(np.uint8)
    except RuntimeError as e:
        raise e
    b, c, ih, iw = image.shape
    if c < 3:
        image = np.concatenate((image, np.zeros((b, 3 - c, ih, iw), dtype=np.uint8)), axis=1)
    h = 2
    w = image.shape[0] // h
    result = np.empty((ih * h, iw * w, 3), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            result[y * ih:y * ih + ih, x * iw:x * iw + iw] = np.moveaxis(image[y * w + x], 0, -1)
    cv2.imshow(name, result)


def main():
    with open('config.json', 'r') as f:
        config = json.load(f)
    base_dir = config['svo_dir']
    batch = config['svo_batch']
    model = UNet(**config['unet']).to(device)
    cuts = {}
    load(cuts, base_dir=base_dir)
    for x, target in generate(cuts, {0: batch, 1: batch}, device=device, base_dir=base_dir):
        # y = model(x)
        show_images(x, 'x')
        # show_images(y, 'y')
        show_images(target, 'target')
        key = cv2.waitKey(1)


if __name__ == "__main__":
    main()

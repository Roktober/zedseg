from unet import UNet, UNetLayerInfo
from csv_cut import load, generate
import numpy as np
import json
import cv2
import torch
from h5gen import h5generate
from torch.nn import BCELoss
from torch.optim import Adam

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
    # cuts = {}
    # load(cuts, base_dir=base_dir)
    loss_f = BCELoss()
    opt = Adam(model.parameters(), lr=1e-4)
    for i, (x, target) in enumerate(h5generate(batch, [0, 1], device=device)):
        opt.zero_grad()
        y = model(x)
        loss = loss_f(y, target)
        loss.backward()
        opt.step()
        show_images(x, 'input')
        show_images(y, 'output')
        show_images(target, 'target')
        if i % 100 == 0:
            torch.save(model.state_dict(), 'models/unet1.pt')
        key = cv2.waitKey(1)


if __name__ == "__main__":
    main()

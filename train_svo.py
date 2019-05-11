from unet import UNet, UNetLayerInfo
from csv_cut import load, generate
import numpy as np
import json
import cv2
import torch
from utils import channels, probs_to_image
from gen import h5_generate, png_generate, comb_generate
from torch.nn import BCELoss
from torch.optim import Adam

print('PyTorch version:', torch.__version__)
USE_CUDA = torch.cuda.is_available()
print('Use CUDA:', torch.version.cuda if USE_CUDA else False)
device = torch.device('cuda' if USE_CUDA else 'cpu')


def show_images(image_torch, name, is_mask=False):
    if is_mask:
        image = probs_to_image(image_torch)
    else:
        try:
            image = (image_torch.detach().cpu().numpy() * 255).astype(np.uint8)
            image = np.moveaxis(image, -3, -1)
        except RuntimeError as e:
            raise e
    b, ih, iw, c = image.shape
    h = 2
    w = image.shape[0] // h
    result = np.empty((ih * h, iw * w, 3), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            result[y * ih:y * ih + ih, x * iw:x * iw + iw] = image[y * w + x]
    cv2.imshow(name, result)


def main():
    with open('config.json', 'r') as f:
        config = json.load(f)
    base_dir = config['svo_dir']
    batch = config['svo_batch']
    assert len(channels) == config['unet']['n_classes']
    model = UNet(**config['unet']).to(device)
    model.train()
    # cuts = {}
    # load(cuts, base_dir=base_dir)
    loss_f = BCELoss()
    opt = Adam(model.parameters(), lr=1e-4)
    for x, target in comb_generate(h5_generate(batch, [0, 1]), png_generate(), device=device):
        opt.zero_grad()
        y = model(x)
        loss = loss_f(y, target)
        loss.backward()
        opt.step()
        show_images(x, 'input')
        show_images(y, 'output', is_mask=True)
        show_images(target, 'target', is_mask=True)
        key = cv2.waitKey(1)
        if key == ord('s'):
            torch.save(model.state_dict(), 'models/unet2.pt')
        elif key == ord('q'):
            break


if __name__ == "__main__":
    main()

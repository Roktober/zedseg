from model import UNet
import numpy as np
import time
import json
import cv2
import torch
from utils import channels, probs_to_image, get_device
from gen import h5_generate, png_generate, comb_generate
from torch.nn import BCELoss
from torch.optim import Adam


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


def main(with_gui=None):
    device = get_device()
    with open('config.json', 'r') as f:
        config = json.load(f)
    base_dir, svo_batch, png_batch, epoch_images = map(config.get, [
        'svo_dir', 'svo_batch', 'png_batch', 'epoch_images'
    ])
    if with_gui is None:
        with_gui = config.get('with_gui')
    assert len(channels) == config['unet']['n_classes']
    model = UNet(**config['unet']).to(device)
    model.train()
    loss_f = BCELoss()
    opt = Adam(model.parameters(), lr=1e-4)
    pause = False
    images, count, loss_sum, epoch = 0, 0, 0, 1
    for x, target in comb_generate(h5_generate(svo_batch, [0, 1]), png_generate(png_batch),
                                   shape=(svo_batch + png_batch, len(channels), 320, 320), device=device):
        opt.zero_grad()
        y = model(x)
        loss = loss_f(y, target)
        loss_sum += loss.item()
        count += 1
        images += len(x)
        loss.backward()
        opt.step()
        if with_gui:
            if not pause:
                show_images(x, 'input')
                show_images(y[:, :, ::2, ::2], 'output', is_mask=True)
                show_images(target[:, :, ::2, ::2], 'target', is_mask=True)
            key = cv2.waitKey(1)
            if key == ord('s'):
                torch.save(model.state_dict(), 'models/unet2.pt')
            elif key == ord('p'):
                pause = not pause
            elif key == ord('q'):
                break
        if images >= epoch_images:
            msg = '%s Epoch %d: train loss %f' % (time.strftime('%Y-%m-%d %H:%M:%S'), epoch, loss_sum / count)
            print(msg)
            count = 0
            images = 0
            loss_sum = 0
            epoch += 1


if __name__ == "__main__":
    main()

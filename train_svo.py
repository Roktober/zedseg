from unet import UNet, UNetLayerInfo
from csv_cut import load, generate
import numpy as np
import cv2


def show_images(image, name):
    image = (image.cpu().numpy() * 255).astype(np.uint8)
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
    model = UNet(
        3, 4,
        downs=[
            (64, 2),
            (128, 2),
            (256, 2),
            (512, 2),
            (512, None)
        ],
        ups=[
            (64, 2),
            (64, 2),
            (128, 2),
            (256, 2)
        ]
    )
    cuts = {}
    load(cuts)
    for x, y in generate(cuts, {0: 4, 1: 4}):
        show_images(x, 'x')
        show_images(y, 'y')
        cv2.waitKey(1)
        print()
    print()


if __name__ == "__main__":
    main()

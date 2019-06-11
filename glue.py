from os.path import basename, join
import numpy as np
import cv2
import csv
from math import sqrt

def mul_2d(a, b):
    m = np.matmul(a[:, :2], b)
    m[:, 2] += a[:, 2]
    return m


def normalize(d):
    n = np.linalg.norm(d[:, 0])
    return d / np.sqrt(n)


def smooth(mat):
    n = np.linalg.norm(mat[:, 0])
    result = np.copy(mat)
    result[:, :2] /= n
    sin = result[1, 0].item()
    sin *= 0.95
    result[0, 2] *= 0.95
    cos = sqrt(1.0 - sin * sin)
    result[:, :2] = np.array([[cos, -sin], [sin, cos]])
    return result


def main():
    points_const = np.array([[x, y, 1] for x, y in [(0, 0), (1280, 0), (1280, 720), (0, 720)]], dtype=float).transpose()
    width, height = 250, 10000
    image = np.zeros((height, width, 3), dtype=np.uint8)

    mask_image = np.zeros_like(image, dtype=np.uint32)
    count_image = np.zeros((height, width), dtype=np.uint32)
    # one_image = np.ones((720, 1280), dtype=np.uint8)

    mat = None
    vn, cap = None, None
    ctr, col = 0, 0
    with open('fly1.csv', 'r') as file:
        for name, frame, diff in csv.reader(file):
            if name == 'name':
                continue
            name = basename(name)
            frame = int(frame)
            if vn != name:
                vn = name
                cap = cv2.VideoCapture(join('D:\\bag\\unet512-2', vn))
                fn = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                if fn != frame:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ret, mask = cap.read()
            diff = np.array(list(map(float, diff.split(' ')))).reshape((2, 3))
            diff = normalize(diff)
            mat = diff if mat is None else smooth(mul_2d(diff, mat))
            points = np.matmul(mat, points_const).transpose()
            points = points / 10 + np.array((400, 10))
            points = points.reshape((-1, 1, 2)).astype(np.int32)
            cv2.polylines(image, [points], True, (255, 0, 0))
            # cv2.imshow('contour', image)

            mm = np.copy(mat)
            mm[0] /= 5
            mm[1] /= 5
            mm[:, 2] += np.array((10, 10))
            # ctr += 1
            # if mm[1, 2] > 1000:
            #    col += 1
            #    mat[1, 2] = 0
            #    ctr = 0

            # one_image = np.ones((720, 1280), dtype=np.uint8)
            rem = mask < 150
            one_image = np.any(rem, axis=2)
            rem[:, :, 0] = ~rem[:, :, 0]
            one_image[np.all(rem, axis=2)] = 0

            # cv2.imshow('one_image', one_image.astype(np.uint8) * 255)
            count_image += cv2.warpAffine(one_image.astype(np.uint8), mm, (width, height))
            ci = np.expand_dims(count_image, -1)
            mask[~one_image] = 0
            mask_image += cv2.warpAffine(mask, mm, (width, height))
            result = mask_image.astype(np.float32) / ci.astype(np.float32)
            result = result.astype(np.uint8)
            result[count_image == 0] = 0
            h = 1000
            result = np.concatenate([result[y:y + h] for y in range(0, height, h)], axis=1)
            cv2.imshow('result', result)

            cv2.waitKey(1)
    return


if __name__ == "__main__":
    main()

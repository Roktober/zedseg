import numpy as np
from argparse import ArgumentParser
from utils import decode_name, image_to_probs
from os.path import join
import cv2
import csv


def frame_stats_np(frame):
    green, yellow, black = [
        np.all((frame > 128) == np.array(c, dtype=np.bool), axis=-1)
        for c in [(0, 1, 0), (0, 1, 1), (0, 0, 0)]
    ]
    # cv2.imshow('black', black.astype(np.uint8) * 255)
    tree = green.sum().item()
    bush = yellow.sum().item()
    ground = black.sum().item()
    sum = tree + bush + ground
    return tree / sum, bush / sum, ground / sum


def main():
    parser = ArgumentParser(description='Create report on processed .svo files')
    parser.add_argument('input', type=str, help='Input directory or file')
    args = parser.parse_args()
    names = decode_name(args.input)
    with open('stats.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['name', 'frame', 'tree', 'bush', 'ground'])
        for _, name, _ in names:
            fn = join('D:\\bag\\unet512-2', name + '.mp4')
            cap = cv2.VideoCapture(fn)
            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    tree, bush, ground = frame_stats_np(frame)
                    writer.writerow([name, frame_idx, tree, bush, ground])
                    frame_idx += 1
                    # probs = image_to_probs(frame)
                    cv2.imshow('v', frame)
                    cv2.waitKey(1)
                else:
                    break


def make_chart():
    with open('stats.csv', 'r') as file:
        reader = csv.reader(file)
        trees = np.array([float(row[2]) for i, row in enumerate(reader) if i > 0])
    count = trees.shape[0]
    w, h = 800, 600
    step = count // w
    trees = np.reshape(trees[:step * w], (w, step)).mean(axis=1)
    chart = np.full((h, w, 3), 255, dtype=np.uint8)
    # chart[(trees * 600).round().astype(np.int), np.arange(w)] = (255, 0, 0)
    trees = np.stack((np.arange(w), (trees * 600).round().astype(np.int)), axis=1)
    trees = trees.reshape((-1, 1, 2))
    cv2.polylines(chart, [trees], False, (255, 0, 0))
    cv2.imshow('Chart', chart)
    cv2.waitKey()
    ...


if __name__ == "__main__":
    # main()
    make_chart()

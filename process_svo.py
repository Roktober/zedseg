import re
import json
import torch
from argparse import ArgumentParser
from os import listdir
from os.path import join, isfile, isdir
import pyzed.sl as sl
import numpy as np
import cv2
from read_svo import read_svo
from os.path import join, isfile, isdir, basename
from os import mkdir
from utils import probs_to_image, visualize, image_to_tensor
from model import load_model

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


def file_idx(fn):
    result = re.search('-(\\d+)\\.svo$', fn)
    return None if result is None else int(result.group(1))


def main(show=True, images_dir='images', image_fmt='%.3d.png'):
    parser = ArgumentParser(description='Process .svo files')
    parser.add_argument('input', type=str, help='Input directory or file')
    parser.add_argument('output', type=str, nargs='?', default=None, help='Output directory or file')
    parser.add_argument('-m', type=str, required=False, help='Model to process with')
    parser.add_argument('-r', type=int, required=False, default=1, help='Reduce factor')
    parser.add_argument('-v', action='store_true', required=False, default=False, help='Mix processed with input')
    args = parser.parse_args()

    model = None if args.m is None else load_model(args.m, device=device)[0]

    if isdir(args.input):
        files = sorted([
            join(args.input, fn)
            for fn in listdir(args.input)
            if isfile(join(args.input, fn)) and fn.endswith('.svo')
        ], key=file_idx)
    elif isfile(args.input):
        files = [args.input]
    else:
        print('No such file or directory: %s' % args.input)
        return

    pause = False
    save_idx = get_save_idx(images_dir, image_fmt)

    out_path = args.output
    writer = None
    to_dir = False if out_path is None else isdir(out_path)
    with torch.no_grad():
        for fn in files:
            print('Processing %s' % fn)
            for source in read_svo(fn):

                # Processing:
                if model is not None:
                    data = image_to_tensor(source, device=device)
                    data = model(data).squeeze(0)  # [:, :2]
                    result = probs_to_image(data)
                    output = visualize(source, result) if args.v else result
                else:
                    output = source

                # Open writer:
                if writer is None and out_path is not None:
                    dst = join(out_path, basename(fn)[:-3] + 'avi') if to_dir else out_path
                    print('Write to %s' % dst)
                    writer = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                                             10, tuple(reversed(output.shape[:2])))

                # Return output:
                if writer is not None:
                    writer.write(output)
                else:
                    cv2.imshow('output', output)
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
            if to_dir and writer is not None:
                writer.release()
                writer = None
        if writer is not None:
            writer.release()


if __name__ == "__main__":
    main()  # files=['svo/sar/rec2018_07_21-33.svo'])

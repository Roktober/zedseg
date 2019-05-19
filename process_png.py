import json
import torch
from model import load_model
from os import listdir
from os.path import join, isfile
from argparse import ArgumentParser
from utils import image_to_tensor, get_device, probs_to_image, load_config, visualize
from cv2 import imread, imwrite


def main():
    parser = ArgumentParser(description='Segmenting .png files')
    parser.add_argument('-v', action='store_true', required=False, default=False, help='Mix input and output')
    parser.add_argument('input_dir', type=str, help='Input directory')
    parser.add_argument('output_dir', type=str, help='Output directory')
    args = parser.parse_args()

    device = get_device()
    model, _, _ = load_model(load_config().model, device=device)

    for fn in listdir(args.input_dir):
        path = join(args.input_dir, fn)
        if fn.endswith('.png') and isfile(path):
            print('Processing %s' % path)
            source = imread(path)
            data = image_to_tensor(source, device=device)
            data = model(data)[0]
            result = probs_to_image(data)
            if args.v:
                result = visualize(source, result)
            imwrite(join(args.output_dir, fn), result)
    print('Completed')


if __name__ == "__main__":
    main()

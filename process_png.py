import json
import torch
from model import UNet
from os import listdir
from os.path import join, isfile
from argparse import ArgumentParser
from utils import image_to_tensor, get_device, probs_to_image
from cv2 import imread, imwrite


def load_model(device):
    with open('config.json', 'r') as f:
        config = json.load(f)
    base_dir = config['svo_dir']
    model = UNet(**config['unet'])
    model.load_state_dict(torch.load('models/unet2.pt'))
    model.eval()
    return model.to(device)


def main():
    parser = ArgumentParser(description='Segmenting .png files')
    parser.add_argument('input_dir', type=str, help='Input directory')
    parser.add_argument('output_dir', type=str, help='Output directory')
    args = parser.parse_args()

    device = get_device()
    model = load_model(device)

    for fn in listdir(args.input_dir):
        path = join(args.input_dir, fn)
        if fn.endswith('.png') and isfile(path):
            print('Processing %s' % path)
            data = image_to_tensor(imread(path), device=device)
            data = model(data)[0]
            result = probs_to_image(data)
            imwrite(join(args.output_dir, fn), result)
    print('Completed')


if __name__ == "__main__":
    main()

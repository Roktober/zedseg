import torch
from torch.nn import BCELoss
from model import load_model
from os import listdir
from os.path import join, isfile
from argparse import ArgumentParser
from utils import image_to_tensor, image_to_probs, get_device, load_config, check_accuracy, acc_to_str, acc_to_details
from cv2 import imread, imwrite


def main():
    parser = ArgumentParser(description='Validate model on .png files')
    parser.add_argument('-m', type=str, required=False, default=None, help='Model to validate')
    parser.add_argument('input_dir', type=str, help='Input directory')
    parser.add_argument('target_dir', type=str, help='Output directory')
    args = parser.parse_args()

    device = get_device()
    model, _, _ = load_model(load_config().model if args.m is None else args.m, device=device)
    loss_f = BCELoss()
    count, loss_sum, acc_sum = 0, 0, 0
    with torch.no_grad():
        for fn in listdir(args.input_dir):
            input_path = join(args.input_dir, fn)
            target_path = join(args.target_dir, fn)
            if fn.endswith('.png') and isfile(input_path) and isfile(target_path):
                print('Validating %s with target %s' % (input_path, target_path))
                data = image_to_tensor(imread(input_path), device=device)
                target = image_to_probs(imread(target_path), device=device)
                data = model(data).squeeze(0)
                loss = loss_f(data, target).item()
                acc = check_accuracy(data, target)
                print('Loss %f, acc %s' % (loss, acc_to_str(acc)))
                count += 1
                acc_sum += acc
                loss_sum += loss
        print('\nSUMMARY\nLoss %f, acc %s' % (loss_sum / count, acc_to_str(acc_sum)))
        print(acc_to_details(acc_sum))


if __name__ == "__main__":
    main()

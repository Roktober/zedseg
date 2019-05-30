import re
import json
import torch
from argparse import ArgumentParser
from os import listdir
from os.path import join, isfile, isdir
import pyzed.sl as sl
import numpy as np
import cv2
from video import read_svo, open_ffmpeg, write_ffmpeg, close_ffmpeg
from os.path import join, isfile, isdir, basename
from os import mkdir
from utils import probs_to_image, visualize, image_to_tensor, decode_name
from model import load_model
from pyzed import sl
from motion import MotionEstimator

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
    with open('config.json', 'r') as f:
        config = json.load(f)
    enc_cfg = config['video_enc']
    parser = ArgumentParser(description='Process .svo files')
    parser.add_argument('input', type=str, help='Input directory or file')
    parser.add_argument('output', type=str, nargs='?', default=None, help='Output directory or file')
    parser.add_argument('-m', type=str, required=False, help='Model to process with')
    parser.add_argument('-f', type=int, required=False, default=1, help='Reduce factor')
    parser.add_argument('-v', action='store_true', required=False, default=False, help='Mix processed with input')
    parser.add_argument('-e', action='store_true', required=False, default=False, help='Estimate motion')
    parser.add_argument('-r', action='store_true', required=False, default=False, help='Read right image')
    parser.add_argument('-p', type=str, required=False, default=enc_cfg.get('default'),
                        help='Select parameters preset for FFMPEG')
    args = parser.parse_args()
    enc_cfg = enc_cfg[args.p]

    model = None if args.m is None else load_model(args.m, device=device)[0]
    view = sl.VIEW.VIEW_RIGHT if args.r else sl.VIEW.VIEW_LEFT
    svo_path = config['svo_path']

    if isdir(args.input):
        files = sorted([
            join(args.input, fn)
            for fn in listdir(args.input)
            if isfile(join(args.input, fn)) and fn.endswith('.svo')
        ], key=file_idx)
    elif isfile(args.input):
        files = [args.input]
    else:
        try:
            files = decode_name(args.input)
        except ValueError:
            print('No such file or directory: %s' % args.input)
            return

    pause = False
    save_idx = get_save_idx(images_dir, image_fmt)

    writer = None
    out_path = args.output
    if out_path == 'auto':
        model_name = args.m
        if model_name is not None and args.v:
            model_name += '-v'
        out_path = config['mp4_path'].replace('{model}', model_name or '')
        if not isdir(out_path):
            mkdir(out_path)

    to_dir = False if out_path is None else isdir(out_path)
    out_height, out_width = None, None
    estimator = MotionEstimator('fly1', with_gui=out_path is None)
    with torch.no_grad():
        for dn in files:
            if type(dn) == str:
                fn = dn
                out_name = basename(fn)[:-4]
            else:
                fn, out_name, view = dn
                fn = join(svo_path, fn)
            out_name = None if out_path is None else join(out_path, out_name + '.mp4')

            if not isfile(fn):
                print('File %s not found!' % fn)
                continue
            print('Processing %s' % fn)
            for frame_idx, source in enumerate(read_svo(fn, view)):

                # Processing:
                if model is not None:
                    data = image_to_tensor(source, device=device)
                    data = model(data).squeeze(0)  # [:, :2]
                    if args.e:
                        mask = (data[4:6] < 0.5).all(0)
                        estimator.add(source, mask.cpu().numpy(), out_name, frame_idx)

                    result = probs_to_image(data, mask=True)
                    output = visualize(source, result) if args.v else result
                else:
                    result = None
                    output = source

                if out_height is None:
                    out_height, out_width = map(lambda s: s // args.f, output.shape[:2])
                if args.f != 1:
                    output = cv2.resize(output, (out_width, out_height))

                # Open writer:
                if writer is None and out_path is not None:
                    dst = out_name if to_dir else out_path
                    print('Write to %s' % dst)
                    writer = open_ffmpeg(dst, (out_width, out_height), params=enc_cfg)

                # Return output:
                if writer is not None:
                    write_ffmpeg(writer, output)
                else:
                    cv2.imshow('output', output)
                    while True:
                        key = cv2.waitKey(1)
                        if key == ord('p'):
                            pause = not pause
                        elif key == ord('s'):
                            cv2.imwrite(join('images', 'saved-in', image_fmt % save_idx), source)
                            if result is not None:
                                cv2.imwrite(join('images', 'saved-out', image_fmt % save_idx), result)
                            save_idx += 1
                        elif key == ord('q'):
                            return
                        elif key == ord('m'):
                            cv2.imshow('output', result)
                        elif key == ord('i'):
                            cv2.imshow('output', source)
                        if not pause:
                            break
            if to_dir and writer is not None:
                close_ffmpeg(writer)
                writer = None
        if writer is not None:
            close_ffmpeg(writer)


if __name__ == "__main__":
    main()

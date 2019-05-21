import cv2
import torch
import numpy as np
from pyzed import sl
from os.path import join, isfile
from os import listdir
import json
import xml.etree.ElementTree as ET
from utils import channel_names, probs_to_image, visualize, decode_name
from h5py import File


def process_xml(fn, svo_dir='svo', show: bool = True):
    # Parse .xml
    root = ET.parse(fn).getroot()
    frames = {}
    for item in root:
        if item.tag == 'meta':
            task = item.find('task')
            name = task.find('name').text
            fn, view = decode_name(name, svo_dir)
            if not isfile(fn):
                print('File %s not found, skipping')
                return None
            size = task.find('original_size')
            size = [int(size.find(p).text) for p in ['height', 'width']]
        elif item.tag == 'track':
            label = item.attrib['label']
            for part in item:
                att = part.attrib
                if part.tag == 'polygon' and att['outside'] == '0':
                    frame = int(att['frame'])
                    points = att['points']
                    frame_data = frames.get(frame, None)
                    if frame_data is None:
                        frame_data = {}
                        frames[frame] = frame_data
                    channel_data = frame_data.get(label)
                    if channel_data is None:
                        channel_data = []
                        frame_data[label] = channel_data
                    channel_data.append(points)

    # Init .svo reading:
    runtime = sl.RuntimeParameters(enable_depth=False, enable_point_cloud=False)
    mat = sl.Mat()
    init = sl.InitParameters(svo_input_filename=fn, svo_real_time_mode=False)
    cam = sl.Camera()
    status = cam.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit()

    # Prepare result buffers
    frame_numbers = sorted(list(frames.keys()))
    channel_count = len(channel_names)
    result_images = np.empty([len(frame_numbers), 3] + size, dtype=np.uint8)
    result_targets = np.empty([len(frame_numbers), channel_count] + size, dtype=np.uint8)
    frame_count = 0
    frames_list = []
    for frame_num in frame_numbers:

        # Prepare target tensor:
        channels = frames[frame_num]
        frame = np.zeros(size, dtype=np.uint8)

        for label in ['ground', 'water', 'bush', 'person', 'tree', 'build', 'tower', 'car', 'wire', 'copter']:
            polygons = channels.get(label, [])
            points = [np.array([[round(float(d)) for d in c.split(',')] for c in poly.split(';')]) for poly in polygons]
            if label in ['water', 'person']:
                label = 'ground'
            color = channel_names.index(label)
            cv2.fillPoly(frame, pts=points, color=color + 1)
        target = torch.zeros([channel_count + 1] + size, dtype=torch.uint8)
        target.reshape(channel_count + 1, -1)[
            torch.tensor(frame, dtype=torch.int64).flatten(),
            torch.arange(size[0] * size[1])
        ] = 1
        mask = ~target[0]
        target = target[1:]
        rate = mask.sum().item() / mask.numel()
        do_proc = rate > 0.5
        print('Frame %d: fill rate is %.1f%%, %s' % (frame_num, rate * 100, 'processing' if do_proc else 'skipping'))
        if not do_proc:
            continue

        # Get source image:
        cam.set_svo_position(frame_num)
        err = cam.grab(runtime)
        # if err == sl.ERROR_CODE.ERROR_CODE_NOT_A_NEW_FRAME:
        #    break
        assert err == sl.ERROR_CODE.SUCCESS
        cam.retrieve_image(mat, view)
        source_image = mat.get_data()[..., :3]

        result_images[frame_count] = np.moveaxis(source_image, -1, 0)
        result_targets[frame_count] = target.cpu().numpy()
        frame_count += 1
        frames_list.append(frame_num)

        if show:
            image = probs_to_image(target, mask=mask.unsqueeze(0))
            image = visualize(source_image, image)
            cv2.imshow('image', image)
            cv2.waitKey(1)
    return result_images[:frame_count], result_targets[:frame_count], name, frames_list


def main(xml_dir='cvat', result_fn='cvat.h5'):
    with open('config.json', 'r') as file:
        config = json.load(file)
        svo_path = config['svo_path']
        show = config['train'].get('with_gui', False)
    with File(result_fn, 'w') as file:
        for fn in filter(lambda f: f.endswith('.xml') and isfile(join(xml_dir, f)), listdir(xml_dir)):
            fn = join(xml_dir, fn)
            result = process_xml(fn, svo_path, show)
            if result is None:
                continue
            images, targets, name, frames = result
            if images.shape[0] > 0:
                group = file.create_group(name)
                group.attrs['frames'] = frames
                group.attrs['channels'] = channel_names
                group.create_dataset('images', data=images)
                group.create_dataset('targets', data=targets)
                print('Completed %s' % fn)
            else:
                print('Skipped %s' % fn)


if __name__ == "__main__":
    main()

import json
from os import listdir
from os.path import isfile, join
import numpy as np
import pyzed.sl as sl
import cv2
from csv_cut import load, save, Cut
from utils import channels


cuts = {}

idx = 5
current_type = 1
pause = False
position = None
file_names = None
SIZE = 320


def save_img(img):
    folder = 'krita/img'
    i = max([int(fn[:-4]) for fn in listdir(folder) if isfile(join(folder, fn)) and fn.endswith('.png')] + [0]) + 1
    cv2.imwrite(join(folder, '%03d.png' % i), img)


def play(filepath):
    global idx, current_type, pause, position, cuts

    init = sl.InitParameters(svo_input_filename=filepath, svo_real_time_mode=False)
    cam = sl.Camera()
    status = cam.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit()

    runtime = sl.RuntimeParameters()
    mat = sl.Mat()

    key = ''
    next_idx = idx + 1
    # print("  Save the current image:     s")
    # print("  Quit the video reading:     q\n")
    err = None
    while key != ord('q'):  # for 'q' key
        if err is None or not pause:
            err = cam.grab(runtime)
        if err == sl.ERROR_CODE.SUCCESS:
            cam.retrieve_image(mat)
            position = cam.get_svo_position()
            img = mat.get_data()
            img_initial = np.copy(img)
            for cut in cuts.get(file_names[idx].replace('\\', '/'), []):
                if cut.start <= position and (cut.stop is None or cut.stop > position):
                    cv2.rectangle(img, (cut.left, cut.top), (cut.left + SIZE, cut.top + SIZE),
                                  tuple(c * 255 for c in channels[cut.type]), 1)

            cv2.imshow("ZED", img)
            key = cv2.waitKey(1)
            # saving_image(key, mat)
        elif err == sl.ERROR_CODE.ERROR_CODE_NOT_A_NEW_FRAME:
            break
        else:
            key = cv2.waitKey(1)
        if key == ord('p'):
            pause = not pause
        elif key == ord('['):
            next_idx = idx - 1
            break
        elif key == ord(']'):
            break
        elif key == ord('r'):
            next_idx = idx
            break
        elif key == ord('-'):
            cuts[idx] = [cut for cut in cuts.get(idx, []) if cut.type != current_type]
        elif key == ord('s'):
            save(cuts)
        elif key == ord('i'):
            save_img(img_initial)
        elif ord('0') <= key <= ord('9'):
            current_type = key - ord('0')

    cam.close()
    idx = next_idx
    return key != 113


def click(event, x, y, flags, param):
    global cuts, current_type

    fn = file_names[idx].replace('\\', '/')
    if event == cv2.EVENT_LBUTTONDOWN:
        cs = [cut for cut in cuts.get(fn, []) if cut.type != current_type]
        cs.append(Cut(fn, x - SIZE // 2, y - SIZE // 2, position, None, current_type))
        cuts[fn] = cs
    elif event == cv2.EVENT_RBUTTONDOWN:
        cuts[fn] = [
            cut if cut.type != current_type else Cut(cut.file, cut.left, cut.top, cut.start, position, cut.type)
            for cut in cuts.get(fn, [])
        ]


def main(folder='sar'):
    with open('config.json', 'r') as f:
        base_folder = json.load(f).get('svo_path', 'svo')
    global idx, file_names, cuts
    load(cuts)
    file_names = sorted([
        join(folder, fn)
        for fn in listdir(join(base_folder, folder))
        if isfile(join(base_folder, folder, fn)) and fn.endswith('.svo')
    ], key=lambda n: ('-'.join(n.split('-')[:-1]), int(n.split('-')[-1][:-4])))

    cv2.namedWindow("ZED")
    cv2.setMouseCallback("ZED", click)
    idx = 0
    while 0 <= idx < len(file_names):
        print(file_names[idx])
        if not play(join(base_folder, file_names[idx])):
            break
    save(cuts, 'cuts_bak.csv')


if __name__ == "__main__":
    main()

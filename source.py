from collections import namedtuple
from os.path import isfile
import csv
import pyzed.sl as sl
import cv2


Cut = namedtuple('Cut', 'idx left top start stop type')

cuts = {}
colors = [
    (0, 0, 0),
    (0, 255, 0),
    (255, 0, 0),
    (0, 0, 255),
    (128, 128, 128),
    (255, 255, 255)
]

idx = 5
current_type = 1
pause = False
position = None
SIZE = 320


def play(filepath):
    global idx, current_type, pause, position

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
    while key != 113:  # for 'q' key
        if err is None or not pause:
            err = cam.grab(runtime)
        if err == sl.ERROR_CODE.SUCCESS:
            cam.retrieve_image(mat)
            position = cam.get_svo_position()
            img = mat.get_data()
            for cut in cuts.get(idx, []):
                if cut.start <= position and (cut.stop is None or cut.stop > position):
                    cv2.rectangle(img, (cut.left, cut.top), (cut.left + SIZE, cut.top + SIZE), colors[cut.type], 1)

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
            save()
        elif chr(key) in '012345':
            current_type = key - ord('0')

    cam.close()
    idx = next_idx
    return key != 113


def click(event, x, y, flags, param):
    global cuts, current_type

    if event == cv2.EVENT_LBUTTONDOWN:
        cs = [cut for cut in cuts.get(idx, []) if cut.type != current_type]
        cs.append(Cut(idx, x - SIZE // 2, y - SIZE // 2, position, None, current_type))
        cuts[idx] = cs
    elif event == cv2.EVENT_RBUTTONDOWN:
        cuts[idx] = [
            cut if cut.type != current_type else Cut(cut.idx, cut.left, cut.top, cut.start, position, cut.type)
            for cut in cuts.get(idx, [])
        ]


def save(name='cuts.csv'):
    global cuts
    with open(name, 'w') as f:
        w = csv.writer(f)
        w.writerow(Cut._fields)
        for cs in cuts.values():
            for cut in cs:
                w.writerow(list(cut))


def load():
    global cuts
    if isfile('cuts.csv'):
        with open('cuts.csv', 'r') as f:
            reader = csv.reader(f)
            is_title = True
            for row in reader:
                if is_title:
                    is_title = False
                    continue
                cut = Cut(*[None if t == '' else int(t) for t in row])
                cs = cuts.get(cut.idx, None)
                if cs is None:
                    cuts[cut.idx] = [cut]
                else:
                    cs.append(cut)


def main(file_mask='/home/igor/terra/svo/sar/rec2018_07_21-%d.svo'):
    global idx
    load()
    cv2.namedWindow("ZED")
    cv2.setMouseCallback("ZED", click)
    idx = 20
    while isfile(file_mask % idx):
        print(idx)
        if not play(file_mask % idx):
            break
    save('cuts_bak.csv')


if __name__ == "__main__":
    main()

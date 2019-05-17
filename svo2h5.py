import json
from h5py import File
from csv_cut import load
from os.path import join
import pyzed.sl as sl
import numpy as np


def main(size=320):
    runtime = sl.RuntimeParameters()
    mat = sl.Mat()
    with open('config.json', 'r') as f:
        config = json.load(f)
    base_dir = config['svo_path']
    cuts = {}
    load(cuts, base_dir=base_dir)
    cuts = sum(cuts.values(), [])
    types = {cut.type for cut in cuts}
    with File('cuts.h5', mode='w') as file:
        for t in types:
            group = file.create_group('type_%d' % t)
            group.attrs['type'] = t
            for idx, cut in enumerate(filter(lambda c: c.type == t, cuts)):
                init = sl.InitParameters(svo_input_filename=join(base_dir, cut.file), svo_real_time_mode=False)
                cam = sl.Camera()
                status = cam.open(init)
                if status != sl.ERROR_CODE.SUCCESS:
                    print(repr(status))
                    exit()
                cam.set_svo_position(cut.start)
                stop = (cut.stop if cut.stop is not None else cam.get_svo_number_of_frames())
                ds = group.create_dataset('cut_%d' % (idx + 1), shape=(stop - cut.start, 3, size, size), dtype=np.uint8)
                for pos in range(stop - cut.start):
                    err = cam.grab(runtime)
                    assert err == sl.ERROR_CODE.SUCCESS
                    cam.retrieve_image(mat)
                    data = np.moveaxis(mat.get_data()[cut.top:cut.top + size, cut.left:cut.left + size, :3], -1, 0)
                    ds[pos] = data
                cam.close()
                print('Writed cut:', cut)


if __name__ == "__main__":
    main()

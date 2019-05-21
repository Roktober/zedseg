from multiprocessing import Process, Queue
from queue import Empty
import pyzed.sl as sl
import numpy as np
import ffmpeg


def process_svo(q: Queue, fn: str, view=None):
    runtime = sl.RuntimeParameters(enable_depth=False, enable_point_cloud=False)
    mat = sl.Mat()
    init = sl.InitParameters(svo_input_filename=fn, svo_real_time_mode=False)
    cam = sl.Camera()
    status = cam.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit()
    while True:
        err = cam.grab(runtime)
        if err == sl.ERROR_CODE.ERROR_CODE_NOT_A_NEW_FRAME:
            break
        assert err == sl.ERROR_CODE.SUCCESS
        cam.retrieve_image(mat, view)
        data = mat.get_data()[..., :3]  # [200:520, 400:720]
        q.put(data)
    cam.close()


def read_svo(file_name, view=sl.VIEW.VIEW_LEFT):
    q = Queue()
    p = Process(target=process_svo, args=(q, file_name, view))
    p.start()
    while p.is_alive():
        try:
            image = q.get(block=True, timeout=0.1)
            yield image
        except Empty:
            pass
    p.join()


def open_ffmpeg(file_name, size=(1280, 720), mode='w', params={}):
    cmd = params.get('cmd', 'ffmpeg')
    params = {k: v for k, v in params.items() if k != 'cmd'}
    return (
        ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt='bgr24', s='{}x{}'.format(*size))
        # .output(file_name, pix_fmt='yuv420p')
        .output(file_name, **params)
        .overwrite_output()
        .run_async(cmd=cmd, pipe_stdin=True)
    )


def write_ffmpeg(proc, frame):
    proc.stdin.write(
        frame
        .astype(np.uint8)
        .tobytes()
    )


def close_ffmpeg(proc):
    proc.stdin.close()
    proc.wait()

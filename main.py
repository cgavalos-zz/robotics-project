import threading
import numpy as np
from aprilmisc import *
from kin_commands import *
from dobot import *
from block import *
import random

class VideoStream:
    def __init__(self, n, fps):
        self.cap = cv2.VideoCapture(n)
        print('Camera %i opened' % n)
        self.stopped = False
        self.fps = fps
        self.started = False
        self.start()

    def start(self):
        self.thread = threading.Thread(target=self.update)
        self.thread.start()
        while not self.started:
            pass

    def update(self):
        while not self.stopped:
            grabbed, img = self.cap.read()
            if grabbed:
                self.img = img
                self.started = True
                time.sleep(1.0 / self.fps)

    def stop(self):
        self.stopped = True
        self.thread.join()
        self.cap.release()

    def image(self):
        return self.img

def main():
    d = Dobot()
    print('Dobot found')
    d.zero()
    d.zero()
    d.move_zero()
    print('Dobot zeroed')

    # Block info
    block = Block()

    # Centering info
    times_per_second = 2
    fps = 30
    max_time = 5
    max_real_error = 10
    max_real_error_2 = 1
    sheight = d.pos_zero()[2]
    delta_z_closeup = block.dim[2] + 50 - sheight
    K = 0.9

    # Build site information
    q1bo = math.pi / 3

    print('Starting stream')

    fps = 60
    vs = VideoStream(1, fps)
    vs.start()

    print('Stream started')

    img_func = vs.image

    build_properties = (q1bo)
    imaging_properties = (img_func, delta_z_closeup, fps, \
        times_per_second, max_time, max_real_error_2, K)

    f1 = 'coolstruct.txt'
    f2 = 'logcabin.txt'

    res = d.build_struct(f1, block, build_properties,
        imaging_properties)
    if res:
        print('%s constructed...' % f1)
        res = d.build_struct(f2, block, build_properties,
            imaging_properties)

    if res:
        print('%s constructed...' % f2)

    vs.stop()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

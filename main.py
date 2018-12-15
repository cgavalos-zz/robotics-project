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

def center_demo():
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

    fps = 30
    vs = VideoStream(2, fps)
    vs.start()

    print('Stream started')

    img_func = vs.image

    num = 0

    locs, angs = poses_from_struct('logcabin.txt')

    while num < len(locs):
        img, res = get_results(img_func())
        cv2.imshow('img', debugannotate(img, res))
        cv2.waitKey(30)
        if len(res) == 0:
            print('No tags!')
            break

        if len(res) == 1:
            tag = res[0]
        else:
            tag = res[random.randrange(0, len(res) - 1)]

        if tag == None:
            print('No tags!')
            break
        else:
            id = tag.tag_id
            print('Choosing tag #' + str(id))

        loc = locs[num]
        ang = angs[num]

        block_properties = (id, loc, ang, block)
        build_properties = (q1bo)
        imaging_properties = (img_func, delta_z_closeup, fps, \
            times_per_second, max_time, max_real_error_2, K)

        if d.pick_and_place(block_properties, build_properties, imaging_properties):
            num += 1

    if num + 1 == len(locs):
        print('Building complete')
    else:
        print('Building incomplete')

    vs.stop()
    cv2.destroyAllWindows()

def test_model_real():
    d = Dobot()
    d.zero()
    d.move_zero()
    p = d.pos()
    print('Zero pos: ' + str(p / 25.4))
    p[0] = 0
    d.move_delta(-p, 0, 0)
    print('z=0 pos:  ' + str(d.pos() / 25.4))

def jzero_vs_modelzero():
    d = Dobot()
    d.zero()
    print('Joint zero: ' + str(d.pos()))
    d.move_zero()
    print('Model zero: ' + str(d.pos()))

def debug(n):
    d = Dobot()
    d.move_zero()
    aprildebug(n)

def calib():
    d = Dobot()
    d.jmove(0, 0.1, 0.1, 0, 0)
    d.move_zero()

def test_bounds():
    d = Dobot()
    d.zero()
    d.move_zero()

    lx = 200
    hx = 300
    ly = -50
    hy = 50
    lz = 2
    hz = 90

    zpos = np.array([(lx + hx) / 2, (ly + hy) / 2, hz + 10])
    bounds = np.array([[lx, ly, lz], [hx, ly, lz], [hx, hy, lz], [lx, hy, lz],
        [lx, ly, hz], [hx, ly, hz], [hx, hy, hz], [lx, hy, hz]])
    for b in bounds:
        d.move_to(b, 0, 0)
        d.move_to(zpos, 0, 0)

    d.move_zero()

def poses_from_struct(filename):
    locs = []
    angs = []

    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            x, y, z, theta = [float(x) for x in line.split(' ')]
            locs.append(np.array([x, y, z]))
            angs.append(theta)

    locs = np.array(locs)
    angs = np.array(angs)
    return (locs, angs)

if __name__ == '__main__':
    #debug(6)
    center_demo()
    #wait_demo()
    #calib()
    #test_model_real()
    #test_bounds()

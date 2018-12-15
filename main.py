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
    time.sleep(1)
    d.move_zero()
    time.sleep(1)
    print('Dobot zeroed')

    z = 160
    dz = -60

    times_per_second = 2
    fps = 30
    max_time = 5
    max_real_error = 10
    max_real_error_2 = 1
    block = Block()
    sheight = d.pos_zero()[2]
    delta_z_closeup = block.dim[2] + 40 - sheight
    K = 0.9

    n = 10

    #for i in range(n):
    #    cap = cv2.VideoCapture(i)
    #    if cap.isOpened():
    #        print('%i opened!' % i)
    #    cap.release()

    print('Starting stream')

    fps = 30
    vs = VideoStream(5, fps)
    vs.start()

    print('Stream started')

    img_func = vs.image

    num = 0

    while True:
        time.sleep(1)
        img, res = get_results(img_func())
        cv2.imshow('img', debugannotate(img, res))
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

        # Center on april tag

        pe, re = d.tag_pos_error(img, tag, block.tagsize)
        d.camera_move([0, 0, delta_z_closeup], 0, 0)
        time.sleep(1)
        d.camera_move(-re, 0, 0)
        time.sleep(1.5)

        c2 = d.center_apriltag(img_func, fps, times_per_second, max_time, max_real_error_2, id, block.tagsize, K)
        if c2 == None:
            print('Tag not in frame')
            d.move_zero()
            break
        else:
            pe, re = c2
            print('Centered with error: %f mm' % norm(re))
        time.sleep(0.5)

        # Find 'up' and 'right'
        # Calculate theta_0

        img = img_func()
        theta_0 = d.center_block(img, block, id)
        print('theta_0: ' + str(theta_0))

        print('Centered on block')

        time.sleep(1.5)

        delta = d.pos()
        delta[0] = 0
        delta[1] = 0
        delta[2] = -delta[2] + block.dim[2] - 2
        d.move_delta(delta, 0, 1)

        print('Picking up block')

        time.sleep(0.5)

        d_theta = round(theta_0 / math.pi) * math.pi - theta_0

        d.move_delta([0, 0, 100], d_theta, 1)
        print('Moving up')

        time.sleep(2)

        d.move_to([100, 250, block.dim[2] * (num + 1) + 5], d_theta, 1)
        print('Moving to new spot')
        time.sleep(2)
        d.move_delta([0, 0, 0], d_theta, 0)
        time.sleep(1)
        np = d.pos()
        np[2] = 150
        d.move_to(np, 0, 0)
        num += 1
        print('Letting go')
        time.sleep(2)
        d.zero()
        print('Zeroing')
        time.sleep(3)

    vs.stop()
    cv2.destroyAllWindows()

def test_model_real():
    d = Dobot()
    d.zero()
    d.move_zero()
    time.sleep(2)
    p = d.pos()
    print('Zero pos: ' + str(p / 25.4))
    p[0] = 0
    d.move_delta(-p, 0, 0)
    time.sleep(2)
    print('z=0 pos:  ' + str(d.pos() / 25.4))

def jzero_vs_modelzero():
    d = Dobot()
    d.zero()
    time.sleep(2)
    print('Joint zero: ' + str(d.pos()))
    d.move_zero()
    time.sleep(2)
    print('Model zero: ' + str(d.pos()))

def debug(n):
    d = Dobot()
    d.move_zero()
    aprildebug(n)

def wait_demo():
    d = Dobot()
    d.zero()
    time.sleep(3)
    d.move_zero()
    time.sleep(3)

    d.move_to([200, 0, 10], 0, 0)
    time.sleep(3)

    d.move_zero()
    time.sleep(3)

    for i in range(4):
        d.move_to([200, 0, 10], 0, 0)
        d.wait_until_stopped()
        print('stopped move')
        d.move_zero()
        d.wait_until_stopped()
        print('stopped zero')

    print('Loop done')

    time.sleep(1)
    print('Stopped')

def calib():
    d = Dobot()
    d.jmove(0, 0.1, 0.1, 0, 0)
    time.sleep(1)
    d.move_zero()
    time.sleep(1)

if __name__ == '__main__':
    #debug(6)
    center_demo()
    #wait_demo()
    #calib()
    #test_model_real()

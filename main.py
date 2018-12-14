import threading
import numpy as np
from aprilmisc import *
from kin_commands import *
from dobot import *
from block import *
import random

def rot_test():
    d = Dobot()
    d.jmove(0, 0, 0, 0, 1)
    time.sleep(1)

    for rot in range(-20, 20, 5):
        d.jmove(0, 0, 0, rot, 1)
        time.sleep(0.25)

    d.zero()
    time.sleep(1)


def old_center():
    cap = cv2.VideoCapture(1)

    d = Dobot()
    d.move_zero()
    time.sleep(1)

    print('zeroed')

    tagsize = 9.0 / 16.0 * 25.4
    z = 160
    dz = -60

    times_per_second = 2

    total_time = 5.0

    i = 0

    while True:
            ret, img = cap.read()
            if i % (fps / times_per_second) == 0:
                pe, re = d.center_apriltag(img, 129, tagsize, 0.9, along_x=True, along_y=False)
                print(norm(re))
                if norm(re) < 5:
                    print('Centered')
                    break
            #img, res = get_results(img)
            #cv2.waitKey(1000 / fps)
            time.sleep(1000.0 / fps)
            i += 1

    ret, img = cap.read()
    img, res = get_results(img)
    cap.release()
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def center_demo():
    d = Dobot()
    print('Dobot found')
    #cap = cv2.VideoCapture(7)
    #print('Capture found')
    d.zero()
    d.move_zero()
    time.sleep(1)
    print('Dobot zeroed')

    z = 160
    dz = -60

    times_per_second = 2
    fps = 30
    max_time = 5
    max_real_error = 5
    max_real_error_2 = 1
    block = Block()
    sheight = d.pos_zero()[2]
    delta_z_closeup = block.dim[2] + 40 - sheight
    K = 0.5

    class VideoStream:
        def __init__(self, n, fps):
            self.cap = cv2.VideoCapture(n)
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

    n = 10

    for i in range(n):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print('%i opened!' % i)
        cap.release()

    fps = 30
    vs = VideoStream(2, fps)
    vs.start()

    img_func = vs.image

    while True:
        time.sleep(2)
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
        delta[2] = -delta[2] + block.dim[2] - 5
        d.move_delta(delta, 0, 1)

        print('Picking up block')

        time.sleep(0.5)

        d_theta = round(theta_0 / math.pi) * math.pi - theta_0

        d.move_delta([0, 0, 100], d_theta, 1)
        print('Moving up')

        time.sleep(2)

        d.move_to([100, 250, block.dim[2] + 5], d_theta, 1)
        print('Moving to new spot')
        time.sleep(2)
        d.move_delta([0, 0, 0], d_theta, 0)
        time.sleep(1)
        d.move_delta([0, 0, 100], 0, 0)
        print('Letting go')
        time.sleep(1)
        d.move_zero()
        print('Zeroing')

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

if __name__ == '__main__':
    #debug(6)
    center_demo()

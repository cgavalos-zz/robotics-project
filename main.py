import threading
import numpy as np
from aprilmisc import *
from kin_commands import *
from dobot import *

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
            #cv2.imshow('final', annotate_image(img, res))
            #cv2.waitKey(1000 / fps)
            time.sleep(1000.0 / fps)
            i += 1

    ret, img = cap.read()
    img, res = get_results(img)
    cv2.imshow('final', annotate_image(img, res))
    cap.release()
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def center_demo():
    cap = cv2.VideoCapture(2)

    d = Dobot()
    d.zero()
    d.move_zero()
    time.sleep(1)
    print('zeroed')

    tagsize = 9.0 / 16.0 * 25.4
    z = 160
    dz = -60

    times_per_second = 5
    fps = 30
    max_time = 10
    max_real_error = 5
    id = 129
    K = 0.5

    def img_func():
        ret, img = cap.read()
        return img

    print(d.center_apriltag(img_func, fps, times_per_second, max_time, max_real_error, id, tagsize, K))
    time.sleep(1.5)
    d.move_delta([0, 0, -100], 0, 0)
    time.sleep(1.5)
    print(d.center_apriltag(img_func, fps, times_per_second, max_time, max_real_error, id, tagsize, K))
    time.sleep(1.5)
    d.move_delta([0, 0, -50], 0, 0)
    time.sleep(1.5)
    print(d.center_apriltag(img_func, fps, times_per_second, max_time, max_real_error, id, tagsize, K))
    time.sleep(1.5)
    print('done')

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

if __name__ == '__main__':
    #aprildebug(1)
    center_demo()

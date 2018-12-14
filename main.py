from DobotSerialInterface import DobotSerialInterface
import time
import cv2
import threading
import numpy as np
from aprilmisc import *
from kin_commands import *

class Dobot:

    P = np.array([
        [0, 0, 0], [0, 0, 103.1875], [0, 0, 135], [160, 0, 0], [42.8625, 0, -71.4375]])
    qo = np.array([0, -0.049, 0.074, 0])

    def __init__(self, port='/dev/dobot'):
        self.interf = DobotSerialInterface(port)
        self.interf.set_speed()
        self.interf.set_ee_config()
        self.interf.set_playback_config()

    def debug(self):
        print('Joint Angles (deg): ' + str(self.interf.current_status.angles))

    def angles(self):
        return np.array(self.interf.current_status.angles) * math.pi / 180

    def jmove(self, base, rear, front, rot, suction):
        rad = 180 / math.pi
        self.interf.send_angle_suction_command(
            base * rad, rear * rad, front * rad, rot * rad, suction)

    def zero(self):
        self.jmove(0, 0, 0, 0, 0)

    def fwdkin(self, q):
        p01 = Dobot.P[0]
        p12 = Dobot.P[1]
        p23 = Dobot.P[2]
        p34 = Dobot.P[3]
        p4t = Dobot.P[4]
        return p01 + np.matmul(Rz(q[0]), (p12 + np.matmul(Ry(q[1]), (p23 + np.matmul(Ry(q[2]), (p34 + np.matmul(Ry(q[3]), p4t)))))))

    def real_to_model_q(self, qreal):
        base, rear, front, rot = qreal
        q1 = base
        q2 = rear - Dobot.qo[1]
        q3 = front - rear - Dobot.qo[2]
        q4 = -q2 - q3
        return np.array([q1, q2, q3, q4])

    def model_to_real_q(self, qmodel):
        q1, q2, q3, q4 = qmodel
        q1r = q1
        q2r = q2 + Dobot.qo[1]
        q3r = q2r + q3 + Dobot.qo[2]
        return np.array([q1r, q2r, q3r])

    def invkin(self, p0t):

        ex = np.array([1, 0, 0])
        ey = np.array([0, 1, 0])
        ez = np.array([0, 0, 1])

        p12 = Dobot.P[1]
        p23 = Dobot.P[2]
        p34 = Dobot.P[3]
        p4t = Dobot.P[4]

        q1v = -subprob4(ez, ey, p0t, np.dot(ey, p12 + p23 + p34 + p4t))

        q1 = q1v[0]

        K = np.matmul(Rz(-q1), p0t) - p12 - p4t

        q3v = subprob3(ey, -p34, p23, norm(K))

        if abs(q3v[0]) < abs(q3v[1]):
            q3 = q3v[0]
        else:
            q3 = q3v[1]

        q2 = -subprob1(ey, K, p23 + np.matmul(Ry(q3), p34))

        q4 = -q2 - q3

        return [q1, q2, q3, q4]

    def move_to(self, p0t, rot, suction):
        q = self.invkin(p0t)
        qrv = self.model_to_real_q(q)
        self.jmove(qrv[0] , qrv[1], qrv[2], rot, suction)

    def pos(self):
        return self.fwdkin(self.real_to_model_q(self.angles()))

    def pos_zero(self):
        return self.fwdkin([0, 0, 0, 0])

    def move_zero(self):
        self.move_to(self.pos_zero(), 0, 0)

    def move_delta(self, delta, rot, suction):
        qreal = self.angles()
        qmodel = self.real_to_model_q(qreal)
        cpos = self.fwdkin(qmodel)
        npos = cpos + delta
        self.move_to(npos, rot, suction)

    def camera_move(self, cdelta, rot, suction):
        # camera up (y) is into robot
        # camera right (x) is to right of robot
        qreal = self.angles()
        base = qreal[0]
        dx, dy, dz = cdelta
        rdelta = np.array([
            -dy * math.cos(base) - dx * math.sin(base),
            dx * math.cos(base) - dy * math.sin(base),
            dz])
        self.move_delta(rdelta, rot, suction)

    # tagsize in mm
    def tag_pos_error(self, img, res, tagsize):
        c = np.array(res.center)
        imgcenter = np.array([img.shape[1], img.shape[0]]) / 2
        error_pixels = np.append(imgcenter - c, 0)

        dist_between_adj_points = norm(res.corners[0] - res.corners[1])
        mm_per_pixel = tagsize / dist_between_adj_points
        error_real = error_pixels * mm_per_pixel
        error_real[1] = -error_real[1]
        return (error_pixels, error_real)

    # tagsize in mm
    def center_apriltag_once(self, img, id, tagsize=0.5625, K=0.5, along_x=True, along_y=True):
        img, results = get_results(img)
        res = findResultByTagID(results, id)

        if res == None:
            print('Tag not found in image')
            return None
        else:
            error_pixels, error_real = self.tag_pos_error(img, res, tagsize)

            if not along_x:
                error_pixels[0] = 0
                error_real[0] = 0
            if not along_y:
                error_pixels[1] = 0
                error_real[1] = 0
            self.camera_move(-K * error_real, 0, 0)
            return (error_pixels, error_real)

    def center_apriltag(self, img_func, fps, times_per_second, max_time,
        max_real_error, id, tagsize=0.5625, K=0.5, along_x=True, along_y=True):
        tstart = time.time()
        i = 0

        while True:
                if time.time() - tstart > max_time:
                    break
                img = img_func()
                if i % (fps / times_per_second) == 0:
                    err_tup = self.center_apriltag_once(img, id, tagsize, K, along_x, along_y)
                    if err_tup == None:
                        break
                    pe, re = err_tup
                    print(norm(re))
                    if norm(re) < max_real_error:
                        cv2.destroyAllWindows()
                        return (pe, re)
                cv2.imshow('img', img)
                cv2.waitKey(1000 / fps)
                i += 1

        cv2.destroyAllWindows()
        return None

def capshow(n):
    cap = cv2.VideoCapture(n)
    while True:
        ret, img = cap.read()
        img, rs = get_results(img)
        cv2.imshow('anno', annotate_image(img, rs))

        key = cv2.waitKey(1000 / 30)
        if key == 27:
            break

    cv2.destroyAllWindows()
    cap.release()

def acapshow(n):
    thr = threading.Thread(target=capshow, args=[n], kwargs={})
    thr.start()

def fill_closed(closed_img):
    th, im_th = cv2.threshold(closed_img, 220, 255, cv2.THRESH_BINARY);

    # Copy the thresholded image.
    im_floodfill = im_th.copy()

    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = im_th.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)

    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0,0), 255);

    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    # Combine the two images to get the foreground.
    return im_th | im_floodfill_inv

def picture_to_blobs(picture, low, ratio):
    # Get edges using hue? (could be saturation but probs not)
    high = ratio * low
    use_hsv = True
    if use_hsv:
        hsv = cv2.cvtColor(picture, cv2.COLOR_BGR2HSV)
        edges = cv2.Canny(hsv[:, :, 1], low, high)
    else:
        edges = cv2.Canny(picture, low, high)

    # Find contours of edges.
    im2, contours, hierarchy = cv2.findContours(
        edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw em thicc to close holes in curves.
    cv2.drawContours(im2, contours, -1, (255, 0, 0), 10)

    # Fill closed curves for blobs.
    return fill_closed(im2)

def object_centroids(img, lowv, ratio):
    blobs = picture_to_blobs(img, lowv, ratio)
    second_edge = cv2.Canny(blobs, lowv, lowv * ratio)
    im, contours, hierarchy = cv2.findContours(
        second_edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    M = [cv2.moments(contour) for contour in contours]
    fM = filter(lambda m: m['m00'] != 0, M)

    Cx = [int(m['m10'] / m['m00']) for m in fM]
    Cy = [int(m['m01'] / m['m00']) for m in fM]
    return zip(Cx, Cy)

def r2():
    acapshow(1)
    return Dobot(6)

def r3():
    d = Dobot('/dev/dobot')
    d.jmove(0, 0, 0, 0, 1)
    time.sleep(1)

    for rot in range(-20, 20, 5):
        d.jmove(0, 0, 0, rot, 1)
        time.sleep(0.25)

    d.zero()
    time.sleep(1)


def r4():
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

def r5():
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

def r6():
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

def r7():
    d = Dobot()
    d.zero()
    time.sleep(2)
    print('Joint zero: ' + str(d.pos()))
    d.move_zero()
    time.sleep(2)
    print('Model zero: ' + str(d.pos()))

if __name__ == '__main__':
    #aprildebug(1)
    r6()

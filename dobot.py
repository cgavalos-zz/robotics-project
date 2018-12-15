from DobotSerialInterface import DobotSerialInterface
import time
import cv2
import numpy as np
from aprilmisc import *
from kin_commands import *
from block import *

class Dobot:

    # all in mm
    P = np.array([
        [0, 0, 0], [0, 0, 103.1875], [0, 0, 135], [160, 0, 0], [42.8625, 0, -71.4375]])
    qo = np.array([0, -0.049, 0.074, 0])
    tcv = np.array([66, 16.5, 0])
    time_move_margin = 0.5

    def __init__(self, port='/dev/dobot'):
        self.interf = DobotSerialInterface(port)
        self.interf.set_speed()
        self.interf.set_ee_config()
        self.interf.set_playback_config()

    def joint_angles(self):
        return np.array(self.interf.current_status.angles) * math.pi / 180

    def wait_until_stopped(self):
        j = self.joint_angles()
        time.sleep(Dobot.time_move_margin)
        while norm(self.joint_angles() - j) == 0.0:
            j = self.joint_angles()
            time.sleep(Dobot.time_move_margin)
        time.sleep(Dobot.time_move_margin)

    def model_angles(self):
        return self.real_to_model_q(self.joint_angles())

    def jmove(self, base, rear, front, rot, suction):
        rad = 180 / math.pi
        self.interf.send_angle_suction_command(
            base * rad, rear * rad, front * rad, rot * rad, suction)

    def zero(self):
        self.jmove(0, 0, 0, 0, 0)

    def t_to_c(self):
        return np.matmul(Rz(self.model_angles()[0]), Dobot.tcv)

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
        return self.fwdkin(self.real_to_model_q(self.joint_angles()))

    def pos_zero(self):
        return self.fwdkin([0, 0, 0, 0])

    def move_zero(self):
        self.move_to(self.pos_zero(), 0, 0)

    def move_delta(self, delta, rot, suction):
        qreal = self.joint_angles()
        qmodel = self.real_to_model_q(qreal)
        cpos = self.fwdkin(qmodel)
        npos = cpos + delta
        self.move_to(npos, rot, suction)

    def camera_move(self, cdelta, rot, suction):
        # camera up (y) is into robot
        # camera right (x) is to right of robot
        qreal = self.joint_angles()
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
                    if norm(re) < max_real_error:
                        return (pe, re)
                cv2.waitKey(1000 / fps)
                i += 1

        return None

    # Returns block frame in 0, block top position in 0, and theta_0
    def locate_block(self, img, block, id):
        img, res = get_results(img)
        if id < 0:
            r = next(iter(res), None)
        else:
            r = findResultByTagID(res, id)

        if r == None:
            print('Tag not in frame')
            return 0
        else:
            # Find block up and right in camera space
            up_c = r.corners[0] - r.corners[3]
            up_c /= norm(up_c)
            up_c[1] *= -1
            theta_c = math.atan2(up_c[1], up_c[0])
            theta_0 = self.model_angles()[0] + theta_c + math.pi / 2

            up_0 = np.matmul(Rz(theta_0), np.array([1, 0, 0]))
            right_0 = np.matmul(Rz(-math.pi / 2), up_0)

            delta = (block.dim[0] / 2 - block.fulltagsize / 2) * right_0 + \
                (block.dim[1] / 2 - block.fulltagsize / 2) * up_0
            block_top = self.pos() + self.t_to_c() + delta
            block_top[2] = block.dim[2]

            return ((up_0, right_0), block_top, theta_0)

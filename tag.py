import apriltag
import cv2
import time
import numpy as np
from scipy.linalg import expm, norm
import math

def main():
    cap = cv2.VideoCapture(0)
    ret, img = cap.read()
    cap.release()

    img, results = get_results(img)

    tag_by_id = lambda id: next((t for t in results if t.tag_id == id), None)

    tl_id = 296
    tr_id = 297
    bl_id = 320
    br_id = 321

    tl = tag_by_id(tl_id)
    tr = tag_by_id(tr_id)
    bl = tag_by_id(bl_id)
    br = tag_by_id(br_id)

    w = 9.0/16
    w0 = 12.0/16
    m = (w0 - w) / 2

    base = np.array([[0, w, 0], [w, w, 0], [w, 0, 0], [0, 0, 0]])

    tlo = base + [0, w0, 0]
    tro = base + [w0, w0, 0]
    blo = base + [0, 0, 0]
    bro = base + [w0, 0, 0]

    objpoints = np.concatenate(
        (tlo, tro, blo, bro))

    objpoints = objpoints.astype('float32')

    imgpoints = np.concatenate(
        (tl.corners, tr.corners,  bl.corners, br.corners))

    imgpoints = imgpoints.astype('float32')

    size = img.shape[-2::-1]

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        [objpoints], [imgpoints], size, None, None)

    print(mtx)
    print(dist)

    cv2.imshow('annot', annotate_image(img, results))

    h,  w = img.shape[:2]
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

    cv2.imshow('undist', cv2.undistort(img, mtx, dist, None, newcameramtx))

    cv2.waitKey(0)

def hat(k):
    return np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])

def Rz(q):
    return expm(hat([0, 0, 1]) * q)

def Ry(q):
    return expm(hat([0, 1, 0]) * q)

def fkin(p, q):
    p01 = p[0]
    p12 = p[1]
    p23 = p[2]
    p34 = p[3]
    p4t = p[4]
    return p01 + np.matmul(Rz(q[0]), (p12 + np.matmul(Ry(q[1]), (p23 + np.matmul(Ry(q[2]), (p34 + np.matmul(Ry(q[3]), p4t)))))))

def qi(base, rear, front):
    return np.array([base, rear, front - rear, -front])

def qr(q1, q2, q3, q4):
    return np.array([q1, q2, q2 + q3])

def subprob1(k, p1, p2):
    p2 = p2/norm(p2) * norm(p1)
    if norm(p1 - p2) < math.sqrt(np.finfo(float).eps):
        return 0
    else:
        k = k / norm(k)
        pp1 = p1 - np.dot(p1, k) * k
        pp2 = p2 - np.dot(p2, k) * k

        epp1 = pp1 / norm(pp1)
        epp2 = pp2 / norm(pp2)

        return math.atan2(np.dot(k, np.cross(epp1, epp2)), np.dot(epp1, epp2))

def subprob3(k, p1, p2, d):
    pp1 = p1 - k * np.dot(k, p1)
    pp2 = p2 - np.dot(k, p2) * k
    dpsq = d**2 - (np.dot(k, p1 - p2))**2

    nan = float('nan')

    if dpsq < 0:
        return np.array([nan, nan])
    elif dpsq == 0:
        return subprob1(k, pp1 / norm(pp1), pp2 / norm(pp2))
    else:
        bb = (norm(pp1)**2 + norm(pp2)**2-dpsq) / (2 * norm(pp1) * norm(pp2))
        if abs(bb) > 1:
            return np.array([nan, nan])
        else:
            phi = math.acos(bb)
            q0 = subprob1(k, pp1 / norm(pp1), pp2 / norm(pp2))
            return np.array([q0 + phi, q0 - phi])

def subprob4(k, h, p, d):
    d = d / norm(h)
    h = h / norm(h)

    c = d - (np.dot(h, p) + np.dot(h, np.matmul(hat(k), np.matmul(hat(k), p))))
    a = np.dot(h, np.matmul(hat(k), p))
    b = -np.dot(h, np.matmul(hat(k), np.matmul(hat(k), p)))

    phi = math.atan2(b, a)

    nan = float('nan')

    if abs(c / math.sqrt(a**2 + b**2)) > 1:
        return np.array([nan, nan])
    else:
        psi = math.asin(c / math.sqrt(a**2 + b**2))
        return np.array([-phi + psi, -phi - psi + math.pi])

def dobot_invkin(p, p0t):

    ex = np.array([1, 0, 0])
    ey = np.array([0, 1, 0])
    ez = np.array([0, 0, 1])

    p12 = p[1]
    p23 = p[2]
    p34 = p[3]
    p4t = p[4]

    q1v = -subprob4(ez, ey, p0t, np.dot(ey, p12 + p23 + p34 + p4t))

    q1 = q1v[0]

    K = np.matmul(Rz(-q1), p0t) - p12 - p4t

    q3v = subprob3(ey, -p34, p23, norm(K))

    print(q3v)

    if abs(q3v[0]) < abs(q3v[1]):
        q3 = q3v[0]
    else:
        q3 = q3v[1]

    q2 = -subprob1(ey, K, p23 + np.matmul(Ry(q3), p34))

    q4 = -q2 - q3

    return [q1, q2, q3, q4]

def dobot_p():
    return np.array([
        [0, 0, 0], [0, 0, 103], [0, 0, 135], [160, 0, 0], [43, 0, -72]])

if __name__ == '__main__':
    p = dobot_p()

    zconf = np.array([160 + 43, 0, 103 + 135 - 72])

    p0t = zconf + np.array([100, 0, 0])
    q = dobot_invkin(p, p0t)

    print("qr: ", qr(q[0], q[1], q[2], q[3]) * 180 / math.pi)
    print("P0t:", p0t)
    print("Fkin: ", fkin(p, q))
    print("Error: ", fkin(p, q) - p0t)

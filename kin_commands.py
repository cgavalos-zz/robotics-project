import numpy as np
from scipy.linalg import expm, norm
import math

def hat(k):
    return np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])

def Rz(q):
    return expm(hat([0, 0, 1]) * q)

def Ry(q):
    return expm(hat([0, 1, 0]) * q)

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

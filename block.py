import numpy as np

class Block:
    def __init__(self):
        self.tagsize = 9.0/16 * 25.4
        self.fulltagsize = 0.75 * 25.4
        self.dim = np.array([24, 75, 14 + 1])

def poses_from_struct(filename):
    locangs = []

    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            x, y, z, theta = [float(x) for x in line.split(' ')]
            locangs.append([np.array([x, y, z]), theta])

    locangs = sorted(locangs, key=lambda la: la[0][2])
    locs = [la[0] for la in locangs]
    angs = [la[1] for la in locangs]
    return (locs, angs)

import numpy as np

class Block:
    def __init__(self):
        self.tagsize = 9.0/16 * 25.4
        self.fulltagsize = 0.75 * 25.4
        self.dim = np.array([24, 75, 14 + 1])

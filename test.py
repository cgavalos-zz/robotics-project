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

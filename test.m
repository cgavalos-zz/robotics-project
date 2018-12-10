robot = RobotRaconteur.Connect('tcp://localhost:10001/dobotRR/dobotController');

sjc = @(a, b, c, d) robot.setJointPositions(int16(a), ...
    int16(b), ...
    int16(c), ...
    int16(d));
sj = @(v) sjc(v(1), v(2), v(3), v(4));
gj = @() robot.getJointPositions();

m = @(v) sj(gj() + int16(v));
mx = @(x) m([x; 0; 0; 0]);
my = @(y) m([0; y; 0; 0]);
mz = @(z) m([0; 0; z; 0]);


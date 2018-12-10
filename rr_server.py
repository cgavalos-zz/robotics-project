#! /usr/bin/env python
from DobotStatusMessage import DobotStatusMessage
from DobotSerialInterface import DobotSerialInterface
import time
import RobotRaconteur as RR


RRN = RR.RobotRaconteurNode.s


class DobotObject():
	def __init__(self, port):
		self.desired_joint_angles = [0,0,0,0];
		self.new_command = False;
		self.dobot_interface = DobotSerialInterface(port)
		self.dobot_interface.set_speed()
		self.dobot_interface.set_playback_config()

	def setJointPositions(self,q1,q2,q3,q4):
		self.new_command = True;
		self.desired_joint_angles = [q1,q2,q3,q4];

	def getJointPositions(self):
		return self.dobot_interface.current_status.angles

	def loop(self):
		print 'hi'
		while 1:
			time.sleep(.01)
			if self.new_command:
				self.dobot_interface.send_absolute_position(self.desired_joint_angles[0], self.desired_joint_angles[1], self.desired_joint_angles[2], self.desired_joint_angles[3])
				self.new_command = False

def main2():
	interf = DobotSerialInterface('COM6')
	interf.set_speed()
	interf.set_playback_config()

	#with open('dobotRR.robodef', 'r') as f:
	#	service_def = f.read()

	#RRN.RegisterServiceType(service_def)
	#RRN.RegisterService("dobotController", "dobotRR.DobotObject", interf)

	interf.send_absolute_position(300, 0, 0, 10)
	time.sleep(0.5)
	print('Angles: ' + str(interf.current_status.angles) + '\n')

	time.sleep(0.5)

def main():


    port = 10001
    t1 = RR.LocalTransport()
    t1.StartServerAsNodeName("dobotRR")
    RRN.RegisterTransport(t1)

    t2 = RR.TcpTransport()
    t2.EnableNodeAnnounce()
    t2.StartServer(port)
    RRN.RegisterTransport(t2)

    my_dobot = DobotObject('COM6')


    with open('dobotRR.robodef', 'r') as f:
        service_def = f.read()

    RRN.RegisterServiceType(service_def)
    RRN.RegisterService("dobotController", "dobotRR.DobotObject", my_dobot)
    print "Conect string: tcp://localhost:" + str(port) + "/dobotRR/dobotController"
    my_dobot.loop()

    RRN.Shutdown()


if __name__ == '__main__':
    main2()

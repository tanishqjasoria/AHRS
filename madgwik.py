import numpy  as np
import time
from numpy.linalg import norm

delta_t	= 0.032	
lastTime = 0    
totalTime = 0 	# sample frequency in Hz
betaDef		= 0.1		# 2 * proportional gain
beta = 0.1
data_t1 = []
data_t2 = []
data_k1 = []
data_k2 = []
accelerometer = [0.0,0.0,0.0]
gyroscope =[0.0,0.0,0.0]
magnetometer = [0.0,0.0,0.0]
quaternion = []
i = 0
q    = [float(1.0),float(0.0),float(0.0),float(0.0)]      #quaternion of sensor frame relative to auxiliary frame4
acc  = accelerometer
gyro = gyroscope
mag  = magnetometer

def getData(fileName):
	file = open( fileName , 'r')
	for line in file:
		data = line.split(" ")
		m = data.pop()
		m = data.pop()
		m = data.pop()
		m = data.pop()
		data = [float(a[5:]) for a in data]
		data_t1.append(data[:9])
		data_t2.append(data[9:18])
		data_k1.append(data[18:27])
		data_k2.append(data[27:])
	print(data_t1)

def calcEuler(q):
	yaw   =  np.arctan2(2.0 * (q[1] * q[2] + q[0] * q[3]), q[0] * q[0] + q[1] * q[1] - q[2] * q[2] - q[3] * q[3])
	pitch = -1 * np.arcsin(2.0 * (q[1] * q[3] - q[0] * q[2]))
	roll  =  np.arctan2(2.0 * (q[0] * q[1] + q[2] * q[3]), q[0] * q[0] - q[1] * q[1] - q[2] * q[2] + q[3] * q[3])
	pitch =  pitch * 180.0 / np.pi
	yaw   =  yaw * 180.0/ np.pi 
	yaw   =  yaw - 13.8;        # Declination at Danville, California is 13 degrees 48 minutes and 47 seconds on 2014-04-04
	roll  =  roll * 180.0/ np.pi;
	return [yaw, pitch, roll]

def _print(fileName):
	file =open(fileName,'w')
	for data in quaternion:
		data = [ str(a) for a in data]
		file.write(" ".join(data)+'\n')

def MadgwickAHRSupdate(gyro, acc, mag,q):
	#if((mag[1] == 0.0) and (mag[1] == 0.0) and (mag[2] == 0.0)):
	#	MadgwickAHRSupdateIMU(gyro, acc)
	#	return

	#Rate of change of quaternion from gyroscope
	qDot = np.multiply(0.5 ,[-q[1] * gyro[0] - q[2] * gyro[1] - q[3] * gyro[2],
				   q[0] * gyro[0] + q[2] * gyro[2] - q[3] * gyro[1],
	               q[0] * gyro[1] - q[1] * gyro[2] + q[3] * gyro[0],
	               q[0] * gyro[2] + q[1] * gyro[1] - q[2] * gyro[0]])
	recipNorm = invSqrt(np.squeeze(np.sum(np.square(acc))))
	acc = np.multiply(acc , recipNorm) 
	# Normalise magnetometer measurement
	recipNorm = invSqrt(np.squeeze(np.sum(np.square(mag))))
	mag = np.multiply(mag , recipNorm)
	# Auxiliary variables to avoid repeated arithmetic
	_2q0mx = 2.0 * q[0] * mag[0]
	_2q0my = 2.0 * q[0] * mag[1]
	_2q0mz = 2.0 * q[0] * mag[2]
	_2q1mx = 2.0 * q[1] * mag[0]
	_2q0   = 2.0 * q[0]
	_2q1   = 2.0 * q[1]
	_2q2   = 2.0 * q[2]
	_2q3   = 2.0 * q[3]
	_2q0q2 = 2.0 * q[0] * q[2]
	_2q2q3 = 2.0 * q[2] * q[3]
	q0q0   = q[0] * q[0]
	q0q1   = q[0] * q[1]
	q0q2   = q[0] * q[2]
	q0q3   = q[0] * q[3]
	q1q1   = q[1] * q[1]
	q1q2   = q[1] * q[2]
	q1q3   = q[1] * q[3]
	q2q2   = q[2] * q[2]
	q2q3   = q[2] * q[3]
	q3q3   = q[3] * q[3]

	# Reference direction of Earth's magnetic field
	hx = mag[0] * q0q0 - _2q0my * q[3] + _2q0mz * q[2] + mag[0] * q1q1 + _2q1 * mag[1] * q[2] + _2q1 * mag[2] * q[3] - mag[0] * q2q2 - mag[0] * q3q3
	hy = _2q0mx * q[3] + mag[1] * q0q0 - _2q0mz * q[1] + _2q1mx * q[2] - mag[1] * q1q1 + mag[1] * q2q2 + _2q2 * mag[2] * q[3] - mag[1] * q3q3
	_2bx = np.sqrt(hx * hx + hy * hy)
	_2bz = -_2q0mx * q[2] + _2q0my * q[1] + mag[2] * q0q0 + _2q1mx * q[3] - mag[2] * q1q1 + _2q2 * mag[1] * q[3] - mag[2] * q2q2 + mag[2] * q3q3
	_4bx = 2.0 * _2bx
	_4bz = 2.0 * _2bz
	#Gradient decent algorithm corrective step
	s = [-_2q2 * (2.0 * q1q3 - _2q0q2 - acc[0]) + _2q1 * (2.0 * q0q1 + _2q2q3 - acc[1]) - _2bz * q[2] * (_2bx * (0.5 - q2q2 - q3q3) + _2bz * (q1q3 - q0q2) - mag[0]) + (-_2bx * q[3] + _2bz * q[1]) * (_2bx * (q1q2 - q0q3) + _2bz * (q0q1 + q2q3) - mag[1]) + _2bx * q[2] * (_2bx * (q0q2 + q1q3) + _2bz * (0.5 - q1q1 - q2q2) - mag[2]),
			  _2q3 * (2.0 * q1q3 - _2q0q2 - acc[0]) + _2q0 * (2.0 * q0q1 + _2q2q3 - acc[1]) - 4.0 * q[1] * (1 - 2.0 * q1q1 - 2.0 * q2q2 - acc[2]) + _2bz * q[3] * (_2bx * (0.5 - q2q2 - q3q3) + _2bz * (q1q3 - q0q2) - mag[0]) + (_2bx * q[2] + _2bz * q[0]) * (_2bx * (q1q2 - q0q3) + _2bz * (q0q1 + q2q3) - mag[1]) + (_2bx * q[3] - _4bz * q[1]) * (_2bx * (q0q2 + q1q3) + _2bz * (0.5 - q1q1 - q2q2) - mag[2]),
			 -_2q0 * (2.0 * q1q3 - _2q0q2 - acc[0]) + _2q3 * (2.0 * q0q1 + _2q2q3 - acc[1]) - 4.0 * q[2] * (1 - 2.0 * q1q1 - 2.0 * q2q2 - acc[2]) + (-_4bx * q[2] - _2bz * q[0]) * (_2bx * (0.5 - q2q2 - q3q3) + _2bz * (q1q3 - q0q2) - mag[0]) + (_2bx * q[1] + _2bz * q[3]) * (_2bx * (q1q2 - q0q3) + _2bz * (q0q1 + q2q3) - mag[1]) + (_2bx * q[0] - _4bz * q[2]) * (_2bx * (q0q2 + q1q3) + _2bz * (0.5 - q1q1 - q2q2) - mag[2]),
			  _2q1 * (2.0 * q1q3 - _2q0q2 - acc[0]) + _2q2 * (2.0 * q0q1 + _2q2q3 - acc[1]) + (-_4bx * q[3] + _2bz * q[1]) * (_2bx * (0.5 - q2q2 - q3q3) + _2bz * (q1q3 - q0q2) - mag[0]) + (-_2bx * q[0] + _2bz * q[2]) * (_2bx * (q1q2 - q0q3) + _2bz * (q0q1 + q2q3) - mag[1]) + _2bx * q[1] * (_2bx * (q0q2 + q1q3) + _2bz * (0.5 - q1q1 - q2q2) - mag[2])]
	recipNorm = invSqrt(np.squeeze(np.sum(np.square(s)))) # normalise step magnitude
	s = np.multiply(s , recipNorm)

	#Apply feedback step
	qDot = qDot - beta*s
	# Integrate rate of change of quaternion to yield quaternion
	q = q + np.multiply(qDot , delta_t)

	# Normalise quaternion
	recipNorm = invSqrt(np.squeeze(np.sum(np.square(q))))
	q = np.multiply(q , recipNorm)
	print(q)
	return q

def invSqrt(number):
    threehalfs = 1.5
    x2 = number * 0.5
    y = np.float32(number)
    
    i = y.view(np.int32)
    i = np.int32(0x5f3759df) - np.int32(i >> 1)
    y = i.view(np.float32)
    
    y = y * (threehalfs - (x2 * y * y))
    return float(y)

def update_MPU():
	data = data_k1[::-1]
	a = data.pop()
	print(a)
	acc = a[:3]
	gyro = a[3:6]
	mag = a[6:9]
	return acc, gyro, mag

couter = 0
getData("back_final1.txt")
update_MPU()
euler=[]
while i<2000:
	nowTime=time.time()
	delta_t = (nowTime - lastTime) #set integration time by time elapsed since last filter update
	lastTime = nowTime
	totalTime = totalTime + delta_t
	if totalTime >= 0.032:
		totalTime =0
		acc, gyro, mag = update_MPU();
		print(acc)
		print(gyro)
		print(mag)
		i = i+1
		print(i)
		quaternion.append(euler);
	q = MadgwickAHRSupdate(gyro, acc, mag,q)
	euler = calcEuler(q)

_print("quaternion.txt")





           

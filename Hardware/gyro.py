
# What is the GY-951?
# The GY-951 is actually a 9-axis IMU (Inertial Measurement Unit) module that contains:
# Three Sensors in One:
# 1- Gyroscope (3 axes) - Measures angular velocity (rotation speed)
# X, Y, Z rotation rates
# 2- Accelerometer (3 axes) - Measures linear acceleration (and gravity)
# X, Y, Z acceleration
# 3- Magnetometer (3 axes) - Measures magnetic field (like a compass)
# X, Y, Z magnetic field strength


import serial as s

from Hardware import flexsensors

sers = s.Serial(port='/dev/serial0', baudrate=57600, timeout=1)
f = flexsensors()
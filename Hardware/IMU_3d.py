import serial
import time
from vpython import *

try:
    se = serial.Serial('/dev/serial0', 57600, timeout=1)
    se.reset_input_buffer()
    print("Serial is ON")
except Exception as e:
    print("Serial failed", e)
    exit()


scene.title = "IMU XYZ VISUALIZATION"
scene.background = color.black
scene.width = 800
scene.height = 600
scene.range = 5

gy951_body = box(length=4, height=0.5, width=2, color=color.blue)
front_dot = sphere(pos=vector(2,0,0), radius=0.2, color=color.red)
IMU_comp = compound([gy951_body, front_dot])

# Axes
x_arrow = arrow(pos=vector(0,0,0), axis=vector(2,0,0), color=color.red, shaftwidth=0.1)
y_arrow = arrow(pos=vector(0,0,0), axis=vector(0,2,0), color=color.green, shaftwidth=0.1)
z_arrow = arrow(pos=vector(0,0,0), axis=vector(0,0,2), color=color.blue, shaftwidth=0.1)

print("Waiting for data...")

while True:
    rate(50)

    if se.in_waiting > 100:
        se.reset_input_buffer()
        
        continue 
    
    if se.in_waiting > 0:
        try:
            line = se.readline().decode('utf-8', errors='replace').strip()
            
         
            if line.startswith("#YPR="):
                data = line.replace("#YPR=", "")
                data_listed = data.split(',')

                if len(data_listed) == 3:
                    yaw_d   = float(data_listed[0])
                    pitch_d = float(data_listed[1])
                    roll_d  = float(data_listed[2])

                   
                    yaw_r   = radians(yaw_d)
                    pitch_r = radians(pitch_d)
                    roll_r  = radians(roll_d)

                   
                    IMU_comp.axis = vector(1, 0, 0)
                    IMU_comp.up   = vector(0, 1, 0)
                    
                   
                    IMU_comp.rotate(angle=-yaw_r,   axis=vector(0,1,0))
                    IMU_comp.rotate(angle=-pitch_r, axis=vector(0,0,1))
                    IMU_comp.rotate(angle=roll_r,   axis=vector(1,0,0))

        except ValueError:
            pass
        except Exception:
            pass


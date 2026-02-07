from collections import deque
import math
import serial
import time


SerialPort = "/dev/serial0"
BaudRate = 57600

PrevRoll = 0
PrevPitch = 0  
PrevYaw = 0

count = 0
offset_yaw = 0
offset_pitch = 0
offset_roll = 0
first_time = True
previous_buffer = deque(maxlen=25)

def get_corrected_yaw(raw_yaw):
    return (raw_yaw - offset_yaw) % 360 

def get_corrected_pitch(raw_pitch):
    return raw_pitch - offset_pitch

def get_corrected_roll(raw_roll):
    return raw_roll - offset_roll

start = time.time() 
try:
    ser = serial.Serial(SerialPort, BaudRate, timeout=1)
    ser.reset_input_buffer()
    print("Serial Connected!")
except Exception as E:
    print("Error establishing serial connection: ", E)
    exit() 

print("Calibrating... Keep sensor STILL and LEVEL.")

time.sleep(0.1)

while True:
    if ser.in_waiting > 0:
        line = ser.readline().decode('utf-8', errors='replace').strip()
        
        if line.startswith("#YPR="):
            data = line.replace("#YPR=", "").split(',')
            if len(data) == 3:
                offset_yaw = float(data[0])
                offset_pitch = float(data[1])
                offset_roll = float(data[2])
                
                print(f"Calibration Done.. ")
                print(f"  yaw offset: {offset_yaw:.2f} deg ")
                print(f"  pitch offset: {offset_pitch:.2f} deg")
                print(f"  roll offset: {offset_roll:.2f} deg ")
                break

print("Starting Data Stream...")

while time.time() - start < 10:
    if ser.in_waiting > 0:
        try:
            line = ser.readline().decode('utf-8', errors='replace').strip()
            
            if line.startswith("#YPR="):
                data = line.replace("#YPR=", "").split(',')
                if len(data) == 3:
                    count = count + 1
                    RawYaw   = float(data[0])
                    RawPitch = float(data[1])
                    RawRoll  = float(data[2])

        
                    CurrentYaw = get_corrected_yaw(RawYaw)
                    CurrentPitch = get_corrected_pitch(RawPitch)
                    CurrentRoll = get_corrected_roll(RawRoll)

                    if first_time: 
                        DeltaYaw, DeltaPitch, DeltaRoll = 0, 0, 0
                        first_time = False
                    else:
                        DeltaYaw = CurrentYaw - PrevYaw
                        DeltaPitch = CurrentPitch - PrevPitch
                        DeltaRoll = CurrentRoll - PrevRoll

                        #yaw wraparound bc u fking technically can move around yourself in a circle (360)
                        #pitch and roll of your hand wont rotate over 180 unless u break ur wrist ok 
                        if DeltaYaw > 180: 
                            DeltaYaw -= 360
                        if DeltaYaw < -180:
                            DeltaYaw += 360

                   
                    YawRadians = math.radians(CurrentYaw)
                    PitchRadians = math.radians(CurrentPitch)
                    RollRadians = math.radians(CurrentRoll)

                    features = [
                        round(math.sin(YawRadians), 3),
                        round(math.cos(YawRadians), 3),
                        round(math.sin(PitchRadians), 3),
                        round(math.cos(PitchRadians), 3),
                        round(math.sin(RollRadians), 3),
                        round(math.cos(RollRadians), 3),
                        round(DeltaYaw, 2),
                        round(DeltaPitch, 2),
                        round(DeltaRoll, 2)
                    ]

                    PrevYaw = CurrentYaw
                    PrevPitch = CurrentPitch
                    PrevRoll = CurrentRoll

                    previous_buffer.append(features)
             
                    print(f"Features: {features}")

        except ValueError:
            pass 
        except Exception as e:
            print(f"Error: {e}")

print(count)
print(count/10, " hz")
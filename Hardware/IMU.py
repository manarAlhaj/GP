from collections import deque
import math
import serial


SerialPort = "/dev/serial0"
BaudRate = 57600


PrevRoll = 0
PrevPitch = 0  
PrevYaw = 0


offset_yaw = 0 
first_time = True
previous_buffer = deque(maxlen=25)

def get_corrected_yaw(raw_yaw):

    return (raw_yaw - offset_yaw) % 360 


try:
    ser = serial.Serial(SerialPort, BaudRate, timeout=1)
    ser.reset_input_buffer()
    print("Serial Connected!")
except Exception as E:
    print("Error establishing serial connection: ", E)
    exit() 

print("Calibrating... Keep sensor STILL.")

while True:
    if ser.in_waiting > 0:
        
        line = ser.readline().decode('utf-8', errors='replace').strip()
        
        if line.startswith("#YPR="):
            data = line.replace("#YPR=", "").split(',')
            if len(data) == 3:
                
                offset_yaw = float(data[0]) 
                print(f"Calibration Done! North is set to {offset_yaw}°")
                break

print("Starting Data Stream...")

while True:
    if ser.in_waiting > 0:
        try:
            line = ser.readline().decode('utf-8', errors='replace').strip()
            
            if line.startswith("#YPR="):
                data = line.replace("#YPR=", "").split(',')
                if len(data) == 3:
                    RawYaw   = float(data[0])
                    RawPitch = float(data[1])
                    RawRoll  = float(data[2])

                    CurrentYaw = get_corrected_yaw(RawYaw)
                    CurrentPitch = RawPitch
                    CurrentRoll = RawRoll

                 
                    if first_time: 
                        DeltaYaw, DeltaPitch, DeltaRoll = 0, 0, 0
                        first_time = False
                    else:
                        DeltaYaw = CurrentYaw - PrevYaw
                        DeltaPitch = CurrentPitch - PrevPitch
                        DeltaRoll = CurrentRoll - PrevRoll

                        
                        if DeltaYaw > 180: 
                            DeltaYaw -= 360
                        if DeltaYaw < -180:
                            DeltaYaw += 360

                    YawRadians = math.radians(CurrentYaw)
                    YawSine = round(math.sin(YawRadians),3)
                    YawCosine = round(math.cos(YawRadians),3)

                    features = [
                        YawSine, 
                        YawCosine,
                        round(CurrentPitch, 2), 
                        round(CurrentRoll, 2), 
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
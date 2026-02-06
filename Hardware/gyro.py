import serial as s
import time


sers = s.Serial(port='/dev/serial0', baudrate=57600, timeout=1)
sers.reset_input_buffer()

try:
    while True:
        if sers.in_waiting > 0:
         
            line = sers.readline().decode('utf-8').rstrip()
            print(line)
        else:
        
            print("Waiting for data...") 
            time.sleep(1)
            
except KeyboardInterrupt:
    print("Exiting\n")

finally:
    sers.close()

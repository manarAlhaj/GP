

from collections import *
import time
import math 
from serial import Serial
import serial


class IMUc:
    def __init__(self, port="/dev/serial0", baudrate=57600,buffersize=25):

        self.port = port
        self.baudrate = baudrate
        self.buffersize = buffersize

        self.PrevRoll = 0
        self.PrevPitch = 0  
        self.PrevYaw = 0

        self.offset_yaw = 0
        self.offset_pitch = 0
        self.offset_roll = 0

        self.FirstTime = True
        self.isCalib = False
        self.feature_buffer = deque(maxlen=buffersize)


        self.ser = None
        
        self._connect()

    def _connect(self):
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=1)
            self.ser.reset_input_buffer()
            print("Serial Connected!")
        except Exception as E:
            print("Error establishing serial connection: ", E)
            exit()

    def calibrate(self, timeout=5, samples = 50 ):

        print ("Calibrating now, sit still..  ")

        YawSamples = []
        PitchSamples = []
        RollSamples = []


        startTime = time.time()
        while len(YawSamples) < samples  and time.time() - startTime < timeout:
            if self.ser.in_waiting > 0:
                try:
                    line = self.ser.readline().decode('utf-8', errors='replace').strip()
            
                    if line.startswith("#YPR="):
                        data = line.replace("#YPR=", "").split(',')
                        if len(data) == 3:
                            yaw = float(data[0])
                            pitch = float(data[1])
                            roll = float(data[2])

                            YawSamples.append(yaw)
                            PitchSamples.append(pitch)
                            RollSamples.append(roll)

                except Exception as E:
                    print(E, " Timeout.. trying again :p ")
            
        if len(YawSamples) < 10:
            print("failed, need more samples .. ")    

        YawSamples.sort()
        PitchSamples.sort()
        RollSamples.sort()

        #median y3mi 3shan better calibration results appearantly 
        self.offset_yaw = YawSamples[ len (YawSamples) // 2]
        self.offset_pitch = PitchSamples[ len (PitchSamples) // 2]
        self.offset_roll = RollSamples[ len (RollSamples) // 2]

        self.isCalib = True

        print(f"Calibration Done ({len(YawSamples)} samples)")
        print(f"  yaw offset: {self.offset_yaw:.2f} deg")
        print(f"  pitch offset: {self.offset_pitch:.2f} deg")
        print(f"  roll offset: {self.offset_roll:.2f} deg")

        return True


    def _get_corrected_yaw(self,raw_yaw):
        return (raw_yaw - self.offset_yaw) % 360 

    def _get_corrected_pitch(self,raw_pitch):
        return raw_pitch - self.offset_pitch

    def _get_corrected_roll(self,raw_roll):
        return raw_roll - self.offset_roll
    

    def read_raw_YPR(self):
        if self.ser.in_waiting > 0:
            try:
                line = self.ser.readline().decode('utf-8', errors='replace').strip()
            
                if line.startswith("#YPR="):
                    data = line.replace("#YPR=", "").split(',')
                    if len(data) == 3:
                        #count = count + 1
                        RawYaw   = float(data[0])
                        RawPitch = float(data[1])
                        RawRoll  = float(data[2])

                        return RawYaw, RawPitch, RawRoll

            except Exception as E:
                print(E)
        return None, None, None
    


    def read_features(self):

        yaw, pitch, roll = self.read_raw_YPR()

        if yaw == None: # l2no first value bs
            return None
        
        CurrentYaw = self._get_corrected_yaw(yaw)
        CurrentPitch = self._get_corrected_pitch(pitch)
        CurrentRoll = self._get_corrected_roll(roll)

        if self.FirstTime: 
            DeltaYaw, DeltaPitch, DeltaRoll = 0, 0, 0
            self.FirstTime = False
        else:
            DeltaYaw = CurrentYaw - self.PrevYaw
            DeltaPitch = CurrentPitch - self.PrevPitch
            DeltaRoll = CurrentRoll - self.PrevRoll

        
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
        self.PrevYaw = CurrentYaw
        self.PrevPitch = CurrentPitch
        self.PrevRoll = CurrentRoll

        self.feature_buffer.append(features)

        return features
    
    def getBuffer (self):
        return list(self.feature_buffer)
    
    def clear_buffer(self):
        self.feature_buffer.clear()
        self.FirstTime = True

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("connection closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


if __name__ == "__main__":
    try:
        with IMUc(port="/dev/serial0") as imu:
            
            # Calibrate
            if not imu.calibrate(samples=50):
                print("Calibration failed")
                exit()
            
            # Read features for 5 seconds
            print("\nReading features...")
            start = time.time()
            count = 0
            
            while time.time() - start < 5:
                features = imu.read_features()
                
                if features:
                    count += 1
                    print(f"Sample {count}: {features}")
                
                time.sleep(0.01)
            
            print(f"\nCollected {count} samples ({count/5:.1f} Hz)")
    
    except KeyboardInterrupt:
        print("\nStopped")
        



        







    





        







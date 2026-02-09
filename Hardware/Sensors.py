
import time
from flexsensors import FlexSensors
from IMUc import IMUc


def CollectData():

    #step 1 : initialize the classes for both

    print("Initializing sensors .. ")

    flex = FlexSensors()
    imu = IMUc(port="/dev/serial0")
    

    #step 2 : call the calibration function of the flex sensors first , bc flex sensors need MOVEMENT 

    print("\n" + "="*40)
    print("\n Calibrating the flex sensors..")
    input("Press Enter when you are ready ")

    print("="*40)

    flex.calibrate(duration=5)
    


    #step 3 : call the calibration func of the IMU now, BECAUSE U NEED TO SIT STILL WHILE CALIBRATING THE IMU
    print("\n" + "="*40)
    print("Flex calibration is done .. ")
    print("Prepare for IMU calibration...")
    print("Place IMU on flat surface and keep it STILL") 
    input("Press ENTER when you are ready ")
    print( "="*40)

    print("\n" + "="*40)
    print("Calibrating IMU")
    print("="*40)
    if not imu.calibrate(samples=50):
        print("IMU calibration failed .. :( ")
        return
    

    # step 4: allow IMU to be nonblocking ... bc I need it not block reading for the flex sensors
    
    imu.AllowNonBlocking()


    #step 5: Now 
    print("\n" + "="*40)
    print("Reading Data now .. ")
    print("="*40)

    SamplingRate = 50 #I measured it by code it was 48.7 .. 
    duration = 5
    TimeInterval = 1.0 / SamplingRate

    SamplesCollected = []
    StartTime= time.time()

    RecentIMU = None
    while time.time() - StartTime < duration:
        Start = time.time()

        FlexData = flex.read_flex_norm()

        IMUData = imu.read_features()

        if IMUData is not None:
            RecentIMU = IMUData

        if RecentIMU is None:
            continue

        

        samples = FlexData + RecentIMU

        SamplesCollected.append(samples)

        ElapsedTime = time.time() - Start
        SleepTime = max(0, TimeInterval - ElapsedTime) #this is for to avoid negative sleep time 
        #because yk, we can be late a bit than the usual 0.02 time needed , I mean the loop can take longer
        # than that, which what I need to see, maybe then we'll optimize .. this will help avoid crashing. 
        #so either never sleep, (loop is taking more than 0.02)
        #or sleep for the time it took the loop to finish anyways. 


        time.sleep(SleepTime)
        

    print(f"Collected {len(SamplesCollected)} samples ")
    imu.close()
    return SamplesCollected


   






    














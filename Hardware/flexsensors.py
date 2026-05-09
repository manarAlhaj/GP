
import random
import time

#  please note that this must read the values of the flex sensors from the adc .. 
#I have implemented many ways to get or read from the ADC, I will prolly get the values as their normal range, but 
# I was just experimenting.. Any modifications required will be done when I modify the data set for the model training ;) 
# also i just implemented a way to avoid this program to crash on windows. 
class FlexSensors:

    def __init__(self):
        self.simulation = False
        self.sensors = []

        try:
            import board
            import busio
            import adafruit_ads1x15.ads1115 as ADS
            from adafruit_ads1x15.analog_in import AnalogIn # type: ignore

            i2c = busio.I2C(board.SCL, board.SDA)

            ads1 = ADS.ADS1115(i2c, address=0x48)
            ads1.data_rate = 860
            ads1.gain=1

            ads2 = ADS.ADS1115(i2c, address=0x49)
            ads2.data_rate = 860
            ads2.gain=1
            f1 = AnalogIn(ads1, 0)  # ring
            f2 = AnalogIn(ads1, 1)  # middle
            f3 = AnalogIn(ads1, 2)  # pointer
            f4 = AnalogIn(ads1, 3)  # thumb
            f5 = AnalogIn(ads2, 3)  # Pinky

            self.sensors = [f1, f2, f3, f4, f5]
            print("Five flex sensors initialized")

        except Exception as ex:
            print("Flex sensors setup failed:", ex)
            self.simulation = True

    def read(self):

        if self.simulation:
            return [random.randint(10000, 50000) for _ in range(5)]
        for s in self.sensors:
            print(s.value)
            time.sleep(1)


if __name__ == "__main__":
    flex = FlexSensors()
    while True:
        print("Sample read:", flex.read())

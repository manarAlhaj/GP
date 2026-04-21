
import random

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
            import adafruit_ads1x15.ads1115 as ADS
            from adafruit_ads1x15.analog_in import AnalogIn

            i2c = board.I2C()

            ads1 = ADS.ADS1115(i2c, address=0x48)
            ads1.data_rate = 860

            ads2 = ADS.ADS1115(i2c, address=0x49)
            ads2.data_rate = 860

            f1 = AnalogIn(ads1, ADS.P0)  # Thumb
            f2 = AnalogIn(ads1, ADS.P1)  # Pointer
            f3 = AnalogIn(ads1, ADS.P2)  # Middle
            f4 = AnalogIn(ads1, ADS.P3)  # Ring
            f5 = AnalogIn(ads2, ADS.P0)  # Pinky

            self.sensors = [f1, f2, f3, f4, f5]
            print("Five flex sensors initialized")

        except Exception as ex:
            print("Flex sensors setup failed:", ex)
            self.simulation = True

    def read(self):
        """Returns a list of 5 raw ADC values in order: thumb, pointer, middle, ring, pinky."""
        if self.simulation:
            return [random.randint(10000, 50000) for _ in range(5)]
        return [s.value for s in self.sensors]


if __name__ == "__main__":
    flex = FlexSensors()
    print("Sample read:", flex.read())
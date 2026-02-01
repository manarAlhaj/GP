import time
import random

class FlexSensors:

    def __init__(self):
        self.simulation = False
        self.sensors = []
        self.calibrated = False
        self.min_flex = None
        self.max_flex = None

        try:
            import board
            import adafruit_ads1x15.ads1115 as ADS
            from adafruit_ads1x15.analog_in import AnalogIn

            # this initializes the i2c pins (pi-safe)
            i2c = board.I2C()

            ads1 = ADS.ADS1115(i2c, address=0x48)  # GND address, check datasheet
            ads1.data_rate = 860

            ads2 = ADS.ADS1115(i2c, address=0x49)  # VCC address
            ads2.data_rate = 860

            # ummm shsmo we can do 3 on adc1 and 2 on adc2 doesn't matter what mapping we follow
            f1 = AnalogIn(ads1, ADS.P0)  # Thumb
            f2 = AnalogIn(ads1, ADS.P1)  # Pointer
            f3 = AnalogIn(ads1, ADS.P2)  # Middle
            f4 = AnalogIn(ads1, ADS.P3)  # Ring
            f5 = AnalogIn(ads2, ADS.P0)  # Pinky

            self.sensors = [f1, f2, f3, f4, f5]

            print("five flex sensors are initialized")

        except Exception as ex:
            print("flex sensors setup has failed ..", ex)
            self.simulation = True  # debug on windows tamam


    def read_flex(self):
        if self.simulation:
            return [random.randint(10000, 50000) for _ in range(5)]
        else:
            values = []
            for sensor in self.sensors:
                values.append(sensor.value)
            return values


    def read_flex_norm(self):
        raw_vals = self.read_flex()

        # if not calibrated ,just scale raw values for debugging
        if not self.calibrated:
            MAX_ADC = 32767  # ADS1115 signed max
            normalized = []
            for val in raw_vals:
                norm = val / MAX_ADC
                norm = max(0.0, min(1.0, norm))
                normalized.append(norm)
            return normalized

        normalized = []
        for i, val in enumerate(raw_vals):
            range_val = self.max_flex[i] - self.min_flex[i]
            if range_val > 0:
                norm = (val - self.min_flex[i]) / range_val
                norm = max(0.0, min(1.0, norm))
            else:
                norm = 0.0
            normalized.append(norm)

        return normalized


    def read_voltage(self):
        if self.simulation:
            return [random.uniform(0.5, 3) for _ in range(5)]
        else:
            voltages = []
            for v in self.sensors:
                voltages.append(v.voltage)
            return voltages


    def test(self, duration=5):
        print("testing for", duration, "seconds")

        start_time = time.time()

        while (time.time() - start_time) < duration:

            flexValues = self.read_flex()

            print("=" * 20)
            print("FLEX RAW VALUES")
            print("THUMB :", flexValues[0])
            print("POINTER :", flexValues[1])
            print("MIDDLE FINGER :", flexValues[2])
            print("RING FINGER :", flexValues[3])
            print("PINKY :", flexValues[4])
            print("=" * 20)

            time.sleep(0.1)


    def calibrate(self, duration=5):
        print("\n== Calibration Step 1: OPEN HAND ==")
        print("Keep your hand fully open...")
        time.sleep(1)

        mins = [32767] * 5  # [32767, 32767, ...]
        start = time.time()
        while time.time() - start < duration:
            vals = self.read_flex()
            for i in range(5):
                mins[i] = min(mins[i], vals[i])
            time.sleep(0.05)

        print("\n== Calibration Step 2: CLOSED FIST ==")
        print("Make a tight fist...")
        time.sleep(1)

        maxs = [0] * 5
        start = time.time()
        while time.time() - start < duration:
            vals = self.read_flex()
            for i in range(5):
                maxs[i] = max(maxs[i], vals[i])
            time.sleep(0.05)


        # For each finger, if what we thought was the minimum is actually larger than the maximum, swap them.
        for i in range(5):
            if mins[i] > maxs[i]:
                mins[i], maxs[i] = maxs[i], mins[i]

        self.max_flex = maxs
        self.min_flex = mins
        self.calibrated = True

        print("\ncalibration done")
        print("MAX:", maxs)
        print("MIN:", mins)


def main():

    print("\n== Starting Testing on Main ==")

    flex = FlexSensors()

    if flex.simulation:
        print("No hardware detected .. simulating")


    if not flex.simulation:
        print("\n== Running Calibration ==")
        flex.calibrate(duration=5)


    print("\n== Running test 1 .. Flex Raw Vals ==")
    values = flex.read_flex()
    print("Raw values:", values)


    print("\n== Running test 2 .. Flex Norm Vals ==")
    norm_values = flex.read_flex_norm()
    finger_names = ["Thumb", "Pointer", "Middle", "Ring", "Pinky"]
    for i, norm in enumerate(norm_values):
        print(f"{finger_names[i]:10s} : {norm:.6f}")


    print("\n== Running test 3 .. Flex Voltage Vals ==")
    voltages = flex.read_voltage()
    for volt in voltages:
        print("Flex Voltage Value:", volt)

    print("\n== Running test 4 .. Continuous reading ==")
    flex.test(duration=5)


if __name__ == "__main__":
    main()
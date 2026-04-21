import serial
import time
#Reads raw YPR (yaw, pitch, roll) from a GY-951 IMU over UART
#The IMU does sensor fusion internally and streams ->>> #YPR=yaw,pitch,roll lines
class IMUc:

    def __init__(self, port="/dev/serial0", baudrate=57600):
        self.port = port
        self.baudrate = baudrate
        self.ser = None
        self._connect()

    def _connect(self):
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=0)
            self.ser.reset_input_buffer()
            print("Serial connected on", self.port)
        except Exception as e:
            print("Serial connection failed:", e)
            raise

    def read(self):
        #Returns (yaw, pitch, roll) as floats, or (None, None, None) if no fresh line is available.
        #Non blocking. Call this in your sampling loop.

        if self.ser.in_waiting <= 0:
            return None, None, None

        try:
            line = self.ser.readline().decode("utf-8", errors="replace").strip()
            if line.startswith("#YPR="):
                data = line.replace("#YPR=", "").split(",")
                if len(data) == 3:
                    return float(data[0]), float(data[1]), float(data[2])
        except Exception as e:
            print("Read error:", e)

        return None, None, None

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("Serial closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


if __name__ == "__main__":
    
    with IMUc() as imu:
        start = time.time()
        while time.time() - start < 3:
            y, p, r = imu.read()
            if y is not None:
                print(f"Y={y:.2f} P={p:.2f} R={r:.2f}")
            time.sleep(0.01)
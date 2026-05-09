
import asyncio
import struct
from bleak import BleakClient, BleakScanner


SERVICE_UUID = "3643ae00-3da0-4225-9cac-05fd164d9028"
YPR_CHAR_UUID = "4ff69c09-4e25-4c91-a6d8-8ce3c98eac39"
DEVICE_NAME = "ESP32-IMU"


class BLEIMUReceiver:

    def __init__(self):
        self.latest = (None, None, None)
        self._client = None
        self._address = None

    async def connect(self):
        print(f"Scanning for '{DEVICE_NAME}'...")
        device = await BleakScanner.find_device_by_name(DEVICE_NAME, timeout=10)

        if device is None:
            raise RuntimeError(f"Could not find device named '{DEVICE_NAME}'")

        self._address = device.address
        print(f"Found {DEVICE_NAME} at {self._address}")

        self._client = BleakClient(self._address)
        await self._client.connect()
        print("Connected!")

        # subscribe to notifications
        await self._client.start_notify(YPR_CHAR_UUID, self._on_notify)
        print("Subscribed to YPR notifications.")

    def _on_notify(self, sender, data: bytearray):
        if len(data) == 12:
            yaw, pitch, roll = struct.unpack("<fff", data)
            self.latest = (yaw, pitch, roll)

    def read(self):

        val = self.latest
        self.latest = (None, None, None)  
        return val

    async def disconnect(self):
        if self._client and self._client.is_connected:
            await self._client.stop_notify(YPR_CHAR_UUID)
            await self._client.disconnect()
            print("Disconnected.")

    def close(self):

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self.disconnect())
            else:
                loop.run_until_complete(self.disconnect())
        except Exception:
            pass


async def main():
    receiver = BLEIMUReceiver()
    await receiver.connect()

    print("Reading for 5 seconds...")
    start = asyncio.get_event_loop().time()

    try:
        while asyncio.get_event_loop().time() - start < 5:
            y, p, r = receiver.read()
            if y is not None:
                print(f"Y={y:.2f}  P={p:.2f}  R={r:.2f}")
            await asyncio.sleep(0.02) 
    finally:
        await receiver.disconnect()


if __name__ == "__main__":
    asyncio.run(main())

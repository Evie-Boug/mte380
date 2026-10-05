import threading
import time
import serial

PORT = "/dev/ttyACM0"
BAUD = 115200

class SensorReader:
    def __init__(self, port=PORT, baud=BAUD):
        self.values = [0]*8
        self.stamp = 0.0
        self.count = 0
        self._new = threading.Condition()
        self.ser = serial.Serial(port, baud, timeout=1)
        time.sleep(2)
        self.ser.reset_input_buffer()
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()
    def _loop(self):
        while self.running:
            try:
                line = self.ser.readline().decode(errors="ignore").strip()
            except (serial.SerialException, TypeError):
                break
            if line.startswith("S,"):
                try:
                    vals = [int(v) for v in line.split(",")[1:]]
                except ValueError:
                    continue
                if len(vals) == 8:
                    with self._new:
                        self.values = vals
                        self.stamp = time.monotonic()
                        self.count += 1
                        self._new.notify_all()

    # waits for a new sample, returns (values, time, count) or None if it times out
    def wait_new(self, last_count, timeout=0.5):
        with self._new:
            if not self._new.wait_for(lambda: self.count != last_count, timeout):
                return None
            return list(self.values), self.stamp, self.count

    def close(self):
        self.running = False
        self.ser.close()

if __name__ == "__main__":
        s = SensorReader()
        try:
            while True:
                print(s.values)
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        finally:
            s.close()
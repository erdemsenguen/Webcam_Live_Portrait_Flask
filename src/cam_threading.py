import cv2
import threading
import logging


class WebcamStream:
    def __init__(self, src=0, width=1280, height=720):
        self.logger = logging.getLogger(__name__)
        self.cap = cv2.VideoCapture(src)

        if not self.cap.isOpened():
            self.logger.error(f"Unable to open video source {src}")
            raise RuntimeError(f"Unable to open video source {src}")

        # Set resolution
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        # Read first frame
        self.ret, self.frame = self.cap.read()
        if not self.ret or self.frame is None:
            self.logger.error("Unable to read from the camera.")
            raise RuntimeError("Unable to read from the camera.")

        self.running = True
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            with self.lock:
                self.ret = ret
                self.frame = frame

    def read(self):
        with self.lock:
            return self.ret, self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.running = False
        self.thread.join()
        self.cap.release()
        self.logger.info("🛑 WebcamStream stopped and camera released.")

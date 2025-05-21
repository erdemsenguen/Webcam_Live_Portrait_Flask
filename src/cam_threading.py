import cv2
import threading
import logging
class WebcamStream:
    def __init__(self):
        self.logger=logging.getLogger(__name__)
        self.running = True
        self.thread=threading.Thread(target=self.update)
        self.thread.start()
    def update(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.stop()
            self.logger.error(f"Error: Unable to open video source {0}")
        self.ret, self.frame = self.cap.read()
        if not self.ret or self.frame is None:
            self.stop()
            self.logger.error("Error: Unable to read from the camera.")
        while self.running:
            self.ret, self.frame = self.cap.read()
    def read(self):
        return self.ret, self.frame
    def set(self,prop_id,value):
        self.cap.set(prop_id, value)
    def stop(self):
        self.running = False
        self.thread.join()
        self.cap.release()
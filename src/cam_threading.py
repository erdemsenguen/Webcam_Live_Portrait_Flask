import cv2
import threading
import logging
class WebcamStream:
    def __init__(self, src=0):
        self.logger=logging.getLogger(__name__)
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            self.logger.error(f"Error: Unable to open video source {src}")
        self.ret, self.frame = self.cap.read()
        if not self.ret or self.frame is None:
            self.logger.error("Error: Unable to read from the camera.")
        else:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,800)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT,450)
        self.running = True
        self.thread=threading.Thread(target=self.update)
        self.thread.start()
    def update(self):
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
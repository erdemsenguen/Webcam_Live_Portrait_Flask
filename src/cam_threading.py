import cv2
import threading
class WebcamStream:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.ret, self.frame = self.cap.read()
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
import cv2
import pyvirtualcam
import numpy as np
import time
import logging
class FrameProcessor():
    def __init__(self):
        self.gpu_img=cv2.cuda_GpuMat()
        self.logger=logging.getLogger(__name__)
    def operate(self,
                frame,
                flip:bool=False,
                color:bool=False,
                send_to_cam:bool=False,
                cam:pyvirtualcam.Camera=None,
                width:int=None,
                height:int=None)->None:
            if frame is None:
                return None
            if isinstance(frame, np.ndarray):
                if frame.size == 0:
                    self.logger.warning("Empty frame, skipping upload.")
                    return None
                self.gpu_img.upload(frame)
            elif isinstance(frame, cv2.cuda_GpuMat):
                self.gpu_img = frame
            if height and width:
                h, w = self.gpu_img.size()
                if w != width or h != height:
                    self.gpu_img = self.resize(self.gpu_img, width, height)
            if color:
                self.gpu_img=self.bgr_to_rgb(self.gpu_img)
            if flip:
                self.gpu_img=self.flip_img(self.gpu_img)
            before_download=time.time()
            img=self.gpu_img.download()
            if send_to_cam:           
                cam.send(img)
            return img
    def resize(self,img,width:int,height:int):
        resized_gpu=cv2.cuda.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
        return resized_gpu
    def bgr_to_rgb(self,img):
        recolor_gpu=cv2.cuda.cvtColor(img, cv2.COLOR_BGR2RGB)
        return recolor_gpu
    def flip_img(self,img):
        flipped_gpu=cv2.cuda.flip(img,flipCode=1)
        return flipped_gpu
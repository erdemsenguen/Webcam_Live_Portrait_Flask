import cv2
import pyvirtualcam
import numpy as np
import time
import logging


class FrameProcessor:
    def __init__(self):

        self.logger = logging.getLogger(__name__)
        self.has_cuda = cv2.cuda.getCudaEnabledDeviceCount()
    def operate(
        self,
        frame,
        flip: bool = False,
        color: bool = False,
        send_to_cam: bool = False,
        cam: pyvirtualcam.Camera = None,
        width: int = None,
        height: int = None,
    ) -> None:
        if frame is None:
            return None
        if self.has_cuda>0:
            gpu_img = cv2.cuda_GpuMat()
            gpu_img.upload(frame)
        else:
            gpu_img = frame
        if height and width:
            if self.has_cuda>0:
                h, w = gpu_img.size()
            else:
                h, w = gpu_img.shape[:2]
            if w != width or h != height:
                gpu_img = self.resize(gpu_img, width, height)
        if color:
            gpu_img = self.bgr_to_rgb(gpu_img)
        if flip:
            gpu_img = self.flip_img(gpu_img)
        if self.has_cuda>0:
            img = gpu_img.download()
        else:
            img = gpu_img
        if send_to_cam:
            cam.send(img)
        return img

    def resize(self, img, width: int, height: int):
        if self.has_cuda>0:
            resized_gpu = cv2.cuda.resize(
                img, (width, height), interpolation=cv2.INTER_LINEAR
            )
        else:
            resized_gpu = cv2.resize(img,(width,height),interpolation=cv2.INTER_LINEAR)
        return resized_gpu

    def bgr_to_rgb(self, img):
        if self.has_cuda>0:
            recolor_gpu = cv2.cuda.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            recolor_gpu = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        return recolor_gpu
    def flip_img(self, img):
        if self.has_cuda>0:
            flipped_gpu = cv2.cuda.flip(img, flipCode=1)
        else:
            flipped_gpu= cv2.flip(img,flipCode=1)
        return flipped_gpu
        
import cv2
import pyvirtualcam
import numpy as np
import time
import logging


class FrameProcessor:
    def __init__(self):

        self.logger = logging.getLogger(__name__)

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
        gpu_img = cv2.cuda_GpuMat()
        if isinstance(frame, np.ndarray):
            if frame.size == 0:
                self.logger.warning("Empty frame, skipping upload.")
                return None
            gpu_img.upload(frame)
        elif isinstance(frame, cv2.cuda_GpuMat):
            gpu_img = frame
        if height and width:
            h, w = gpu_img.size()
            if w != width or h != height:
                gpu_img = self.resize(gpu_img, width, height)
        if color:
            gpu_img = self.bgr_to_rgb(gpu_img)
        if flip:
            gpu_img = self.flip_img(gpu_img)
        img = gpu_img.download()
        if send_to_cam:
            cam.send(img)
        return img

    def resize(self, img, width: int, height: int):
        resized_gpu = cv2.cuda.resize(
            img, (width, height), interpolation=cv2.INTER_LINEAR
        )
        return resized_gpu

    def bgr_to_rgb(self, img):
        recolor_gpu = cv2.cuda.cvtColor(img, cv2.COLOR_BGR2RGB)
        return recolor_gpu

    def flip_img(self, img):
        flipped_gpu = cv2.cuda.flip(img, flipCode=1)
        return flipped_gpu

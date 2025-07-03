import cv2
import pyvirtualcam
import numpy as np
import time
import logging


class FrameProcessor:
    def __init__(self):

        self.logger = logging.getLogger(__name__)
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                self.use_cuda = True
                self.logger.info("CUDA-enabled OpenCV detected.")
        else:
                self.use_cuda = False
                self.logger.warning("CUDA not available. Falling back to CPU.")

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
        if self.use_cuda:
            return self.gpu(
                frame,
                flip=flip,
                color=color,
                send_to_cam=send_to_cam,
                cam=cam,
                width=width,
                height=height
            )
        else:
            return self.cpu(
                frame,
                flip=flip,
                color=color,
                send_to_cam=send_to_cam,
                cam=cam,
                width=width,
                height=height
            )
    def gpu(self,
        frame,
        flip: bool = False,
        color: bool = False,
        send_to_cam: bool = False,
        cam: pyvirtualcam.Camera = None,
        width: int = None,
        height: int = None):
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
    def cpu(self,
        frame,
        flip: bool = False,
        color: bool = False,
        send_to_cam: bool = False,
        cam: pyvirtualcam.Camera = None,
        width: int = None,
        height: int = None):
        if frame is None:
            return None
        if isinstance(frame, np.ndarray):
            if frame.size == 0:
                self.logger.warning("Empty frame, skipping upload.")
                return None
        if height and width:
            h, w = frame.shape[:2]
            if w != width or h != height:
                frame = self.resize(frame, width, height)
        if color:
            frame = self.bgr_to_rgb(frame)
        if flip:
            frame = self.flip_img(frame)
        if send_to_cam:
            cam.send(frame)
        return frame
    def resize(self, img, width: int, height: int):
        if self.use_cuda:
            resized_gpu = cv2.cuda.resize(
                img, (width, height), interpolation=cv2.INTER_LINEAR
            )
        else:
            resized_gpu = cv2.resize(
                img, (width, height), interpolation=cv2.INTER_LINEAR
            )
        return resized_gpu
    
    def bgr_to_rgb(self, img):
        if self.use_cuda:
            recolor_gpu = cv2.cuda.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            recolor_gpu = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return recolor_gpu

    def flip_img(self, img):
        if self.use_cuda:
            flipped_gpu = cv2.cuda.flip(img, flipCode=1)
        else:
            flipped_gpu = cv2.flip(img, flipCode=1)    
        return flipped_gpu

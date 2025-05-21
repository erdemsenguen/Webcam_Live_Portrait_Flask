import cv2
import pyvirtualcam
import numpy as np
import time
def operate(frame,flip:bool=False,color:bool=False,send_to_cam:bool=False,cam:pyvirtualcam.Camera=None,width:int=None,height:int=None)->None:
        if frame is None:
            return None
        if isinstance(frame, np.ndarray):
            gpu_img = cv2.cuda_GpuMat()
            gpu_img.upload(frame)
        elif isinstance(frame, cv2.cuda_GpuMat):
            gpu_img = frame
        if height and width:
            if frame.shape[1] != width or frame.shape[0] != height:
                gpu_img = resize(gpu_img, width, height)
        if color:
            gpu_img=bgr_to_rgb(gpu_img)
        if flip:
            gpu_img=flip_img(gpu_img)
        before_download=time.time()
        img=gpu_img.download()
        print(f"Download took {time.time()-before_download}")
        if send_to_cam:           
            cam.send(img)
        return img
def resize(img,width:int,height:int):
     resized_gpu=cv2.cuda.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
     return resized_gpu
def bgr_to_rgb(img):
     recolor_gpu=cv2.cuda.cvtColor(img, cv2.COLOR_BGR2RGB)
     return recolor_gpu
def flip_img(img):
     flipped_gpu=cv2.cuda.flip(img,flipCode=1)
     return flipped_gpu
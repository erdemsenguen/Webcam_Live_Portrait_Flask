import tyro
from .src.config.argument_config import ArgumentConfig
from .src.config.inference_config import InferenceConfig
from .src.config.crop_config import CropConfig
from .src.live_portrait_pipeline import LivePortraitPipeline
from .src.crop import face_detector
from .src.utils.io import load_image_rgb
import cv2
import logging
import numpy as np
import platform
import pyvirtualcam
import os
import sys
if platform.system() == "Windows":
    from pygrabber.dshow_graph import FilterGraph
class Inference:
    def __init__(self):
        tyro.extras.set_accent_color("bright_cyan")
        self.args = tyro.cli(ArgumentConfig)
        self.logger=logging.getLogger(__name__)
        self.is_face= None
        self.first_iter=True
        # specify configs for inference
        self.inference_cfg = self.partial_fields(InferenceConfig, self.args.__dict__)
        self.crop_cfg = self.partial_fields(CropConfig, self.args.__dict__)
        self.source_image_path=None
        self.stop_signal=False
        self.live_portrait_pipeline = LivePortraitPipeline(
            inference_cfg=self.inference_cfg,
            crop_cfg=self.crop_cfg
        )
        # Get the directory of the current script
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.running=False
        self.active=False
        # Build the full path to the target file (e.g., a PNG inside a subfolder)
        frame_path = os.path.join(self.script_dir, 'assets', 'frame.png')
        self.overlay=cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)
        # Initialize webcam 'assets/examples/driving/d6.mp4'
        self.backend=None
        self.log_counter_face_start=0
        self.log_counter_face_success=0
        self.log_counter_cam_dupe=0
        self.log_counter_cam_dupe_success=0
        self.log_counter_face_not_found=0
        self.x_s=None
        self.f_s=None
        self.R_s=None
        self.x_s_info=None
        self.lip_delta_before_animation=None
        self.crop_info=None
        self.img_rgb=None
        self.conf_virt_live_webcam()
    def partial_fields(self,target_class, kwargs):
        return target_class(**{k: v for k, v in kwargs.items() if hasattr(target_class, k)})

    def main(self):
        with pyvirtualcam.Camera(width=1920, height=1080, fps=30, backend='v4l2loopback', device='/dev/video10') as cam, \
             pyvirtualcam.Camera(width=1920, height=1080, fps=30, backend='v4l2loopback', device='/dev/video11') as cam2:
            black_image = np.zeros((1080, 1920, 3), dtype=np.uint8)
            while True:
                if not self.running:
                    cam.send(black_image)
                    cam2.send(black_image)
                    continue
                else:
                    black_image=None
                    break
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,1920)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT,1080)
                # Process the first frame to initialize
            ret, frame = cap.read()
            if not ret:
                self.logger.debug("No camera input found.")
                return
            while True:    
                ret, frame = cap.read()
                if not ret:
                    break
                frame_fhd= cv2.resize(frame,(1920,1080))
                frame_fhd = cv2.cvtColor(frame_fhd, cv2.COLOR_BGR2RGB)
                cam2.send(frame_fhd)
                is_face = face_detector(frame)
                if self.first_iter and self.source_image_path:
                    self.logger.debug("DeepFake source image is set!")
                    self.x_s, self.f_s, self.R_s, self.x_s_info, self.lip_delta_before_animation, self.crop_info, self.img_rgb = self.live_portrait_pipeline.execute_frame(frame, self.source_image_path)
                    self.first_iter=False
                # Process the frame
                if is_face: 
                    if self.log_counter_face_start==0:
                            self.logger.debug("Face found.")
                            self.log_counter_face_start+=1
                            self.log_counter_face_not_found=0
                    if self.source_image_path:
                        self.manipulation(cam=cam,frame=frame)
                    else:
                        self.log_counter_face_success=0
                        self.no_manipulation(cam=cam,frame=frame_fhd)
                else:
                    self.no_manipulation(cam=cam,frame=frame_fhd)
                    if self.log_counter_face_not_found==0:
                        self.logger.debug("Face not found.")
                        self.log_counter_face_not_found+=1
                    self.log_counter_face_start=0
            cap.release()        
        # live_portrait_pipeline.execute_frame(result_bgr)
    def manipulation(self,cam,frame):
        self.log_counter_cam_dupe=0
        self.log_counter_cam_dupe_success=0
        self.active=True
        pad=np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = self.live_portrait_pipeline.generate_frame(self.x_s, self.f_s, self.R_s, self.x_s_info, self.lip_delta_before_animation, self.crop_info, self.img_rgb, frame)
        result_height,result_width=result.shape[:2]
        if result_height>1080:
            result=cv2.resize(result,(int(result_width*1080/result_height),1080))
        result_height,result_width=result.shape[:2]
        if result_width>1920:
            result=cv2.resize(result,(1920,int(result_height*1920/result_width)))
        result_height,result_width=result.shape[:2]
        x_offset=(1920-result_width)//2
        y_offset=(1080-result_height)//2
        pad[y_offset:y_offset+result_height,x_offset:x_offset+result_width]=result
        if self.log_counter_face_success==0:
            self.logger.debug("Face control established.")
            self.log_counter_face_success+=1
        cam.send(pad)
    def no_manipulation(self,cam,frame):
        self.x_s, self.f_s, self.R_s, self.x_s_info, self.lip_delta_before_animation, self.crop_info, self.img_rgb = None, None, None, None, None, None, None
        if self.log_counter_cam_dupe==0:
            self.log_counter_cam_dupe+=1
        self.active=False
        self.source_image_path=None
        self.first_iter=True
        overlay_resized = cv2.resize(self.overlay, (frame.shape[1], frame.shape[0]))
        overlay_rgb = overlay_resized[..., :3]
        try:
            alpha_mask = overlay_resized[..., 3:]/255
        except Exception as e:
            alpha_mask= np.full((1080,1920),0.4)
            self.logger.error(e) 
        blended = (1.0 - alpha_mask) * frame + alpha_mask * overlay_rgb
        blended = blended.astype(np.uint8)
        blended = cv2.resize(blended,(1920,1080))
        if self.log_counter_cam_dupe_success==0:
            self.logger.debug("Duplicated camera feed is succesful.")
            self.log_counter_cam_dupe_success+=1
        cam.send(blended)
    def conf_virt_live_webcam(self):
        if platform.system() == "Windows":
            self.backend = "obs"
        elif platform.system() == "Linux":
            self.backend = "v4l2loopback"
        else:
            self.backend = "unknown"  # or raise an error
    def set_source(self,source_img_path:str):
        self.first_iter=True
        try:
            load_image_rgb(source_img_path)
            self.source_image_path=source_img_path
            self.logger.debug("Image set successfully!")
            return "Image set successfully."
        except Exception as e:
            self.source_image_path=None
            return e
    def status_funct(self):
        return(self.active)
    def set_run(self):
        self.running=True
    def stop(self):
        self.stop_signal=True
if __name__ == '__main__':
    logging.basicConfig(
    level=logging.DEBUG,  # or INFO
    format='[%(asctime)s] %(levelname)s in %(name)s: %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
    )
    inf=Inference()
    inf.main()

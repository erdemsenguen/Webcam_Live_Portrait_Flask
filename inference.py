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
import mediapipe as mp
import platform
import pyvirtualcam
import os
import sys
import typing
import random
import onnxruntime as ort
import time
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
        self.background_images=[f"{os.path.dirname(self.script_dir)}/Backgrounds/wall.jpg",
                                f"{os.path.dirname(self.script_dir)}/Backgrounds/bookshelf.jpg",
                                f"{os.path.dirname(self.script_dir)}/Backgrounds/inIT_Hindergrund.jpg"]
        self.green_screens=[f"{os.path.dirname(self.script_dir)}/Backgrounds/meeting_green1.jpg",
                            f"{os.path.dirname(self.script_dir)}/Backgrounds/meeting_green2.jpg"]
        self.running=False
        self.active=False
        frame_path = os.path.join(self.script_dir, 'assets', 'frame.png')
        self.overlay=cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)
        self.overlay=cv2.resize(self.overlay,(960,540))
        self.overlay_rgb = self.overlay[..., :3]
        try:
            self.alpha_mask = self.overlay[..., 3:]/255
        except Exception as e:
            self.alpha_mask= np.full((540,960),0.4)
            self.logger.error(e) 
        self.background_image=None
        self.background_image_path=None
        self.green_screen=None
        self.previous_green_screen=None
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
        self.session=ort.InferenceSession(f"{self.script_dir}/pretrained_weights/u2-segmentation/u2netp.onnx",
                                          providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.pad=np.zeros((540, 960, 3), dtype=np.uint8)
        self.conf_virt_live_webcam()
    def partial_fields(self,target_class, kwargs):
        return target_class(**{k: v for k, v in kwargs.items() if hasattr(target_class, k)})

    def main(self):
        with pyvirtualcam.Camera(width=960, height=540, fps=30, backend='v4l2loopback', device='/dev/video10') as cam, \
             pyvirtualcam.Camera(width=960, height=540, fps=30, backend='v4l2loopback', device='/dev/video11') as cam2:
            black_image = np.zeros((540, 960, 3), dtype=np.uint8)
            while True:
                if not self.running:
                    cam.send(black_image)
                    cam2.send(black_image)
                    continue
                else:
                    black_image=None
                    break
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT,760)
            ret, frame = cap.read()
            if not ret:
                self.logger.debug("No camera input found.")
                return
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame=cv2.resize(frame,(960,540))
                frame=cv2.flip(frame, 1)
                frame_clr = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cam2.send(frame_clr)
                is_face = face_detector(frame)
                if self.first_iter and self.source_image_path:
                    self.logger.debug("DeepFake source image is set!")
                    self.x_s, self.f_s, self.R_s, self.x_s_info, self.lip_delta_before_animation, self.crop_info, self.img_rgb = self.live_portrait_pipeline.execute_frame(frame, self.source_image_path)
                    self.first_iter=False
                if is_face: 
                    if self.log_counter_face_start==0:
                            self.logger.debug("Face found.")
                            self.log_counter_face_start+=1
                            self.log_counter_face_not_found=0
                    if self.source_image_path:
                        self.manipulation(cam=cam,frame=frame)
                    else:
                        self.log_counter_face_success=0
                        self.no_manipulation(cam=cam,frame=frame_clr)
                else:
                    self.no_manipulation(cam=cam,frame=frame_clr)
                    if self.log_counter_face_not_found==0:
                        self.logger.debug("Face not found.")
                        self.log_counter_face_not_found+=1
                    self.log_counter_face_start=0
            cap.release()        
    def manipulation(self,cam,frame):
        self.logger.debug("Manipulation starts!")
        mani=time.time()
        self.log_counter_cam_dupe=0
        self.log_counter_cam_dupe_success=0
        self.active=True
        result = self.live_portrait_pipeline.generate_frame(self.x_s, self.f_s, self.R_s, self.x_s_info, self.lip_delta_before_animation, self.crop_info, self.img_rgb, frame)
        result_height,result_width=result.shape[:2]
        x_offset=(960-result_width)//2
        y_offset=540-result_height
        pad=self.pad.copy()
        pad[y_offset:y_offset+result_height,x_offset:x_offset+result_width]=result
        if self.background_image_path:
            self.logger.debug("Background starts!")
            background_time=time.time()
            background=self.background_image
            bg_image_resize=cv2.cvtColor(background,cv2.COLOR_BGR2RGB)
            out=self.background_blur(pad,bg_image_resize)
            self.logger.debug(f"Background took {time.time()-background_time} seconds!")
            if self.green_screen:
                self.logger.debug(f"Monitor overlay starts!")
                moni=time.time()
                out=self.overlay_on_monitor(self.green_img,out)
                self.logger.debug(f"Monitor took {time.time()-moni} seconds")
            cam.send(out)
            self.logger.debug(f"Manipulation with background took and with monitor projection {time.time()-mani} seconds!")
        else:
            cam.send(pad)
            self.logger.debug(f"Manipulation without background took {time.time()-mani} seconds!")
        if self.log_counter_face_success==0:
            self.logger.debug("Face control established.")
            self.log_counter_face_success+=1
    def overlay_on_monitor(self,background_img, overlay_img):
        def order_points(pts):
            rect = np.zeros((4, 2), dtype="float32")
            s = pts.sum(axis=1)
            diff = np.diff(pts, axis=1)
            rect[0] = pts[np.argmin(s)]
            rect[2] = pts[np.argmax(s)]
            rect[1] = pts[np.argmin(diff)]
            rect[3] = pts[np.argmax(diff)]
            return rect
        def expand_quad(pts, expand_px):
            cx, cy = np.mean(pts, axis=0)
            expanded = []
            for x, y in pts:
                direction = np.array([x - cx, y - cy], dtype=np.float32)
                norm = np.linalg.norm(direction)
                if norm != 0:
                    unit = direction / norm
                    new_pt = np.array([x, y]) + unit * expand_px
                else:
                    new_pt = np.array([x, y])
                expanded.append(new_pt)
            return np.array(expanded, dtype="float32")
        if self.previous_green_screen==None or (self.previous_green_screen != self.green_screen):
            hsv = cv2.cvtColor(background_img, cv2.COLOR_BGR2HSV)            
            lower_green = np.array([50, 100, 100])
            upper_green = np.array([95, 255, 255])            
            mask = cv2.inRange(hsv, lower_green, upper_green)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                print("No green screen found.")
                return background_img            
            largest_contour = max(contours, key=cv2.contourArea)
            epsilon = 0.02 * cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)

            if len(approx) != 4:
                print("Monitor contour not detected as a quadrilateral.")
                x, y, w, h = cv2.boundingRect(largest_contour)
                dst_pts = np.array([
                        [x, y],
                        [x + w, y],
                        [x + w, y + h],
                        [x, y + h]
                    ], dtype="float32")
            else:
                dst_pts = np.array([point[0] for point in approx], dtype="float32")
            dst_pts = order_points(dst_pts)
            self.dst_pts = expand_quad(dst_pts,2)
            self.previous_green_screen=self.green_screen
        h, w = overlay_img.shape[:2]
        src_pts = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype="float32")

        M = cv2.getPerspectiveTransform(src_pts, self.dst_pts)
        warped_overlay = cv2.warpPerspective(overlay_img, M, (background_img.shape[1], background_img.shape[0]))

        # Mask out green screen
        mask_warp = cv2.warpPerspective(np.ones_like(overlay_img, dtype=np.uint8)*255, M, (background_img.shape[1], background_img.shape[0]))
        mask_gray = cv2.cvtColor(mask_warp, cv2.COLOR_BGR2GRAY)
        mask_inv = cv2.bitwise_not(mask_gray)

        bg_masked = cv2.bitwise_and(background_img, background_img, mask=mask_inv)
        fg_masked = cv2.bitwise_and(warped_overlay, warped_overlay, mask=mask_gray)

        # Combine background and foreground
        combined = cv2.add(bg_masked, fg_masked)
        combined = cv2.resize(combined,(960,540))
        return combined
    def no_manipulation(self,cam,frame):
        self.logger.debug("No manipulation starts.")
        no_mani=time.time()
        self.x_s, self.f_s, self.R_s, self.x_s_info, self.lip_delta_before_animation, self.crop_info, self.img_rgb = None, None, None, None, None, None, None
        if self.log_counter_cam_dupe==0:
            self.log_counter_cam_dupe+=1
        self.active=False
        self.source_image_path=None
        self.first_iter=True
        blended=cv2.addWeighted(frame, 1.0 - 0.2, self.overlay_rgb, 0.2, 0)
        blended = blended.astype(np.uint8)
        if self.log_counter_cam_dupe_success==0:
            self.logger.debug("Duplicated camera feed is succesful.")
            self.log_counter_cam_dupe_success+=1
        cam.send(blended)
        self.logger.debug(f"No manipulation took {time.time()-no_mani} seconds!")
    def background_blur(self,frame,background_img):
            input_blob=self.preprocess(frame)
            result=self.session.run(None,{"input.1":input_blob})[0]
            mask=self.postprocess(result,frame.shape[:2])
            fg=frame.astype(np.float32)/255.0
            bg=background_img.astype(np.float32)/255.0
            composite = fg * mask + bg * (1 - mask)
            composite = (composite * 255).astype(np.uint8)
            if composite.shape[:2] != (540,960): 
                return cv2.resize(composite,(960,540))
            else:
                return composite
    def preprocess(self,frame):
        img = cv2.resize(frame, (320, 320)).astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)[np.newaxis, :]
        return img.astype(np.float32)
    
    def postprocess(self, pred, shape):
            pred = pred.squeeze()
            pred = cv2.resize(pred, shape[::-1])
            pred = np.clip(pred, 0, 1)
            pred= np.power(pred, 0.8) # threshold may be lowered
            return pred[:, :, np.newaxis]

    def conf_virt_live_webcam(self):
        if platform.system() == "Windows":
            self.backend = "obs"
        elif platform.system() == "Linux":
            self.backend = "v4l2loopback"
        else:
            self.backend = "unknown"
    def set_source(self,source_img_path:str):
        self.first_iter=True
        if source_img_path==self.source_image_path:
            if source_img_path.endswith("7.jpg") or source_img_path.endswith("11.jpg"):
                pass
            else:
                self.green_screen=random.choice(self.green_screens)
                self.green_img=cv2.imread(self.green_screen)
                self.green_img=cv2.resize(self.green_img,(960,540))
                self.green_img=cv2.cvtColor(self.green_img, cv2.COLOR_BGR2RGB)
        else:
            self.green_screen=None
            try:
                load_image_rgb(source_img_path)
                self.source_image_path=source_img_path
                self.logger.debug("Image set successfully!")
                if source_img_path.endswith("7.jpg") or source_img_path.endswith("11.jpg"):
                    self.background_image_path=None
                else:
                    self.background_image_path=random.choice(self.background_images)
                    self.background_image=cv2.imread(self.background_image_path)
                    self.background_image=cv2.resize(self.background_image,(960,540))
                return "Image set successfully."
            except Exception as e:
                self.source_image_path=None
                return e
    def set_parameters(self,**kwargs):
        self.live_portrait_pipeline.update_values(kwargs)
    def status_funct(self):
        return(self.active)
    def set_run(self):
        self.running=True
    def stop(self):
        self.stop_signal=True
    def test(self,video_path:str,conf_list:typing.List[typing.Dict],pic_path:str):
        base_pic = os.path.splitext(os.path.basename(pic_path))[0]
        filename =str(base_pic)+'.mp4'
        output_path = os.path.join("output_videos", filename)
        os.makedirs("output_videos", exist_ok=True)
        img = cv2.imread(pic_path)
        if img is None:
            raise ValueError(f"Failed to load image from {pic_path}")
        height, width = img.shape[:2]
        fps = cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        for conf in conf_list:
            cap=cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            if not ret:
                self.logger.debug("Reached end of video or failed to read frame.")
                continue
            self.set_parameters(**conf)
            self.x_s, self.f_s, self.R_s, self.x_s_info, self.lip_delta_before_animation, self.crop_info, self.img_rgb = self.live_portrait_pipeline.execute_frame(frame, pic_path)
            text_lines = [f"{key}: {value}" for key, value in conf.items()]
            y0, dy = 30, 30
            while True:
                if not ret:
                    self.logger.debug("Reached end of video or failed to read frame.")
                    break
                result = self.live_portrait_pipeline.generate_frame(self.x_s, self.f_s, self.R_s, self.x_s_info, self.lip_delta_before_animation, self.crop_info, self.img_rgb, frame)
                result=cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                result = result[:-100, :, :]
                for i, line in enumerate(text_lines):
                    y = y0 + i * dy
                    cv2.putText(
                        result, line, (10, y),  # position
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,                    # font scale
                        (0, 0, 0),        # black text
                        2,                      # thickness
                        cv2.LINE_AA             # anti-aliased
                    )
                out.write(result)
                ret, frame = cap.read()
        cap.release()
        out.release()
if __name__ == '__main__':
    logging.basicConfig(
    level=logging.DEBUG,  # or INFO
    format='[%(asctime)s] %(levelname)s in %(name)s: %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
    )
    inf=Inference()
    inf.main()

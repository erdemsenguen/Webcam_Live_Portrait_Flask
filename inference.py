import tyro
from .src.config.argument_config import ArgumentConfig
from .src.config.inference_config import InferenceConfig
from .src.config.crop_config import CropConfig
from .src.live_portrait_pipeline import LivePortraitPipeline
from .src.crop import SCRFD, face_detector
from .src.utils.io import load_image_rgb
from .src.cuda_functions import FrameProcessor
from .src.cam_threading import WebcamStream
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
    def __init__(self, device: str, src: int = 0):
        tyro.extras.set_accent_color("bright_cyan")
        self.args = tyro.cli(ArgumentConfig)
        self.logger = logging.getLogger(__name__)
        self.virtual_cam_res_x = 1920
        self.virtual_cam_res_y = 1080
        self.src=src
        self.cap = None
        self.is_face = None
        self.first_iter = True
        # specify configs for inference
        self.inference_cfg = self.partial_fields(InferenceConfig, self.args.__dict__)
        self.crop_cfg = self.partial_fields(CropConfig, self.args.__dict__)
        self.source_image_path = None
        self.stop_signal = False
        self.live_portrait_pipeline = LivePortraitPipeline(
            inference_cfg=self.inference_cfg, crop_cfg=self.crop_cfg
        )
        # Get the directory of the current script
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.background_images = [
            f"{os.path.dirname(self.script_dir)}/Backgrounds/wall.jpg",
            f"{os.path.dirname(self.script_dir)}/Backgrounds/bookshelf.jpg",
            f"{os.path.dirname(self.script_dir)}/Backgrounds/inIT_Hindergrund.jpg",
        ]
        self.green_screens = [
            f"{os.path.dirname(self.script_dir)}/Backgrounds/meeting_green1.jpg",
            f"{os.path.dirname(self.script_dir)}/Backgrounds/meeting_green2.jpg",
        ]
        self.running = False
        self.active = False
        self.face_detector_model = SCRFD(
            model_file=f"{os.path.dirname(os.path.abspath(__file__))}/pretrained_weights/insightface/models/buffalo_l/det_10g.onnx"
        )
        self.background_image = None
        self.background_image_path = None
        self.green_screen = None
        self.previous_green_screen = None
        self.change_green_screen = False
        self.backend = None
        self.x_s = None
        self.f_s = None
        self.R_s = None
        self.x_s_info = None
        self.lip_delta_before_animation = None
        self.crop_info = None
        self.img_rgb = None
        self.cuda_cv2 = FrameProcessor()
        self.green_img = None
        self.dst_pts = None
        self.temp_source = None
        self.temp_green = None
        self.set_background=True
        self.first_source = True
        self.device = device
        self.session = ort.InferenceSession(
            f"{self.script_dir}/pretrained_weights/modnet/modnet_photographic_portrait_matting.onnx",
            providers=["CUDAExecutionProvider"],
        )
        self.logger.debug([i.name for i in self.session.get_inputs()])
        self.pad = np.zeros(
            (self.virtual_cam_res_y, self.virtual_cam_res_x, 3), dtype=np.uint8
        )
        self.conf_virt_live_webcam()

    def partial_fields(self, target_class, kwargs):
        return target_class(
            **{k: v for k, v in kwargs.items() if hasattr(target_class, k)}
        )

    def main(self):
        with pyvirtualcam.Camera(
            width=self.virtual_cam_res_x,
            height=self.virtual_cam_res_y,
            fps=30,
            backend="v4l2loopback",
            device=self.device,
        ) as cam:
            black_image = np.zeros(
                (self.virtual_cam_res_y, self.virtual_cam_res_x, 3), dtype=np.uint8
            )
            while True:
                if not self.running:
                    self.cuda_cv2.operate(
                        cam=cam,
                        frame=black_image,
                        width=self.virtual_cam_res_x,
                        height=self.virtual_cam_res_y,
                        flip=False,
                        color=False,
                        send_to_cam=True,
                    )
                    continue
                else:
                    black_image = None
                    break
            while True:
                if self.cap is None:
                    self.cap = WebcamStream(width=self.virtual_cam_res_x, height=self.virtual_cam_res_y, src=self.src)
                if self.first_source is False:
                    self.first_source = None
                    (
                        self.x_s,
                        self.f_s,
                        self.R_s,
                        self.x_s_info,
                        self.lip_delta_before_animation,
                        self.crop_info,
                        self.img_rgb,
                    ) = (None, None, None, None, None, None, None)
                    self.active = False
                    self.first_iter = True
                    self.source_image_path=None
                    self.temp_green=None
                    self.temp_source=None
                if self.first_source is True:
                    self.first_source = False
                    self.set_source(f"{os.path.dirname(self.script_dir)}/photos/1.jpg")
                loop_start = time.time()
                ret, frame = self.cap.read()
                if not ret:
                    continue
                frame = self.cuda_cv2.operate(
                    cam=None,
                    frame=frame,
                    width=self.virtual_cam_res_x,
                    height=self.virtual_cam_res_y,
                    flip=True,
                    color=True,
                    send_to_cam=False,
                )
                self.logger.debug(
                    f"Initial frame reading and refinement took{time.time()-loop_start}"
                )
                face_time = time.time()
                is_face = face_detector(
                    self.cuda_cv2.operate(cam=None, frame=frame, width=160, height=160),
                    self.face_detector_model,
                )
                self.logger.debug(f"Face detector took {time.time()-face_time}")
                self.temp_source_setter()
                self.temp_green_setter()
                if self.first_iter and self.source_image_path:
                    self.logger.debug("DeepFake source image is set!")
                    im_time = time.time()
                    (
                        self.x_s,
                        self.f_s,
                        self.R_s,
                        self.x_s_info,
                        self.lip_delta_before_animation,
                        self.crop_info,
                        self.img_rgb,
                    ) = self.live_portrait_pipeline.execute_frame(
                        frame, self.source_image_path
                    )
                    self.logger.debug(
                        f"Source image set took {time.time()-im_time} seconds"
                    )
                if is_face and self.img_rgb is not None:
                    if self.source_image_path:
                        mani_time = time.time()
                        self.manipulation(cam=cam, frame=frame)
                        self.logger.debug(f"Manipulation Took{time.time()-mani_time}")
                    else:
                        no_mani_time = time.time()
                        self.no_manipulation(cam=cam, frame=frame)
                        self.logger.debug(
                            f"No manipulation Took{time.time()-no_mani_time}"
                        )
                else:
                    no_mani_time = time.time()
                    self.no_manipulation(cam=cam, frame=frame)
                    self.logger.debug(f"No manipulation Took{time.time()-no_mani_time}")
                self.logger.debug(f"A loop took {time.time()-loop_start} seconds!")

    def manipulation(self, cam, frame):
        mani = time.time()
        self.active = True
        result = self.live_portrait_pipeline.generate_frame(
            self.x_s,
            self.f_s,
            self.R_s,
            self.x_s_info,
            self.lip_delta_before_animation,
            self.crop_info,
            self.img_rgb,
            frame,
        )
        self.first_iter = False
        self.logger.debug(
            f"The model has generated the image in {time.time()-mani} seconds!"
        )
        result_height, result_width = result.shape[:2]
        if (
            result_height > self.virtual_cam_res_y
            or result_width > self.virtual_cam_res_x
        ):
            result = self.cuda_cv2.operate(
                frame=result,
                width=self.virtual_cam_res_x
                * self.virtual_cam_res_y
                // result_height,
                height=self.virtual_cam_res_y,
            )
        x_offset = (self.virtual_cam_res_x - result_width) // 2
        y_offset = self.virtual_cam_res_y - result_height
        pad = self.pad.copy()
        pad[
            y_offset : y_offset + result_height, x_offset : x_offset + result_width
        ] = result
        if self.background_image is not None:
            background = self.background_image
            if self.set_background==True:
                out = self.background_blur(pad, background)
                if self.green_screen and self.green_img is not None:
                    out = self.overlay_on_monitor(self.green_img, out)
                self.cuda_cv2.operate(
                    cam=cam,
                    frame=out,
                    width=self.virtual_cam_res_x,
                    height=self.virtual_cam_res_y,
                    flip=False,
                    color=False,
                    send_to_cam=True,
                )
            self.set_background=True
        else:
            self.cuda_cv2.operate(
                cam=cam,
                frame=pad,
                width=self.virtual_cam_res_x,
                height=self.virtual_cam_res_y,
                flip=False,
                color=False,
                send_to_cam=True,
            )
            self.logger.debug(
                f"Manipulation without background took {time.time()-mani} seconds!"
            )

    def overlay_on_monitor(self, background_img, overlay_img):
        if self.dst_pts is not None:
            h, w = overlay_img.shape[:2]
            src_pts = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype="float32")

            M = cv2.getPerspectiveTransform(src_pts, self.dst_pts)
            warped_overlay = cv2.warpPerspective(
                overlay_img, M, (background_img.shape[1], background_img.shape[0])
            )

            # Mask out green screen
            mask_warp = cv2.warpPerspective(
                np.ones_like(overlay_img, dtype=np.uint8) * 255,
                M,
                (background_img.shape[1], background_img.shape[0]),
            )
            mask_gray = cv2.cvtColor(mask_warp, cv2.COLOR_BGR2GRAY)
            mask_inv = cv2.bitwise_not(mask_gray)

            bg_masked = cv2.bitwise_and(background_img, background_img, mask=mask_inv)
            fg_masked = cv2.bitwise_and(warped_overlay, warped_overlay, mask=mask_gray)

            # Combine background and foreground
            combined = cv2.add(bg_masked, fg_masked)
            combined = self.cuda_cv2.operate(
                frame=combined,
                width=self.virtual_cam_res_x,
                height=self.virtual_cam_res_y,
            )
            return combined
        else:
            pass

    def green_screen_change(self, background_img):
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

        hsv = cv2.cvtColor(background_img, cv2.COLOR_BGR2HSV)
        lower_green = np.array([50, 100, 100])
        upper_green = np.array([95, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return background_img
        largest_contour = max(contours, key=cv2.contourArea)
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)

        if len(approx) != 4:
            x, y, w, h = cv2.boundingRect(largest_contour)
            dst_pts = np.array(
                [[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype="float32"
            )
        else:
            dst_pts = np.array([point[0] for point in approx], dtype="float32")
        dst_pts = order_points(dst_pts)
        self.dst_pts = expand_quad(dst_pts, 5)
        self.previous_green_screen = self.green_screen
        self.change_green_screen = False

    def no_manipulation(self, cam, frame):
        self.logger.debug("No manipulation starts.")
        no_mani = time.time()
        (
            self.x_s,
            self.f_s,
            self.R_s,
            self.x_s_info,
            self.lip_delta_before_animation,
            self.crop_info,
            self.img_rgb,
        ) = (None, None, None, None, None, None, None)
        self.active = False
        self.first_iter = True
        self.source_image_path=None
        self.temp_green=None
        self.temp_source=None
        self.cuda_cv2.operate(
            cam=cam,
            frame=frame,
            width=self.virtual_cam_res_x,
            height=self.virtual_cam_res_y,
            flip=False,
            color=False,
            send_to_cam=True,
        )
        self.logger.debug(f"No manipulation took {time.time()-no_mani} seconds!")

    def background_blur(self, frame, background_img):
        input_blob = self.preprocess(frame)
        result = self.session.run(None, {"input": input_blob})[0]
        mask = self.postprocess(result, frame.shape[:2])
        self.logger.debug(
            f"[MASK] min={mask.min()}, max={mask.max()}, shape={mask.shape}"
        )
        fg = frame.astype(np.float32) / 255.0
        bg = background_img.astype(np.float32) / 255.0
        composite = fg * mask + bg * (1 - mask)
        composite = (composite * 255).astype(np.uint8)
        if composite.shape[:2] != (self.virtual_cam_res_y, self.virtual_cam_res_x):
            return self.cuda_cv2.operate(
                frame=composite,
                width=self.virtual_cam_res_x,
                height=self.virtual_cam_res_y,
            )
        else:
            return composite

    def preprocess(self, frame):
        frame = self.cuda_cv2.operate(frame=frame, height=512, width=512)
        frame = frame.astype(np.float32) / 255.0
        frame = frame.transpose(2, 0, 1)[np.newaxis, :]
        return frame

    def postprocess(self, pred, shape):
        pred=np.squeeze(pred)
        pred = self.cuda_cv2.operate(frame=pred, width=shape[1], height=shape[0])
        pred = np.expand_dims(pred, axis=2)
        return pred

    def conf_virt_live_webcam(self):
        if platform.system() == "Windows":
            self.backend = "obs"
        elif platform.system() == "Linux":
            self.backend = "v4l2loopback"
        else:
            self.backend = "unknown"

    def temp_source_setter(self):
        if self.temp_source != self.source_image_path and self.temp_source is not None:
            self.first_iter = True
            self.source_image_path = self.temp_source
            self.background_image = None
            self.set_background= False
            self.temp_source = None
            try:
                self.temp_green = None
                self.green_screen = None
                self.previous_green_screen = None
                self.change_green_screen = True
                self.logger.debug("Image set successfully!")
                if self.source_image_path.endswith(
                    "7.jpg"
                ) or self.source_image_path.endswith("11.jpg"):
                    self.background_image = None
                else:
                    self.background_image_path = random.choice(self.background_images)
                    self.background_image = self.cuda_cv2.operate(
                        frame=cv2.imread(self.background_image_path),
                        width=self.virtual_cam_res_x,
                        height=self.virtual_cam_res_y,
                        color=True,
                    )
                return "Image set successfully."
            except Exception as e:
                self.logger.error(e)
                self.source_image_path = None
                return e

    def set_source(self, source_img_path: str):
        if source_img_path == self.source_image_path:
            pass
        else:
            self.temp_source = source_img_path

    def temp_green_setter(self):
        if self.temp_green != self.green_screen and self.temp_green is not None:
            self.green_screen = self.temp_green
            self.change_green_screen = True
            self.green_img = cv2.imread(self.green_screen)
            if self.green_screen and self.green_img is not None:
                self.green_img = self.cuda_cv2.operate(
                    frame=self.green_img,
                    width=self.virtual_cam_res_x,
                    height=self.virtual_cam_res_y,
                    color=True,
                )
            self.green_screen_change(self.green_img)
        elif self.temp_green is None:
            self.green_screen = None
            self.green_img = None

    def set_greenscreen(self, green_screen_path: str):
        if green_screen_path != self.green_screen:
            self.temp_green = green_screen_path
        if green_screen_path == self.green_screen:
            self.temp_green = None

    def set_parameters(self, **kwargs):
        self.live_portrait_pipeline.update_values(kwargs)

    def status_funct(self):
        return self.active

    def active_image(self):
        return os.path.splitext(os.path.basename(self.source_image_path))[0]

    def set_run(self):
        self.running = True

    def stop(self):
        self.stop_signal = True
        self.cap.stop()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,  # or INFO
        format="[%(asctime)s] %(levelname)s in %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    inf = Inference()
    inf.main()

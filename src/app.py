import logging
import cv2
import os
import time
from datetime import datetime
import numpy as np

from src.config import Config
from src.camera import RTSPStream
from src.detector import YOLODetector, Detection
from src.tracker import SortTracker
from src.notifier import Notifier
from src.tray import SystemTray

logger = logging.getLogger(__name__)


class Application:
    def __init__(self, config: Config):
        self.config = config

        self.camera1 = RTSPStream(config.get_rtsp_url(config.nvr_channel_1), target_size=(640, 360), preprocess_fn=self._enhance_night_vision)
        self.camera2 = RTSPStream(config.get_rtsp_url(config.nvr_channel_2), target_size=(640, 360), preprocess_fn=self._enhance_night_vision)
        self.camera3 = RTSPStream(config.get_rtsp_url(config.nvr_channel_3), target_size=(640, 360), preprocess_fn=self._enhance_night_vision)
        self.camera4 = RTSPStream(config.get_rtsp_url(config.nvr_channel_4), target_size=(640, 360), preprocess_fn=self._enhance_night_vision)

        self.detector = YOLODetector(
            model_path=config.model_path,
            confidence_threshold=config.confidence_threshold,
            person_class_id=config.person_class_id,
        )

        #initialize 4 trackers
        self.tracker1 = SortTracker()
        self.tracker2 = SortTracker()
        self.tracker3 = SortTracker()
        self.tracker4 = SortTracker()

        self.notifier = Notifier(
            topic=config.ntfy_topic,
            server=config.ntfy_server,
            cooldown_seconds=config.notification_cooldown_seconds,
        )
        self.tray = SystemTray(on_quit=self._request_stop)
        self._should_run = True

        #screenshot setup
        self.screenshot_dir = "screenshots"
        os.makedirs(self.screenshot_dir, exist_ok=True) #create only if it doesn't exist
        self.last_screenshot_time = 0.0
        self.screenshot_cooldown = 8

        #motion detection setup
        self.bg_subtractors = {
            1: cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=False),
            2: cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=False),
            3: cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=False),
            4: cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=False),
        }

        self.motion_threshold = 5000
        self.windows_open = {1: True, 2: True, 3: True, 4: True}
        self.mask_windows_open = {1: True, 2: True, 3: True, 4: True}

        self.motion_rois = {
            1: np.array([[275, 5],
                [283, 66],
                [290, 99],
                [331, 112],
                [347, 134],
                [348, 175],
                [346, 209],
                [343, 248],
                [338, 283],
                [336, 318],
                [372, 305],
                [418, 353],
                [9, 354],
                [5, 52],
                [157, 9]],
                dtype=np.int32),
            2: np.array([[6, 353],
                        [6, 190],
                        [112, 101],
                        [125, 159],
                        [393, 114],
                        [443, 135],
                        [483, 156],
                        [449, 170],
                        [457, 202],
                        [254, 273],
                        [232, 246],
                        [129, 276],
                        [151, 356]],
                        dtype=np.int32),
            3: np.array([[401, 263],
                        [294, 228],
                        [267, 188],
                        [245, 136],
                        [212, 131],
                        [164, 114],
                        [115, 116],
                        [133, 146],
                        [143, 174],
                        [187, 178],
                        [201, 203],
                        [216, 229],
                        [204, 248],
                        [168, 260],
                        [135, 276],
                        [131, 307],
                        [147, 332],
                        [178, 324],
                        [195, 292],
                        [235, 285],
                        [259, 288],
                        [287, 312],
                        [287, 343],
                        [321, 353],
                        [361, 355]],
                        dtype=np.int32),
            4: np.array([
                [192, 80],
                [280, 49],
                [358, 76],
                [420, 110],
                [473, 152],
                [509, 188],
                [526, 238],
                [507, 287],
                [478, 318],
                [404, 313],
                [364, 256],
                [329, 218],
                [279, 255],
                [171, 136],
                [189, 105],
            ], dtype=np.int32)
            
        }
    def _is_night(self, start_hour=21, end_hour=7):
        current_hour = datetime.now().hour
        return current_hour>= start_hour or current_hour < end_hour

    def _get_motion_data(self,cam_id, frame):
        #convert to grayscale to make motion math faster
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #ROI mask logic
        roi_corners = self.motion_rois.get(cam_id)
        if roi_corners is not None:
            mask = np.zeros_like(gray)
            #fill roi with white
            cv2.fillPoly(mask, [roi_corners], 255)
            #everything outside roi turns black
            gray = cv2.bitwise_and(gray, gray, mask=mask)

            if self.tray.is_mask_enabled() and self.tray.is_window_enabled():
                win_name = f"MASK VIEW - CAM {cam_id}"
                cv2.imshow(win_name, gray)
                x_pos = 1280 + ((cam_id - 1) % 2) * 640
                y_pos = ((cam_id - 1) // 2) * 360
                cv2.moveWindow(win_name, x_pos, y_pos)

        fg_mask = self.bg_subtractors[cam_id].apply(gray)
        _,fg_mask = cv2.threshold(fg_mask,200,255,cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        fg_mask = cv2.dilate(fg_mask,kernel,iterations=2)
        contours, _= cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        has_motion = False
        big_blobs = []

        for c in contours:
            area = cv2.contourArea(c)
            if area > 400:
                has_motion = True

            human_blob_size = 2000
            if area > human_blob_size:
                x,y,w,h = cv2.boundingRect(c)
                big_blobs.append((x, y, x + w, y + h))
        return has_motion, big_blobs

    def _enhance_night_vision(self,frame, start_hour=21, end_hour=7):
        if not self._is_night(): return frame
        clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(8,8))
        #convert to LAB (lightness, color A, color B)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)
        #apply clahe to lab channel
        cl = clahe.apply(l_channel)
        enhanced_lab = cv2.merge((cl,a,b))
        return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    def _request_stop(self):
        self._should_run = False

    def run(self, show_window: bool = True):
        logger.info("Connecting to camera...")
        self.camera1.start()
        self.camera2.start()
        self.camera3.start()
        self.camera4.start()

        self.tray.start()

        frame_counter = 0

        try:
            while self._should_run:
                if self.tray.paused.is_set():
                    # Skip detection entirely while paused, but keep
                    # looping so Quit is still responsive.
                    cv2.waitKey(100)
                    continue

                mask_enabled =self.tray.is_mask_enabled()
                windows_enabled = self.tray.is_window_enabled()

                for cam_id in range(1, 5):
                    is_enabled = self.tray.is_camera_enabled(cam_id)
                    
                    if (not is_enabled or not windows_enabled) and self.windows_open[cam_id]:
                        try:
                            cv2.destroyWindow(f"Camera{cam_id}")
                            cv2.destroyWindow(f"MASK VIEW - CAM {cam_id}")
                        except cv2.error:
                            pass
                        self.windows_open[cam_id] = False
                        
                    elif is_enabled and windows_enabled:
                        self.windows_open[cam_id] = True

                    if (not mask_enabled or not is_enabled or not windows_enabled) and self.mask_windows_open[cam_id]:
                        try:
                            cv2.destroyWindow(f"MASK VIEW - CAM {cam_id}")
                        except cv2.error:
                            pass
                        self.mask_windows_open[cam_id] = False
                    elif mask_enabled and is_enabled and windows_enabled:
                        self.mask_windows_open[cam_id] = True


                frame1 = self.camera1.read()
                frame2 = self.camera2.read()
                frame3 = self.camera3.read()
                frame4 = self.camera4.read()

                if frame1 is None or frame2 is None or frame3 is None or frame4 is None:
                    cv2.waitKey(50)
                    continue
                    

                #optimization analytics
                start_time = time.time()
                frame_counter += 1

                detections1,detections2,detections3,detections4 = [],[],[],[]

                if frame_counter % 3 == 0:
                    is_night = self._is_night()
                    #drop conf to 0.25 from 0.5 at night
                    current_conf = 0.25 if is_night else self.config.confidence_threshold

                    raw_cams = {1:frame1, 2:frame2, 3:frame3, 4:frame4}
                    cams = {cam_id: f for cam_id, f in raw_cams.items() if self.tray.is_camera_enabled(cam_id)}

                    active_ids = []
                    active_frames = []
                    night_blobs_dict = {}

                    for cam_id, frame in cams.items():
                        has_motion, big_blobs = self._get_motion_data(cam_id, frame)
                        if has_motion:
                            active_ids.append(cam_id)
                            active_frames.append(frame)
                        if is_night:
                            night_blobs_dict[cam_id] = big_blobs
                    if active_frames:
                        batch_results = self.detector.detect_batch(active_frames, conf_override=current_conf)

                        for cam_id,detections in zip(active_ids,batch_results):

                            if is_night and cam_id in night_blobs_dict:
                                for blob_box in night_blobs_dict[cam_id]:
                                    bx1, by1, bx2, by2 = blob_box
                                    blob_center = ((bx1 + bx2) / 2, (by1 + by2) / 2)

                                    is_overlapping = False
                                    for det in detections:
                                        dx1, dy1, dx2, dy2 = det.bbox
                                        if dx1 < blob_center[0] < dx2 and dy1 < blob_center[1] < dy2:
                                            is_overlapping = True
                                            break
                                    if not is_overlapping:
                                        detections.append(Detection(bbox=blob_box, confidence=0.4, source="blob"))

                            if cam_id == 1: detections1 = detections
                            elif cam_id == 2: detections2 = detections
                            elif cam_id == 3: detections3 = detections
                            elif cam_id == 4: detections4 = detections
      
                tracked_people1 = self.tracker1.update(detections1) if self.tray.is_camera_enabled(1) else []
                tracked_people2 = self.tracker2.update(detections2) if self.tray.is_camera_enabled(2) else []
                tracked_people3 = self.tracker3.update(detections3) if self.tray.is_camera_enabled(3) else []
                tracked_people4 = self.tracker4.update(detections4) if self.tray.is_camera_enabled(4) else []


                if show_window:
                    current_time = time.time()
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    can_take_screenshot = (current_time - self.last_screenshot_time > self.screenshot_cooldown)
                    saved_any = False
                    camera_data = [
                        (1, frame1, tracked_people1),
                        (2, frame2, tracked_people2),
                        (3, frame3, tracked_people3),
                        (4, frame4, tracked_people4)]

                    for cam_id, frame,tracked_people in camera_data:

                        if not self.tray.is_camera_enabled(cam_id):
                            continue

                        #draw boxes
                        self._draw(frame,tracked_people)

                        #handle screenshots
                        if can_take_screenshot and len(tracked_people) > 0:
                            filename = os.path.join(self.screenshot_dir, f"cam{cam_id}_{timestamp}.jpg")
                            cv2.imwrite(filename,frame)
                            logger.info(f"Saved: {filename}")
                            saved_any = True

                            self.notifier.notify_person_detected(len(tracked_people), image_path=filename)

                        if self.tray.is_window_enabled():
                            window_name = f"Camera{cam_id}"
                            cv2.imshow(window_name, frame)
                            #handle drawing
                            window_name = f"Camera{cam_id}"
                            cv2.imshow(window_name,frame)

                            x_pos = ((cam_id-1)%2)*640
                            y_pos = ((cam_id-1)//2)*360

                            cv2.moveWindow(window_name, x_pos,y_pos)

                    #update screenshot cooldown
                    if saved_any:
                        self.last_screenshot_time = current_time

                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
            logger.info(f"Loop time: {time.time() - start_time:.3f}s")
        finally:
            self._shutdown()

    @staticmethod
    def _draw(frame, tracked_people):
        """Draws each tracked person's box and stable ID on the frame."""
        for person in tracked_people:
            x1, y1, x2, y2 = (int(v) for v in person.bbox)

            color = (0,200,0) if person.source == "yolo" else (0,165,255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame, f"ID {person.id}", (x1, max(y1 - 10, 0)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
            )

    def _shutdown(self):
        logger.info("Shutting down...")
        self.camera1.stop()
        self.camera2.stop()
        self.camera3.stop()
        self.camera4.stop()
        self.tray.stop()
        cv2.destroyAllWindows()


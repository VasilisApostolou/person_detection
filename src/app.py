import logging
import cv2
import os
import time
from datetime import datetime

from src.config import Config
from src.camera import RTSPStream
from src.detector import YOLODetector
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
        self.screenshot_cooldown = 10

        #motion detection setup
        self.bg_subtractors = {
            1: cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=False),
            2: cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=False),
            3: cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=False),
            4: cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=False),
        }

        self.motion_threshold = 5000

    def _has_motion(self,cam_id, frame):
        #convert to grayscale to make motion math faster
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fg_mask = self.bg_subtractors[cam_id].apply(gray)
        #compare current to memorized background
        _,fg_mask = cv2.threshold(fg_mask,200,255,cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        fg_mask = cv2.dilate(fg_mask,kernel,iterations=2)
        contours, _= cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            if cv2.contourArea(c) > 400:
                return True
        return False

    def _enhance_night_vision(self,frame, start_hour=21, end_hour=7):
        current_hour = datetime.now().hour
        is_night = current_hour >= start_hour or current_hour < end_hour
        if not is_night:
            return frame
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
                    cams = {1: frame1, 2:frame2, 3:frame3, 4:frame4}
                    active_ids = []
                    active_frames = []

                    for cam_id, frame in cams.items():
                        if self._has_motion(cam_id, frame):
                            active_ids.append(cam_id)
                            active_frames.append(frame)

                    if active_frames:
                        batch_results = self.detector.detect_batch(active_frames)

                        for cam_id,detections in zip(active_ids,batch_results):
                            if cam_id == 1: detections1 = detections
                            elif cam_id == 2: detections2 = detections
                            elif cam_id == 3: detections3 = detections
                            elif cam_id == 4: detections4 = detections
      
                tracked_people1 = self.tracker1.update(detections1)
                tracked_people2 = self.tracker2.update(detections2)
                tracked_people3 = self.tracker3.update(detections3)
                tracked_people4 = self.tracker4.update(detections4)


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
                        #draw boxes
                        self._draw(frame,tracked_people)

                        #handle screenshots
                        if can_take_screenshot and len(tracked_people) > 0:
                            filename = os.path.join(self.screenshot_dir, f"cam{cam_id}_{timestamp}.jpg")
                            cv2.imwrite(filename,frame)
                            logger.info(f"Saved: {filename}")
                            saved_any = True

                            self.notifier.notify_person_detected(len(tracked_people), image_path=filename)

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
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
            cv2.putText(
                frame, f"ID {person.id}", (x1, max(y1 - 10, 0)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2,
            )

    def _shutdown(self):
        logger.info("Shutting down...")
        self.camera1.stop()
        self.camera2.stop()
        self.camera3.stop()
        self.camera4.stop()
        self.tray.stop()
        cv2.destroyAllWindows()

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

        self.camera1 = RTSPStream(config.get_rtsp_url(config.nvr_channel_1))
        self.camera2 = RTSPStream(config.get_rtsp_url(config.nvr_channel_2))
        self.camera3 = RTSPStream(config.get_rtsp_url(config.nvr_channel_3))
        self.camera4 = RTSPStream(config.get_rtsp_url(config.nvr_channel_4))

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
        os.makedirs(self.screenshot_dir, exist_ok=True) #create only if it doesnt exist
        self.last_screenshot_time = 0.0
        self.screenshot_cooldown = 10


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

                frame1 = cv2.resize(frame1, (1280,720))
                frame2 = cv2.resize(frame2, (1280,720))
                frame3 = cv2.resize(frame3, (1280,720))
                frame4 = cv2.resize(frame4, (1280,720))

                frame_counter += 1

                if frame_counter % 3 == 0:
                    detections1 = self.detector.detect(frame1)
                    detections2 = self.detector.detect(frame2)
                    detections3 = self.detector.detect(frame3)
                    detections4 = self.detector.detect(frame4)
                else:
                    detections1 = []
                    detections2 = []
                    detections3 = []
                    detections4 = []
      
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

                        x_pos = ((cam_id-1)%2)*1280
                        y_pos = ((cam_id-1)//2)*640

                        cv2.moveWindow(window_name, x_pos,y_pos)

                    #update screenshot cooldown
                    if saved_any:
                        self.last_screenshot_time = current_time

                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
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

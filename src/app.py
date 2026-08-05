import logging
import cv2

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

    def _request_stop(self):
        self._should_run = False

    def run(self, show_window: bool = True):
        logger.info("Connecting to camera...")
        self.camera1.start()
        self.camera2.start()
        self.camera3.start()
        self.camera4.start()

        self.tray.start()

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

                detections1 = self.detector.detect(frame1)
                tracked_people1 = self.tracker1.update(detections1)
                detections2 = self.detector.detect(frame2)
                tracked_people2 = self.tracker2.update(detections2)
                detections3 = self.detector.detect(frame3)
                tracked_people3 = self.tracker2.update(detections3)
                detections4 = self.detector.detect(frame4)
                tracked_people4 = self.tracker2.update(detections4)

                total_people = len(tracked_people1) + len(tracked_people2) + len(tracked_people3) + len(tracked_people4)

                if total_people > 0:
                    self.notifier.notify_person_detected(total_people)

                if show_window:
                    self._draw(frame1, tracked_people1)
                    self._draw(frame2, tracked_people2)
                    self._draw(frame3, tracked_people3)
                    self._draw(frame4, tracked_people4)

                    cv2.imshow("Camera 1", frame1)
                    cv2.moveWindow("Camera 1", 0, 0)

                    cv2.imshow("Camera 2", frame2)
                    cv2.moveWindow("Camera 2", 1280,0)

                    cv2.imshow("Camera 3", frame3)
                    cv2.moveWindow("Camera 3", 0, 630)

                    cv2.imshow("Camera 4", frame4)
                    cv2.moveWindow("Camera 4", 1280,630)

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
        self.tray.stop()
        cv2.destroyAllWindows()

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
        self.camera = RTSPStream(config.rtsp_url)
        self.detector = YOLODetector(
            model_path=config.model_path,
            confidence_threshold=config.confidence_threshold,
            person_class_id=config.person_class_id,
        )
        self.tracker = SortTracker()
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
        self.camera.start()
        self.tray.start()

        try:
            while self._should_run:
                if self.tray.paused.is_set():
                    # Skip detection entirely while paused, but keep
                    # looping so Quit is still responsive.
                    cv2.waitKey(100)
                    continue

                frame = self.camera.read()
                if frame is None:
                    cv2.waitKey(50)
                    continue

                detections = self.detector.detect(frame)
                tracked_people = self.tracker.update(detections)

                if tracked_people:
                    self.notifier.notify_person_detected(len(tracked_people))

                if show_window:
                    self._draw(frame, tracked_people)
                    cv2.imshow("Person Detection", frame)
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
        self.camera.stop()
        self.tray.stop()
        cv2.destroyAllWindows()

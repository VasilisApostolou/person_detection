import cv2
import threading
import time
import logging

logger = logging.getLogger(__name__)

class RTSPStream:
    def __init__(self, url: str, reconnect_delay: float = 3.0, target_size: tuple = None, preprocess_fn=None):
        self.url = url
        self.reconnect_delay = reconnect_delay
        self.target_size = target_size
        self.preprocess_fn = preprocess_fn
        self._frame = None
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        self._cap = None

    def start(self):
        self._running = True
        self._cap = cv2.VideoCapture(self.url)
        self._thread = threading.Thread(target=self._update_loop, daemon=True)
        self._thread.start()
        return self

    def _update_loop(self):
        failures = 0
        while self._running:
            if self._cap is None or not self._cap.isOpened():
                logger.warning("Reconnecting ...")
                self._cap = cv2.VideoCapture(self.url)
                time.sleep(self.reconnect_delay)
                continue

            ret, frame = self._cap.read()

            if ret:
                failures = 0
                
                # Apply resizing and night-vision enhancement on the background thread
                if self.target_size:
                    frame = cv2.resize(frame, self.target_size)
                if self.preprocess_fn:
                    frame = self.preprocess_fn(frame)
                    
                with self._lock:
                    self._frame = frame
            else:
                failures += 1
                if failures >= 10:
                    logger.warning("Stream dropped. Forcing reconnect.")
                    self._cap.release()
                    self._cap = None
                    failures = 0
                time.sleep(0.1)

    def read(self):
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._cap:
            self._cap.release()
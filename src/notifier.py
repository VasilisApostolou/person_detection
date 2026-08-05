#ntfy notification to phone

import time
import logging
import requests
import threading
import os

logger = logging.getLogger(__name__)

class Notifier:
    def __init__(self, topic: str, server: str, cooldown_seconds: int):
        self.topic = topic
        self.server = server.rstrip("/")
        self.cooldown_seconds = cooldown_seconds
        self._last_sent = 0.0

    def notify_person_detected(self, count: int, image_path: str = None):
        now = time.time()
        if now - self._last_sent < self.cooldown_seconds:
            return

        if not self.topic:
            logger.debug("No ntfy topic configured — skipping notification")
            return

        message = f"{count} person(s) detected" if count != 1 else "Person detected"

        def send_notification():
            try:
                headers = {"Title": "Camera Alert",
                           "Priority": "default",
                           "Tags": "warning"}

                if image_path and os.path.exists(image_path):
                    headers["Filename"] = os.path.basename(image_path)

                    with open(image_path, 'rb') as f:
                        requests.post(
                            f"{self.server}/{self.topic}",
                            data=f,
                            headers=headers,
                            timeout=10,
                        )
                else:
                    requests.post(
                        f"{self.server}/{self.topic}",
                        data=message.encode("utf-8"),
                        headers=headers,
                        timeout=5,
                    )
            except requests.RequestException as e:
                        logger.warning("Failed to send notification: %s", e)
        threading.Thread(target=send_notification, daemon=True).start()
        self._last_sent = now

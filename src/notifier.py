#ntfy notification to phone

import time
import logging
import requests

logger = logging.getLogger(__name__)

class Notifier:
    def __init__(self, topic: str, server: str, cooldown_seconds: int):
        self.topic = topic
        self.server = server.rstrip("/")
        self.cooldown_seconds = cooldown_seconds
        self._last_sent = 0.0

    def notify_person_detected(self, count: int):
        now = time.time()
        if now - self._last_sent < self.cooldown_seconds:
            return

        if not self.topic:
            logger.debug("No ntfy topic configured — skipping notification")
            return

        message = f"{count} person(s) detected" if count != 1 else "Person detected"

        try:
            requests.post(
                f"{self.server}/{self.topic}",
                data=message.encode("utf-8"),
                headers={"Title": "Camera Alert", "Priority": "default", "Tags": "warning"},
                timeout=5,
            )
            self._last_sent = now
        except requests.RequestException as e:
            # A failed notification (e.g. no internet momentarily)
            # should never crash the detection loop — log it and move on.
            logger.warning("Failed to send notification: %s", e)

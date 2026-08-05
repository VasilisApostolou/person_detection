import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class Config:
    nvr_ip: str = os.getenv("NVR_IP", "192.168.1.100")
    nvr_port: int = int(os.getenv("NVR_PORT", "554"))
    nvr_user: str = os.getenv("NVR_USER", "admin")
    nvr_password: str = os.getenv("NVR_PASSWORD", "")
    nvr_channel: int = int(os.getenv("NVR_CHANNEL", "1"))
    nvr_stream_type: str = os.getenv("NVR_STREAM_TYPE", "sub")

    model_path: str = os.getenv("MODEL_PATH", "yolov8n.pt")
    confidence_threshold: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))
    person_class_id: int = 0

    ntfy_topic: str = os.getenv("NTFY_TOPIC", "")
    ntfy_server: str = os.getenv("NTFY_SERVER", "https://ntfy.sh")
    notification_cooldown_seconds: int = int(os.getenv("NOTIFICATION_COOLDOWN_SECONDS", "60"))

    @property
    def rtsp_url(self) -> str:
        return (
            f"rtsp://{self.nvr_user}:{self.nvr_password}@"
            f"{self.nvr_ip}:{self.nvr_port}/"
            f"chID={self.nvr_channel}&streamType={self.nvr_stream_type}"
        )

config = Config()
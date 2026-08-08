#making detector class in case of changing model or updating tracker in general
#makes the app scalable for the future by encasing everything in a class.

from dataclasses import dataclass
from ultralytics import YOLO
import numpy as np

@dataclass
class Detection:
    bbox: tuple
    confidence: float
    source: str = "yolo"

    @property
    def center(self):
        x1,y1,x2,y2 = self.bbox
        return ((x1+x2) / 2, (y1+y2) / 2)

class YOLODetector:
    def __init__(self, model_path: str, confidence_threshold: float, person_class_id: int):
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.person_class_id = person_class_id

    def detect_batch(self, frames: list, conf_override: float = None) -> list:
        if not frames:
            return []

        threshold = conf_override if conf_override is not None else self.confidence_threshold
        results = self.model(frames,
                             classes=[self.person_class_id],
                             conf=threshold,
                             verbose=False)
        all_detections = []
        for result in results:
            detections = []
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = float(box.conf[0])
                detections.append(Detection(bbox=(x1, y1, x2, y2), confidence=confidence))
            all_detections.append(detections)      
        return all_detections     

    def detect(self,frame: np.ndarray):
        return self.detect_batch([frame])[0]
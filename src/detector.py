#making detector class in case of changing model or updating tracker in general
#makes the app scalable for the future by encasing everything in a class.

from dataclasses import dataclass
from ultralytics import YOLO
import numpy as np

@dataclass
class Detection:
    bbox: tuple
    confidence: float

    @property
    def center(self):
        x1,y1,x2,y2 = self.bbox
        return ((x1+x2) / 2, (y1+y2) / 2)

class YOLODetector:
    def __init__(self, model_path: str, confidence_threshold: float, person_class_id: int):
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.person_class_id = person_class_id

    def detect(self,frame: np.ndarray):
        results = self.model(
            frame,
            classes = [self.person_class_id],
            conf=self.confidence_threshold,
            verbose=False,
        )

        detections = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            confidence = float(box.conf[0])
            detections.append(Detection(bbox=(x1, y1, x2, y2), confidence=confidence))

        return detections

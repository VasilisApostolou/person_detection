import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
from dataclasses import dataclass

@dataclass
class TrackedPerson:
    id: int
    bbox: tuple

class KalmanTrack:
    #class level counter to assign unique IDs to each track
    count = 0  

    def __init__(self, bbox):
        
        #bbox: (x1, y1, x2, y2) from YOLO detector
        #Internally we work with [cx, cy, s, r,  ẋ, ẏ, ṡ]
        #[center_x, center_y, scale, aspect_ratio, velocity_x, velocity_y, velocity_scale]
        
        self.kf = KalmanFilter(dim_x=7, dim_z=4)

        # State transition matrix — constant velocity model
        # Maps current position and velocity to next position
        self.kf.F = np.array([
            [1,0,0,0,1,0,0],
            [0,1,0,0,0,1,0],
            [0,0,1,0,0,0,1],
            [0,0,0,1,0,0,0],
            [0,0,0,0,1,0,0],
            [0,0,0,0,0,1,0],
            [0,0,0,0,0,0,1]
        ], dtype=float)

        # Measurement matrix — detector only gives us [x, y, s, r], not velocities
        self.kf.H = np.array([
            [1,0,0,0,0,0,0],
            [0,1,0,0,0,0,0],
            [0,0,1,0,0,0,0],
            [0,0,0,1,0,0,0]
        ], dtype=float)

        # Measurement noise R — detector is less precise about scale than position
        self.kf.R[2,2] *= 10.0
        self.kf.R[3,3] *= 10.0

        # Initial uncertainty P — we have no idea about velocities at birth so big uncertainty
        self.kf.P[4,4] *= 1000.0
        self.kf.P[5,5] *= 1000.0
        self.kf.P[6,6] *= 1000.0
        self.kf.P[2,2] *= 10.0

        # Process noise Q — how much can the state drift unexpectedly per frame
        self.kf.Q[4,4] *= 0.01
        self.kf.Q[5,5] *= 0.01
        self.kf.Q[6,6] *= 0.0001  # scale changes very smoothly

        # Seed the filter with the first detection
        self.kf.x[:4] = self._bbox_to_z(bbox)

        # Track history and metrics
        self.id = KalmanTrack.count
        KalmanTrack.count += 1
        self.hits = 1             # matched detections so far
        self.no_match_streak = 0  # consecutive frames with no match
        self.age = 0              # total frames alive


    def _bbox_to_z(self, bbox):
        # Converts (x1,y1,x2,y2) to column vector [[cx],[cy],[area],[aspect_ratio]]
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        cx = x1 + w / 2.0
        cy = y1 + h / 2.0
        s  = w * h           # scale = bounding box area
        r  = w / float(h)    # aspect ratio 
        return np.array([[cx],[cy],[s],[r]], dtype=float)

    def _z_to_bbox(self, x=None):
        # Converts State vector to (x1,y1,x2,y2)
        if x is None:
            x = self.kf.x
        cx, cy, s, r = x[0,0], x[1,0], x[2,0], x[3,0]
        w = np.sqrt(max(s * r, 1e-6))
        h = w / max(r, 1e-6)
        return (
            int(cx - w/2), int(cy - h/2),
            int(cx + w/2), int(cy + h/2)
        )

    def predict(self):
        # Predict the next state of the track using the Kalman filter.
        # Prevent area going negative if the scale velocity is too large
        if self.kf.x[6,0] + self.kf.x[2,0] <= 0:
            self.kf.x[6,0] = 0.0
        self.kf.predict()
        self.age += 1
        self.no_match_streak += 1  # will be reset to 0 if matched this frame
        return self._z_to_bbox()

    def update(self, bbox):
        #Correct the filter with a matched detection.
        self.no_match_streak = 0
        self.hits += 1
        self.kf.update(self._bbox_to_z(bbox))

    def get_state(self):
        # Return current best estimate as (x1,y1,x2,y2).
        return self._z_to_bbox()

# SORT implementation
class SortTracker:
    def __init__(self, max_age=3, min_hits=2, iou_threshold=0.25):
        
        #max_age       : frames a track survives without a detection match before being killed
        #min_hits      : how many consecutive matches before we show the track (avoids ghost tracks)
        #iou_threshold : minimum IoU to accept a detection→track assignment
        
        self.tracks = []           # list of active KalmanTrack objects
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold

#IoU = Area of Overlap / Area of Union
    def _iou(self, boxA, boxB):

        #computes basic IoU (Intersection over Union) between two bounding boxes

        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        inter_w = max(0, xB - xA)
        inter_h = max(0, yB - yA)
        inter_area = inter_w * inter_h

        if inter_area == 0:
            return 0.0

        areaA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
        areaB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
        union_area = areaA + areaB - inter_area

        return inter_area / max(union_area, 1e-6)

    def _build_cost_matrix(self, predicted_boxes, detection_boxes):
        # Builds an assignment cost matrix for the Hungarian algorithm based on IoU.
        # Cost = 1 - IoU, so higher IoU = lower cost.
        n = len(predicted_boxes)
        m = len(detection_boxes)
        cost = np.zeros((n, m), dtype=float)
        for i, pb in enumerate(predicted_boxes):
            for j, db in enumerate(detection_boxes):
                cost[i, j] = 1.0 - self._iou(pb, db)
        return cost


    def update(self, detections):

        #1. Predict new positions for every existing track 
        predicted_boxes = []
        for track in self.tracks:
            predicted_boxes.append(track.predict())

        det_boxes = [det.bbox for det in detections]

        #2. Match predictions → detections via Hungarian algorithm
        matched, unmatched_dets = self._match(predicted_boxes, det_boxes)

        #3. Update matched tracks with their detection 
        for track_idx, det_idx in matched:
            self.tracks[track_idx].update(detections[det_idx].bbox)

        #4. Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            self.tracks.append(KalmanTrack(detections[det_idx].bbox))

        #5. Kill tracks that haven't been matched for too long 
        self.tracks = [t for t in self.tracks if t.no_match_streak <= self.max_age]

        #6. Return confirmed tracks as { id: (cx, cy) } 
        result = []
        for track in self.tracks:
            # Only report tracks that have been confirmed (min_hits matched frames)
            # OR are very new but still alive (age check avoids showing ghosts)
            if track.hits >= self.min_hits or track.no_match_streak == 0:
                x1, y1, x2, y2 = track.get_state()
                result.append(TrackedPerson(id=track.id, bbox=(x1,y1,x2,y2)))
        return result

    def _match(self, predicted_boxes, detection_boxes):
        if len(self.tracks) == 0 or len(detection_boxes) == 0:
            return [], list(range(len(detection_boxes)))

        cost_matrix = self._build_cost_matrix(predicted_boxes, detection_boxes)

        # linear_sum_assignment solves the assignment problem in O(n3)
        # It returns the row and column indices of the optimal assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched = []
        unmatched_dets = list(range(len(detection_boxes)))

        for r, c in zip(row_ind, col_ind):
            iou_score = 1.0 - cost_matrix[r, c]
            if iou_score >= self.iou_threshold:
                matched.append((r, c))
                if c in unmatched_dets:
                    unmatched_dets.remove(c)
            # if IoU is too low, the detection stays in unmatched_dets → new track

        return matched, unmatched_dets
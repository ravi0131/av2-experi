import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
import xxhash

@dataclass
class BoundingBox:
    x: float  # center x
    y: float  # center y
    l: float  # length
    w: float  # width
    theta: float  # heading angle
    id: int
    temporal_score: int
    
    def __init__(self, x, y, l, w, theta):
        self.x = x
        self.y = y
        self.l = l
        self.w = w
        self.theta = theta
        attrs = f"{x:.10f},{y:.10f},{l:.10f},{w:.10f},{theta:.10f}".encode()
        self.id = xxhash.xxh64(attrs).intdigest()
        self.temporal_score = 0
    

@dataclass
class Tracklet:
    id: int
    boxes: List[BoundingBox]
    confidence: float
    velocities: List[Tuple[float, float]]
    length: int
    
    def predict_next_position(self) -> Tuple[float, float]:
        assert self.length >= 1
        if len(self.boxes) >= 2:
            # Constant velocity assumption
            last_box = self.boxes[-1]
            prev_box = self.boxes[-2]
            return (
                2 * last_box.x - prev_box.x,
                2 * last_box.y - prev_box.y
            )
        else:
            return self.boxes[-1].x, self.boxes[-1].y
    
class OnlineTracker:
    def __init__(self):
        self.tracklets: List[Tracklet] = []
        self.next_id = 0
        self.max_distance = 5.0  # 5.0m matching threshold
        self.min_confidence = 0.1
        self.nms_iou_threshold = 0.1
        
    def update(self, detections: List[BoundingBox]) -> List[Tracklet]:
        # Predict new positions for existing tracklets
        predictions = []
        for tracklet in self.tracklets:
            x, y = tracklet.predict_next_position()
            predictions.append((x, y))
            
        # Compute distance matrix
        cost_matrix = np.zeros((len(self.tracklets), len(detections)))
        for i, pred in enumerate(predictions):
            for j, det in enumerate(detections):
                dist = np.sqrt((pred[0] - det.x)**2 + (pred[1] - det.y)**2)
                cost_matrix[i,j] = dist
                
        # Greedy matching
        matched_tracklets = set()
        matched_detections = set()
        matches = []
        
        for i in range(len(self.tracklets)):
            if len(detections) == 0:
                break
                
            distances = cost_matrix[i]
            best_match = np.argmin(distances)
            
            if distances[best_match] <= self.max_distance:
                matches.append((i, best_match))
                matched_tracklets.add(i)
                matched_detections.add(best_match)
        
        # Update matched tracklets
        for track_idx, det_idx in matches:
            self._update_tracklet(self.tracklets[track_idx], detections[det_idx])
            
        # Handle unmatched tracklets
        for i in range(len(self.tracklets)):
            if i not in matched_tracklets:
                self._extend_tracklet(self.tracklets[i])
                
        # Create new tracklets
        for i in range(len(detections)):
            if i not in matched_detections:
                self._create_tracklet(detections[i])
                
        # Remove low confidence tracklets
        self.tracklets = [t for t in self.tracklets 
                         if t.confidence >= self.min_confidence]
        
        # Apply NMS
        self._apply_nms()
        
        return self.tracklets
    
    def _update_tracklet(self, tracklet: Tracklet, detection: BoundingBox):
        w = sum(0.9**i for i in range(1, tracklet.length + 1))
        tracklet.confidence = (w * tracklet.confidence + 1.0) / (w + 1.0)
        tracklet.boxes.append(detection)
        tracklet.length += 1
        
    def _extend_tracklet(self, tracklet: Tracklet):
        x, y = tracklet.predict_next_position()
        last_box = tracklet.boxes[-1]
        new_box = BoundingBox(x, y, last_box.l, last_box.w, last_box.theta)
        tracklet.boxes.append(new_box)
        tracklet.confidence *= 0.9
        tracklet.length += 1
        
    def _create_tracklet(self, detection: BoundingBox):
        self.tracklets.append(Tracklet(
            id=self.next_id,
            boxes=[detection],
            confidence=0.9,
            velocities=[],
            length=1
        ))
        self.next_id += 1
        
class BidirectionalTracker:
    def __init__(self):
        self.forward_tracker = OnlineTracker()
        self.backward_tracker = OnlineTracker()
        
    def track_sequence(self, frame_detections: List[List[BoundingBox]]) -> Tuple[List[List[Tracklet]], List[List[Tracklet]]]:
        # Forward pass
        forward_tracklets = []
        for detections in frame_detections:
            tracks = self.forward_tracker.update(detections)
            forward_tracklets.append(tracks)
            
        # Backward pass
        backward_tracklets = []
        for detections in reversed(frame_detections):
            tracks = self.backward_tracker.update(detections)
            backward_tracklets.insert(0, tracks)
            
        return forward_tracklets, backward_tracklets
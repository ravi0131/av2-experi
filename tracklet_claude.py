import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict
import xxhash
from shapely.geometry import Polygon
from prototype_utils import get_bbox_corners
@dataclass
class BoundingBox:
    x: float  # center x
    y: float  # center y
    l: float  # length
    w: float  # width
    theta: float  # heading angle
    id: int
    temporal_score: int
    frame_id: int
    
    def __init__(self, x, y, l, w, theta, frame_id):
        self.x = x
        self.y = y
        self.l = l
        self.w = w
        self.theta = theta
        attrs = f"{x:.10f},{y:.10f},{l:.10f},{w:.10f},{theta:.10f}".encode()
        self.id = xxhash.xxh64(attrs).intdigest()
        self.temporal_score = 0
        self.frame_id = frame_id
    

@dataclass
class Tracklet:
    id: int
    boxes: List[BoundingBox]
    confidence: float
    velocities: List[Tuple[float, float]]
    length: int
    # backward_score: int = 0 # maybe not needed
    # forward_score: int = 0 # maybe not needed

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
        elif len(self.boxes) == 1:
            return self.boxes[-1].x, self.boxes[-1].y
        return 0.0, 0.0
@dataclass
class Frame: 
    detections: List[BoundingBox]
    tracklets: List[Tracklet]
    
    def __init__(self, detections: List[BoundingBox]):
        # Tracklets is a list of tracklets, but at time of initialization, it is empty, because it needs to be computed
        self.detections = detections
        self.tracklets = []

import copy        
class OnlineTracker:
    def __init__(self):
        # self.tracklets: List[List[Tracklet]] = []
        self.frames: List[Frame] = []
        # We store detections in a separate list because ,there might be tracklets that don't match to any detection
        # thus the bbox in such tracklets should not be considered while calculating the temporal score later on
        # from the scores of both forward and backward tracking
        # self.detections: List[List[BoundingBox]] = [] 
        self.next_id = 0
        self.max_distance = 5.0  # 5.0m matching threshold
        self.min_confidence = 0.1
        self.nms_iou_threshold = 0.1
        self.min_tracklet_length = 2  # Minimum length for consistent tracklets
        
    def update(self, detections_boxes: List[BoundingBox]) -> Frame:
        #self.detections.append(detections)
        frame = Frame(detections_boxes)
        
        #For the very first frame
        if len(self.frames) == 0:
            self.frames.append(frame)
            for box_idx, _ in enumerate(detections_boxes):
                self._create_tracklet(box_idx)
            self.frames[-1].tracklets = [t for t in self.frames[-1].tracklets if t.confidence >= self.min_confidence]
            self._apply_nms()
            return frame
        
        tracklets = copy.deepcopy(self.frames[-1].tracklets) # copy previous frame's tracklets
        frame.tracklets = tracklets
        self.frames.append(frame)
        detections = frame.detections
        predictions = [tracklet.predict_next_position() for tracklet in tracklets]

        # Compute distance matrix
        cost_matrix = np.full((len(tracklets), len(detections)), np.inf) # matrix of size num_tracklets x num_detections
        for i, pred in enumerate(predictions):
            for j, det in enumerate(detections):
                dist = np.sqrt((pred[0] - det.x)**2 + (pred[1] - det.y)**2)
                cost_matrix[i, j] = dist

        # Greedy matching
        matches, matched_tracklets, matched_detections = self._greedy_matching(cost_matrix)

        # Update matched tracklets
        for track_idx, det_idx in matches:
            self._update_tracklet(track_idx, det_idx)

        # Extend unmatched tracklets
        for i in range(len(tracklets)):
            if i not in matched_tracklets:
                self._extend_tracklet(i)

        # Create new tracklets for unmatched detections
        for i in range(len(detections)):
            if i not in matched_detections:
                self._create_tracklet(i)

        # Remove low-confidence tracklets
        tracklets = [t for t in tracklets if t.confidence >= self.min_confidence]

        # Apply NMS
        self._apply_nms()
        
        return frame

    def _greedy_matching(self, cost_matrix):
        """
        Calculate matches between tracklets and detections using a greedy algorithm.
        
        Args: 
            cost_matrix: 2D numpy array of shape (num_tracklets, num_detections) containing pairwise distances
            
        Returns:
            matches: List of tuples (track_idx, det_idx) representing matched tracklets and detections
            matched_tracklets: Set of indices of matched tracklets
            matched_detections: Set of indices of matched detections
        """
        matches = []
        matched_tracklets = set()
        matched_detections = set()

        for track_idx in range(cost_matrix.shape[0]):
            detection_distances = cost_matrix[track_idx]
            best_match = np.argmin(detection_distances)

            if best_match not in matched_detections and detection_distances[best_match] <= self.max_distance:
                matches.append((track_idx, best_match))
                matched_tracklets.add(track_idx)
                matched_detections.add(best_match)

        return matches, matched_tracklets, matched_detections

    def _update_tracklet(self, tracklet_idx: int, detection_idx: int):
        """A tracklet is updated if it is matched to a detection in the current frame"""
        tracklet = self.frames[-1].tracklets[tracklet_idx]
        detection = self.frames[-1].detections[detection_idx]
        w = sum(0.9**i for i in range(1, tracklet.length + 1))
        tracklet.confidence = (w * tracklet.confidence + 1.0) / (w + 1.0)
        tracklet.velocities.append((detection.x - tracklet.boxes[-1].x, detection.y - tracklet.boxes[-1].y))
        tracklet.boxes.append(detection)
        tracklet.length += 1
        detection.temporal_score = tracklet.length  # Update temporal score

    def _extend_tracklet(self, tracklet_idx: int):
        """we extend a tracklet if it is not matched to any detection in the current frame
        Args:
            tracklet_idx: index of the tracklet to extend
        """
        tracklet = self.frames[-1].tracklets[tracklet_idx]
        x, y = tracklet.predict_next_position()
        last_box = tracklet.boxes[-1]
        new_box = BoundingBox(x, y, last_box.l, last_box.w, last_box.theta,last_box.frame_id)
        tracklet.boxes.append(new_box)
        tracklet.confidence *= 0.9
        tracklet.length += 1

    def _create_tracklet(self, detection_idx: int):
        """create a new tracket for a detection that is not matched to any existing tracklet"""
        detection = self.frames[-1].detections[detection_idx]
        tracklet_to_add = Tracklet(
            id=self.next_id,
            boxes=[detection],
            confidence=0.9,
            velocities=[],
            length=1
        )
        self.frames[-1].tracklets.append(tracklet_to_add)
        self.next_id += 1
        detection.temporal_score = 0 # Unmatched detections have temporal score 0

    def _apply_nms(self):
        # Convert tracklet boxes to a format suitable for IoU computation
        tracklets = self.frames[-1].tracklets
        all_boxes = [tracklet.boxes[-1] for tracklet in tracklets]
        confidence  = [tracklet.confidence for tracklet in tracklets]
        keep = self._nms(all_boxes,confidence, self.nms_iou_threshold)
        tracklets = [tracklets[i] for i in keep]

    def _nms(self, boxes: List[BoundingBox],confidence_lst: List[float], iou_threshold: float) -> List[int]:
        """
        Apply non-maximum suppression to the given list of boxes.
        Args:
            boxes: List of BoundingBoxes
            iou_threshold: IoU threshold for suppression
        Returns:
            List of indices to keep
        """
        assert len(boxes) == len(confidence_lst)
        if not boxes:
            return []

        sorted_indices = np.argsort([cval for cval in confidence_lst])[::-1]  # Sort by confidence value of tracklets

        keep = []
        while sorted_indices.size > 0:
            current = sorted_indices[0]
            keep.append(current)
            remaining = sorted_indices[1:]

            current_box = boxes[current]
            remaining_boxes = [boxes[i] for i in remaining]

            ious = self._compute_iou(current_box, remaining_boxes)
            sorted_indices = remaining[ious < iou_threshold]

        return keep

    # def _compute_ious(self, box1: BoundingBox, boxes: List[BoundingBox]) -> np.ndarray:
    #     # Placeholder IoU computation (for BEV)
    #     def iou(b1, b2):
    #         inter_x = max(0, min(b1.x + b1.l / 2, b2.x + b2.l / 2) - max(b1.x - b1.l / 2, b2.x - b2.l / 2))
    #         inter_y = max(0, min(b1.y + b1.w / 2, b2.y + b2.w / 2) - max(b1.y - b1.w / 2, b2.y - b2.w / 2))
    #         inter_area = inter_x * inter_y
    #         union_area = b1.l * b1.w + b2.l * b2.w - inter_area
    #         return inter_area / union_area if union_area > 0 else 0

    #     return np.array([iou(box1, b) for b in boxes])

    def _compute_iou(self, box1: BoundingBox, boxes: List[BoundingBox]) -> np.ndarray:
        """Calculates IoU of the given box with the array of the given boxes.
        Note: the areas are passed in rather than calculated here for efficiency. 
        Calculate once in the caller to avoid duplicate work.
        
        Args:
            box: a polygon (shapely.geometry.Polygon)
            boxes: a numpy array of shape (N,), where each member is a shapely.geometry.Polygon
        Returns:
            a numpy array of shape (N,) containing IoU values
        """
        box1 = get_bbox_corners((box1.x, box1.y, box1.l, box1.w, box1.theta))
        boxes = [get_bbox_corners((b.x, b.y, b.l, b.w, b.theta)) for b in boxes]
        box1_poly = Polygon(box1)
        boxes_poly = [Polygon(b) for b in boxes]
        iou_lst = []
        for b in boxes_poly:
            intersection = box1_poly.intersection(b).area
            union = box1_poly.union(b).area
            iou = intersection / union if union > 0 else 0
            iou_lst.append(iou)
        # return np.array(iou_lst, dtype=np.float32)
        return np.array(iou_lst)

class BidirectionalTracker:
    def __init__(self):
        self.forward_tracker = OnlineTracker()
        self.backward_tracker = OnlineTracker()
        self.forward_track_results = []
        self.backward_track_results = []
        self.temporal_scores = []

    def track(self, frame_detections: List[List[BoundingBox]]) -> List[Frame]:
        """
        Perform tracking for a scene (sequence of lidar frames) in both forward and backward directions.
        
        Args:
            frame_detections: List of frames, where each frame is a list of BoundingBoxes
        
        Returns:
            List of frames with tracking results
        """
        forward_track_results, backward_track_results = self.track_sequence(frame_detections)
        return self.compute_temporal_scores(forward_track_results, backward_track_results)
    
    def track_sequence(self, frame_detections: List[List[BoundingBox]]) -> Tuple[List[Frame], List[Frame]]:
        """
        Perform tracking for a scene (sequence of lidar frames) in both forward and backward directions.
        
        Args:
            frame_detections: List of frames, where each frame is a list of BoundingBoxes
        
        Returns:
            forward_track_results: List of frames with forward tracking results
            backward_track_results: List of frames with backward tracking results
        """
        forward_track_results = []
        for frame in frame_detections:
            frame_obj = self.forward_tracker.update(frame)
            forward_track_results.append(frame_obj)

        backward_track_results = []
        for frame in reversed(frame_detections):
            frame_obj = self.backward_tracker.update(frame)
            backward_track_results.insert(0, frame_obj)

        self.forward_track_results = forward_track_results
        self.backward_track_results = backward_track_results
        
        return forward_track_results, backward_track_results
    
    def compute_temporal_scores(self, forward_track_results: List[Frame], backward_track_results: List[Frame]) -> List[List[Dict]]:
        """
        Compute temporal scores for each bbox in a frame 
        
        Args:
            forward_track_results: List of frames with forward tracking results
            backward_track_results: List of frames with backward tracking results
        
        Returns:
            scene_lst: List of lists of dictionaries. 
                Each dictionary is of the form {"bbox": BoundingBox, "forward_tracklet": Tracklet, "backward_tracklet": Tracklet}
                A dictionary corresponds to a tuple of a bbox and its corresponding forward and backward tracklets, in a given frame
                A list of such dictionaries corresponds to a frame in the scene
                A list of such lists corresponds to the entire scene
        """
        assert len(forward_track_results) == len(backward_track_results)
        scene_lst = []
        for forward_frame, backward_frame in zip(forward_track_results, backward_track_results):
            forward_tracklets = {tracklet.boxes[-1].id:(tracklet.boxes[-1], tracklet) for tracklet in forward_frame.tracklets}
            backward_tracklets = {tracklet.boxes[-1].id:( tracklet.boxes[-1], tracklet) for tracklet in backward_frame.tracklets}
            tuple_lst = []
            for bbox_id in forward_tracklets.keys() & backward_tracklets.keys():
                bbox1, tracklet1 = forward_tracklets[bbox_id]
                bbox2, tracklet2 = backward_tracklets[bbox_id]
                
                updated_bbox = copy.deepcopy(bbox1)
                updated_bbox.temporal_score = max(bbox1.temporal_score, bbox2.temporal_score)
                tuple_elem = {
                    "bbox": updated_bbox,
                    "forward_tracklet": tracklet1,
                    "backward_tracklet": tracklet2
                }
                tuple_lst.append(tuple_elem)
            
            scene_lst.append(tuple_lst)
        self.temporal_scores = scene_lst
        return scene_lst
    
    
import pandas as pd

def convert_to_bounding_boxes(df: pd.DataFrame, frame_id: int) -> List[BoundingBox]:
    """
    Convert a DataFrame of pseudo-labels to a list of BoundingBox objects.
    
    Args:
        df (pd.DataFrame): DataFrame with columns cx, cy, length, width, angle.

    Returns:
        List[BoundingBox]: List of bounding box objects for the given frame.
    """
    return [
        BoundingBox(
            x=row['box_center_x'],
            y=row['box_center_y'],
            l=row['box_length'],
            w=row['box_width'],
            theta=row['ry'],
            frame_id=frame_id
        )
        for _, row in df.iterrows()]



import json
from dataclasses import asdict
import numpy as np
import pickle
import os

def serialize_bbox(bbox):
    """Convert BoundingBox object to dictionary"""
    return {
        'x': float(bbox.x),
        'y': float(bbox.y),
        'l': float(bbox.l),
        'w': float(bbox.w),
        'theta': float(bbox.theta),
        'id': int(bbox.id),
        'temporal_score': int(bbox.temporal_score),
        'frame_id': int(bbox.frame_id)
    }

def serialize_tracklet(tracklet):
    """Convert Tracklet object to dictionary"""
    return {
        'id': int(tracklet.id),
        'boxes': [serialize_bbox(box) for box in tracklet.boxes],
        'confidence': float(tracklet.confidence),
        'velocities': [[float(vx), float(vy)] for vx, vy in tracklet.velocities],
        'length': int(tracklet.length)
    }

def serialize_frame(frame):
    """Convert Frame object to dictionary"""
    return {
        'detections': [serialize_bbox(det) for det in frame.detections],
        'tracklets': [serialize_tracklet(track) for track in frame.tracklets]
    }

def save_tracking_results(forward_results, backward_results, save_path_prefix, json=False):
    """
    Save forward and backward tracking results to files.
    
    Args:
        forward_results: List of Frame objects from forward tracking
        backward_results: List of Frame objects from backward tracking
        save_path_prefix: String prefix for save paths (without extension)
    """
    forward_filepath = os.path.join(save_path_prefix, 'forward.pkl')
    backward_filepath = os.path.join(save_path_prefix, 'backward.pkl')
    if json:
        # Save as JSON for human readability
        forward_data = [serialize_frame(frame) for frame in forward_results]
        backward_data = [serialize_frame(frame) for frame in backward_results]
        
        with open(forward_filepath, 'w') as f:
            json.dump(forward_data, f, indent=2)
        
        with open(backward_filepath, 'w') as f:
            json.dump(backward_data, f, indent=2)
        
    # Also save as pickle for easier loading in Python
    with open(forward_filepath, 'wb') as f:
        pickle.dump(forward_results, f)
        
    with open(backward_filepath, 'wb') as f:
        pickle.dump(backward_results, f)
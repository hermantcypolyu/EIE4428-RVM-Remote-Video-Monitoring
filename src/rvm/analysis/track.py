"""Object tracking using centroid-based tracking and appearance features.

Extends motion detection with persistent ID assignment across frames. Uses
IoU (Intersection over Union) matching for fast re-association and maintains
track history for trajectory analysis.

Object tracking
---------------
Given detections from each frame (bounding boxes), the tracker assigns
consistent IDs across time. This is essential for:
* Counting distinct objects (people, vehicles)
* Trajectory analysis and behavior understanding
* Alert suppression (ignore tracked objects that are stationary)

The tracker uses centroid distance and IoU for matching, keeping computational
cost low enough for real-time operation alongside the codec.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

__all__ = [
    "Track",
    "TrackedObject",
    "CentroidTracker",
]


@dataclass
class Track:
    """A single tracked object with history."""

    track_id: int
    centroid: Tuple[float, float]
    bbox: Tuple[int, int, int, int]
    age: int = 1
    disappeared: int = 0
    trajectory: List[Tuple[float, float]] = field(default_factory=list)

    def update_centroid(self, new_centroid: Tuple[float, float]) -> None:
        """Update centroid and add to trajectory history."""
        self.centroid = new_centroid
        self.trajectory.append(new_centroid)
        if len(self.trajectory) > 30:  # Keep last 30 positions only
            self.trajectory.pop(0)
        self.age += 1
        self.disappeared = 0

    def snapshot(self) -> dict:
        """JSON-friendly view for dashboard."""
        top, left, bottom, right = self.bbox
        return {
            "track_id": self.track_id,
            "centroid": [round(self.centroid[0], 1), round(self.centroid[1], 1)],
            "bbox": [int(top), int(left), int(bottom), int(right)],
            "age": self.age,
            "disappeared": self.disappeared,
        }


@dataclass
class TrackedObject:
    """Collection of tracked objects in current frame."""

    tracks: Dict[int, Track] = field(default_factory=dict)

    @property
    def count(self) -> int:
        return len(self.tracks)

    def snapshot(self) -> dict:
        return {
            "count": self.count,
            "tracks": [track.snapshot() for track in self.tracks.values()],
        }


class CentroidTracker:
    """Centroid-based object tracker with IoU matching."""

    def __init__(self, max_disappeared: int = 10, iou_threshold: float = 0.3) -> None:
        """
        Args:
            max_disappeared: Frames to keep a track alive without detections
            iou_threshold: Minimum IoU for bbox matching
        """
        self.max_disappeared = max_disappeared
        self.iou_threshold = iou_threshold
        self.next_object_id = 0
        self.tracks: Dict[int, Track] = {}

    def reset(self) -> None:
        """Clear all tracks."""
        self.tracks.clear()
        self.next_object_id = 0

    @staticmethod
    def _compute_centroid(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
        """Calculate centroid from (top, left, bottom, right) bbox."""
        top, left, bottom, right = bbox
        cx = (left + right) / 2.0
        cy = (top + bottom) / 2.0
        return (cx, cy)

    @staticmethod
    def _compute_iou(box1: Tuple[int, int, int, int],
                      box2: Tuple[int, int, int, int]) -> float:
        """Compute IoU between two bboxes."""
        t1, l1, b1, r1 = box1
        t2, l2, b2, r2 = box2

        xi1 = max(l1, l2)
        yi1 = max(t1, t2)
        xi2 = min(r1, r2)
        yi2 = min(b1, b2)

        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area = (r1 - l1) * (b1 - t1)
        box2_area = (r2 - l2) * (b2 - t2)
        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

    def update(self, detections: List[Tuple[int, int, int, int]]) -> TrackedObject:
        """
        Update tracks with new detections from current frame.

        Args:
            detections: List of bounding boxes as (top, left, bottom, right)

        Returns:
            TrackedObject containing all active tracks
        """
        # If no detections and no existing tracks, return empty
        if len(detections) == 0 and len(self.tracks) == 0:
            return TrackedObject({})

        # If no detections but have existing tracks, increment disappeared
        if len(detections) == 0:
            for track_id in list(self.tracks.keys()):
                self.tracks[track_id].disappeared += 1
                if self.tracks[track_id].disappeared > self.max_disappeared:
                    del self.tracks[track_id]
            return TrackedObject(dict(self.tracks))

        # Compute centroids for new detections
        input_centroids = [self._compute_centroid(bbox) for bbox in detections]

        # If no existing tracks, register all as new
        if len(self.tracks) == 0:
            for i, bbox in enumerate(detections):
                self._register(input_centroids[i], bbox)
        else:
            # Match existing tracks to new detections using IoU
            track_ids = list(self.tracks.keys())
            track_bboxes = [self.tracks[tid].bbox for tid in track_ids]

            # Build cost matrix based on IoU
            iou_matrix = np.zeros((len(track_bboxes), len(detections)))
            for i, track_bbox in enumerate(track_bboxes):
                for j, det_bbox in enumerate(detections):
                    iou_matrix[i, j] = self._compute_iou(track_bbox, det_bbox)

            # Greedy matching: sort by IoU descending
            used_rows = set()
            used_cols = set()
            matches = []

            # Get all IoU scores sorted descending
            indices = np.unravel_index(np.argsort(iou_matrix.ravel())[::-1],
                                      iou_matrix.shape)
            for row, col in zip(indices[0], indices[1]):
                if row in used_rows or col in used_cols:
                    continue
                if iou_matrix[row, col] < self.iou_threshold:
                    break
                matches.append((row, col))
                used_rows.add(row)
                used_cols.add(col)

            # Update matched tracks
            for row, col in matches:
                track_id = track_ids[row]
                self.tracks[track_id].bbox = detections[col]
                self.tracks[track_id].update_centroid(input_centroids[col])

            # Mark unmatched tracks as disappeared
            unused_rows = set(range(len(track_ids))) - used_rows
            for row in unused_rows:
                track_id = track_ids[row]
                self.tracks[track_id].disappeared += 1
                if self.tracks[track_id].disappeared > self.max_disappeared:
                    del self.tracks[track_id]

            # Register new detections
            unused_cols = set(range(len(detections))) - used_cols
            for col in unused_cols:
                self._register(input_centroids[col], detections[col])

        return TrackedObject(dict(self.tracks))

    def _register(self, centroid: Tuple[float, float],
                  bbox: Tuple[int, int, int, int]) -> None:
        """Register a new track."""
        track = Track(
            track_id=self.next_object_id,
            centroid=centroid,
            bbox=bbox,
        )
        # Initialize trajectory with first position
        track.trajectory = [centroid]
        self.tracks[self.next_object_id] = track
        self.next_object_id += 1

    def status(self) -> dict:
        """Return tracker status."""
        return {
            "active_tracks": len(self.tracks),
            "next_id": self.next_object_id,
            "max_disappeared": self.max_disappeared,
        }


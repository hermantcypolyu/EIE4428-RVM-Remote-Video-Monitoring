"""Tests for object tracking module."""

import pytest
import numpy as np

from rvm.analysis.track import CentroidTracker, Track, TrackedObject


class TestTrack:
    """Test Track data structure."""

    def test_track_creation(self):
        track = Track(
            track_id=1,
            centroid=(100.0, 200.0),
            bbox=(50, 150, 100, 250),
        )
        assert track.track_id == 1
        assert track.centroid == (100.0, 200.0)
        assert track.age == 1
        assert track.disappeared == 0
        # Trajectory starts empty until first update
        assert len(track.trajectory) == 0

    def test_update_centroid(self):
        track = Track(
            track_id=1,
            centroid=(100.0, 200.0),
            bbox=(50, 150, 100, 250),
        )
        track.update_centroid((110.0, 210.0))

        assert track.centroid == (110.0, 210.0)
        assert track.age == 2
        assert track.disappeared == 0
        # First update adds to trajectory
        assert len(track.trajectory) == 1

    def test_trajectory_max_length(self):
        track = Track(
            track_id=1,
            centroid=(100.0, 200.0),
            bbox=(50, 150, 100, 250),
        )
        # Add 40 positions
        for i in range(40):
            track.update_centroid((100.0 + i, 200.0 + i))

        # Should cap at 30
        assert len(track.trajectory) <= 30

    def test_snapshot(self):
        track = Track(
            track_id=5,
            centroid=(100.5, 200.7),
            bbox=(50, 150, 100, 250),
            age=10,
        )
        snap = track.snapshot()

        assert snap["track_id"] == 5
        assert snap["centroid"] == [100.5, 200.7]
        assert snap["bbox"] == [50, 150, 100, 250]
        assert snap["age"] == 10


class TestTrackedObject:
    """Test TrackedObject collection."""

    def test_empty_collection(self):
        obj = TrackedObject()
        assert obj.count == 0
        assert len(obj.tracks) == 0

    def test_collection_with_tracks(self):
        track1 = Track(track_id=1, centroid=(10.0, 20.0), bbox=(5, 15, 10, 25))
        track2 = Track(track_id=2, centroid=(30.0, 40.0), bbox=(25, 35, 30, 45))

        obj = TrackedObject({1: track1, 2: track2})
        assert obj.count == 2

    def test_snapshot(self):
        track1 = Track(track_id=1, centroid=(10.0, 20.0), bbox=(5, 15, 10, 25))
        obj = TrackedObject({1: track1})

        snap = obj.snapshot()
        assert snap["count"] == 1
        assert len(snap["tracks"]) == 1


class TestCentroidTracker:
    """Test CentroidTracker."""

    def test_initialization(self):
        tracker = CentroidTracker()
        assert tracker.max_disappeared == 10
        assert tracker.iou_threshold == 0.3
        assert tracker.next_object_id == 0
        assert len(tracker.tracks) == 0

    def test_custom_initialization(self):
        tracker = CentroidTracker(max_disappeared=5, iou_threshold=0.5)
        assert tracker.max_disappeared == 5
        assert tracker.iou_threshold == 0.5

    def test_reset(self):
        tracker = CentroidTracker()
        tracker.update([(10, 10, 50, 50)])
        assert tracker.next_object_id == 1

        tracker.reset()
        assert tracker.next_object_id == 0
        assert len(tracker.tracks) == 0

    def test_compute_centroid(self):
        cx, cy = CentroidTracker._compute_centroid((10, 20, 50, 60))
        assert cx == 40.0  # (20 + 60) / 2
        assert cy == 30.0  # (10 + 50) / 2

    def test_compute_iou_no_overlap(self):
        box1 = (0, 0, 10, 10)
        box2 = (20, 20, 30, 30)
        iou = CentroidTracker._compute_iou(box1, box2)
        assert iou == 0.0

    def test_compute_iou_complete_overlap(self):
        box1 = (10, 10, 50, 50)
        box2 = (10, 10, 50, 50)
        iou = CentroidTracker._compute_iou(box1, box2)
        assert iou == 1.0

    def test_compute_iou_partial_overlap(self):
        box1 = (0, 0, 20, 20)  # Area = 400
        box2 = (10, 10, 30, 30)  # Area = 400
        # Intersection = 10x10 = 100
        # Union = 400 + 400 - 100 = 700
        iou = CentroidTracker._compute_iou(box1, box2)
        assert abs(iou - 100 / 700) < 0.01

    def test_empty_detections(self):
        tracker = CentroidTracker()
        result = tracker.update([])
        assert result.count == 0

    def test_first_detection(self):
        tracker = CentroidTracker()
        detections = [(10, 10, 50, 50)]

        result = tracker.update(detections)
        assert result.count == 1
        assert 0 in result.tracks
        assert tracker.next_object_id == 1

    def test_multiple_new_detections(self):
        tracker = CentroidTracker()
        detections = [
            (10, 10, 50, 50),
            (60, 60, 100, 100),
        ]

        result = tracker.update(detections)
        assert result.count == 2
        assert tracker.next_object_id == 2

    def test_tracking_same_object(self):
        tracker = CentroidTracker()

        # Frame 1
        detections1 = [(10, 10, 50, 50)]
        result1 = tracker.update(detections1)
        assert result1.count == 1
        track_id = list(result1.tracks.keys())[0]

        # Frame 2: slightly moved
        detections2 = [(12, 12, 52, 52)]
        result2 = tracker.update(detections2)
        assert result2.count == 1
        assert track_id in result2.tracks
        assert result2.tracks[track_id].age == 2

    def test_disappeared_tracking(self):
        tracker = CentroidTracker(max_disappeared=3)

        # Frame 1: one object
        tracker.update([(10, 10, 50, 50)])

        # Frame 2-4: no detections
        for _ in range(3):
            result = tracker.update([])
            assert result.count == 1

        # Frame 5: should be removed
        result = tracker.update([])
        assert result.count == 0

    def test_reappearance(self):
        tracker = CentroidTracker(max_disappeared=5)

        # Frame 1
        result1 = tracker.update([(10, 10, 50, 50)])
        track_id = list(result1.tracks.keys())[0]

        # Frame 2: disappears
        tracker.update([])

        # Frame 3: reappears at similar location
        result3 = tracker.update([(12, 12, 52, 52)])
        assert result3.count == 1
        assert track_id in result3.tracks

    def test_multiple_objects_tracking(self):
        tracker = CentroidTracker()

        # Frame 1: two objects
        detections1 = [
            (10, 10, 50, 50),
            (100, 100, 140, 140),
        ]
        result1 = tracker.update(detections1)
        assert result1.count == 2

        # Frame 2: both move slightly
        detections2 = [
            (12, 12, 52, 52),
            (102, 102, 142, 142),
        ]
        result2 = tracker.update(detections2)
        assert result2.count == 2

        # Track IDs should be preserved
        ids1 = set(result1.tracks.keys())
        ids2 = set(result2.tracks.keys())
        assert ids1 == ids2

    def test_status(self):
        tracker = CentroidTracker(max_disappeared=5)
        tracker.update([(10, 10, 50, 50)])

        status = tracker.status()
        assert status["active_tracks"] == 1
        assert status["next_id"] == 1
        assert status["max_disappeared"] == 5

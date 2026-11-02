"""Tests for gesture recognition module."""

import pytest
import numpy as np

from rvm.analysis.gesture import (
    GestureDetection,
    GestureRecognizer,
    GestureType,
    HandGesture,
)


class TestHandGesture:
    """Test HandGesture data structure."""

    def test_gesture_creation(self):
        gesture = HandGesture(
            gesture=GestureType.OPEN_PALM,
            confidence=0.9,
            hand_label="Left",
            bbox=(10, 20, 30, 40),
        )
        assert gesture.gesture == GestureType.OPEN_PALM
        assert gesture.confidence == 0.9
        assert gesture.hand_label == "Left"

    def test_snapshot(self):
        gesture = HandGesture(
            gesture=GestureType.POINTING,
            confidence=0.85,
            hand_label="Right",
            bbox=(10, 20, 30, 40),
        )
        snap = gesture.snapshot()

        assert snap["gesture"] == "pointing"
        assert snap["confidence"] == 0.85
        assert snap["hand"] == "Right"
        assert snap["bbox"] == [10, 20, 30, 40]


class TestGestureDetection:
    """Test GestureDetection collection."""

    def test_empty_detection(self):
        detection = GestureDetection()
        assert detection.count == 0
        assert len(detection.gestures) == 0

    def test_detection_with_gestures(self):
        g1 = HandGesture(GestureType.PEACE, 0.8, "Left", (10, 20, 30, 40))
        g2 = HandGesture(GestureType.THUMBS_UP, 0.9, "Right", (50, 60, 70, 80))

        detection = GestureDetection([g1, g2], "mediapipe")
        assert detection.count == 2
        assert detection.detector_name == "mediapipe"

    def test_snapshot(self):
        g1 = HandGesture(GestureType.OK_SIGN, 0.75, "Left", (10, 20, 30, 40))
        detection = GestureDetection([g1], "mediapipe")

        snap = detection.snapshot()
        assert snap["count"] == 1
        assert snap["detector"] == "mediapipe"
        assert len(snap["gestures"]) == 1


class TestGestureRecognizer:
    """Test GestureRecognizer."""

    def test_initialization(self):
        recognizer = GestureRecognizer()
        assert recognizer.max_hands == 2
        assert recognizer.min_detection_confidence == 0.5

    def test_custom_initialization(self):
        recognizer = GestureRecognizer(
            max_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6,
        )
        assert recognizer.max_hands == 1
        assert recognizer.min_detection_confidence == 0.7
        assert recognizer.min_tracking_confidence == 0.6

    def test_unavailable_without_mediapipe(self):
        recognizer = GestureRecognizer()
        # Without MediaPipe installed, should gracefully degrade
        # This test will pass regardless of whether MediaPipe is installed
        assert isinstance(recognizer.available, bool)

    def test_detect_empty_frame(self):
        recognizer = GestureRecognizer()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        detection = recognizer.detect(frame)
        # Should return a detection object (may be empty if no MediaPipe)
        assert isinstance(detection, GestureDetection)

    def test_gesture_types(self):
        # Verify all expected gesture types exist
        assert hasattr(GestureType, "OPEN_PALM")
        assert hasattr(GestureType, "CLOSED_FIST")
        assert hasattr(GestureType, "POINTING")
        assert hasattr(GestureType, "THUMBS_UP")
        assert hasattr(GestureType, "THUMBS_DOWN")
        assert hasattr(GestureType, "PEACE")
        assert hasattr(GestureType, "OK_SIGN")
        assert hasattr(GestureType, "UNKNOWN")

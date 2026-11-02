"""Hand gesture recognition using MediaPipe or OpenCV.

Gesture recognition
-------------------
Detects and classifies hand gestures for non-contact control and interaction
monitoring. Uses hand landmark detection to identify poses like:
- Open palm (stop/alert)
- Closed fist
- Pointing
- Thumbs up/down
- Peace sign

Designed to operate in real-time alongside the video codec without blocking
the encode pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np

__all__ = [
    "GestureType",
    "HandGesture",
    "GestureDetection",
    "GestureRecognizer",
]


class GestureType(str, Enum):
    """Recognized gesture types."""

    OPEN_PALM = "open_palm"
    CLOSED_FIST = "closed_fist"
    POINTING = "pointing"
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    PEACE = "peace_sign"
    OK_SIGN = "ok_sign"
    UNKNOWN = "unknown"


@dataclass
class HandGesture:
    """A single detected hand gesture."""

    gesture: GestureType
    confidence: float
    hand_label: str  # "Left" or "Right"
    bbox: Tuple[int, int, int, int]
    landmarks: Optional[List[Tuple[float, float]]] = None

    def snapshot(self) -> dict:
        x, y, w, h = self.bbox
        return {
            "gesture": self.gesture.value,
            "confidence": round(self.confidence, 3),
            "hand": self.hand_label,
            "bbox": [int(x), int(y), int(w), int(h)],
        }


@dataclass
class GestureDetection:
    """Collection of detected gestures in a frame."""

    gestures: List[HandGesture] = field(default_factory=list)
    detector_name: str = "none"

    @property
    def count(self) -> int:
        return len(self.gestures)

    def snapshot(self) -> dict:
        return {
            "count": self.count,
            "gestures": [g.snapshot() for g in self.gestures],
            "detector": self.detector_name,
        }


class GestureRecognizer:
    """Hand gesture recognizer using MediaPipe Hands."""

    def __init__(
        self,
        max_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        """
        Args:
            max_hands: Maximum number of hands to detect
            min_detection_confidence: Minimum confidence for hand detection
            min_tracking_confidence: Minimum confidence for hand tracking
        """
        self.max_hands = max_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self._hands = None
        self._load_failed = False
        self._frame_counter = 0

    @property
    def available(self) -> bool:
        if self._hands is None and not self._load_failed:
            self._load_model()
        return self._hands is not None

    def _load_model(self):
        """Load MediaPipe Hands model."""
        if self._hands is not None or self._load_failed:
            return self._hands

        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_tasks
            from mediapipe.tasks.python import vision as mp_vision

            # Create hand landmarker for gesture recognition
            import os
            model_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
                "models",
                "hand_landmarker.task"
            )
            base_options = mp_tasks.BaseOptions(
                model_asset_path=model_path
            )
            options = mp_vision.HandLandmarkerOptions(
                base_options=base_options,
                num_hands=self.max_hands,
                min_hand_detection_confidence=self.min_detection_confidence,
                running_mode=mp_vision.RunningMode.IMAGE
            )
            self._hands = mp_vision.HandLandmarker.create_from_options(options)
            self._frame_counter = 0  # Needed for VIDEO mode
            return self._hands
        except Exception as e:
            print(f"GestureRecognizer model loading failed: {e}")
            import traceback
            traceback.print_exc()
            self._load_failed = True
            return None

    def detect(self, frame: np.ndarray) -> GestureDetection:
        """
        Detect hand gestures in frame.

        Args:
            frame: RGB image from the capture source

        Returns:
            GestureDetection with recognized gestures
        """
        hands = self._load_model()
        if hands is None:
            return GestureDetection([], "unavailable")

        try:
            import cv2
            import mediapipe as mp

            # Frames from rvm.server.source are already RGB; only expand
            # single-channel input so MediaPipe receives an SRGB image.
            if len(frame.shape) == 2:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            else:
                frame_rgb = np.ascontiguousarray(frame)

            # Convert to MediaPipe Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            # Process frame
            results = hands.detect(mp_image)
        except Exception:
            return GestureDetection([], "error")

        if not results.hand_landmarks:
            # Return empty detection instead of fake "no_hands"
            return GestureDetection([], "mediapipe")

        gestures = []
        h, w = frame.shape[:2]

        for idx, hand_landmarks in enumerate(results.hand_landmarks):
            # Get hand label (Left/Right)
            hand_label = "Unknown"
            if results.handedness and idx < len(results.handedness):
                hand_label = results.handedness[idx][0].category_name

            # Extract landmark coordinates
            landmarks = []
            x_coords, y_coords = [], []
            for lm in hand_landmarks:
                x, y = int(lm.x * w), int(lm.y * h)
                landmarks.append((lm.x, lm.y))
                x_coords.append(x)
                y_coords.append(y)

            # Compute bounding box
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            bbox = (x_min, y_min, x_max - x_min, y_max - y_min)

            # Classify gesture
            gesture, confidence = self._classify_gesture(hand_landmarks)

            gestures.append(
                HandGesture(
                    gesture=gesture,
                    confidence=confidence,
                    hand_label=hand_label,
                    bbox=bbox,
                    landmarks=landmarks,
                )
            )

        return GestureDetection(gestures, "mediapipe")

    def _classify_gesture(self, hand_landmarks) -> Tuple[GestureType, float]:
        """
        Classify gesture from hand landmarks using geometric rules.

        Landmark indices (MediaPipe Hands):
        0: Wrist
        4: Thumb tip, 8: Index tip, 12: Middle tip, 16: Ring tip, 20: Pinky tip
        """
        # Landmarks are now a list of NormalizedLandmark objects
        thumb_tip = hand_landmarks[4]
        index_tip = hand_landmarks[8]
        middle_tip = hand_landmarks[12]
        ring_tip = hand_landmarks[16]
        pinky_tip = hand_landmarks[20]

        # Finger base/MCP indices
        thumb_ip = hand_landmarks[3]
        index_mcp = hand_landmarks[5]
        middle_mcp = hand_landmarks[9]
        ring_mcp = hand_landmarks[13]
        pinky_mcp = hand_landmarks[17]

        wrist = hand_landmarks[0]

        # Check if fingers are extended
        thumb_extended = thumb_tip.x < thumb_ip.x - 0.02  # Simplified
        index_extended = index_tip.y < index_mcp.y
        middle_extended = middle_tip.y < middle_mcp.y
        ring_extended = ring_tip.y < ring_mcp.y
        pinky_extended = pinky_tip.y < pinky_mcp.y

        extended_count = sum(
            [index_extended, middle_extended, ring_extended, pinky_extended]
        )

        # Gesture classification rules
        confidence = 0.8  # Default confidence

        # Open palm: all fingers extended
        if extended_count == 4 and thumb_extended:
            return GestureType.OPEN_PALM, 0.9

        # Closed fist: no fingers extended
        if extended_count == 0:
            return GestureType.CLOSED_FIST, 0.9

        # Pointing: only index extended
        if index_extended and extended_count == 1:
            return GestureType.POINTING, 0.85

        # Peace sign: index and middle extended
        if index_extended and middle_extended and extended_count == 2:
            return GestureType.PEACE, 0.85

        # Thumbs up: thumb extended, others closed
        if thumb_extended and extended_count == 0:
            # Check if thumb is pointing up
            if thumb_tip.y < wrist.y:
                return GestureType.THUMBS_UP, 0.8
            else:
                return GestureType.THUMBS_DOWN, 0.8

        # OK sign: thumb and index form circle
        thumb_index_dist = np.sqrt(
            (thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2
        )
        if thumb_index_dist < 0.05 and extended_count >= 2:
            return GestureType.OK_SIGN, 0.75

        return GestureType.UNKNOWN, 0.5

    def reset(self) -> None:
        """Reset the recognizer (clear tracking state)."""
        if self._hands is not None:
            try:
                self._hands.close()
            except Exception:
                pass
        self._hands = None
        self._load_failed = False

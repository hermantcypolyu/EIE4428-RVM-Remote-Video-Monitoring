"""Analysis modules: detection, recognition, tracking, and gesture analysis."""

from .detect import (
    FaceDetection,
    FaceDetector,
    MotionDetector,
    MotionResult,
    sensitivity_settings,
)
from .quality import psnr, ssim
from .recognition import (
    EmotionDetection,
    EmotionRecognizer,
    FaceRecognition,
    FaceRecognizer,
)
from .gesture import (
    GestureDetection,
    GestureRecognizer,
    GestureType,
    HandGesture,
)
from .track import (
    CentroidTracker,
    Track,
    TrackedObject,
)

__all__ = [
    # Detection
    "FaceDetection",
    "FaceDetector",
    "MotionDetector",
    "MotionResult",
    "sensitivity_settings",
    # Quality
    "psnr",
    "ssim",
    # Recognition
    "EmotionDetection",
    "EmotionRecognizer",
    "FaceRecognition",
    "FaceRecognizer",
    # Gesture
    "GestureDetection",
    "GestureRecognizer",
    "GestureType",
    "HandGesture",
    # Tracking
    "CentroidTracker",
    "Track",
    "TrackedObject",
]

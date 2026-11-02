"""Face recognition and emotion detection.

Face recognition
----------------
Goes beyond detection to identify known individuals. Uses face embeddings
for comparison, maintaining a database of known faces. Gracefully degrades
when models are unavailable.

Emotion recognition
-------------------
Classifies facial expressions into emotional states. Uses either deep learning
models (when available) or falls back to geometric features. Designed to run
on each frame without blocking the video pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

__all__ = [
    "FaceRecognition",
    "FaceRecognizer",
    "EmotionDetection",
    "EmotionRecognizer",
]


def _ensure_rgb(frame: np.ndarray) -> np.ndarray:
    """Return an RGB view of *frame*.

    Frames arriving from :mod:`rvm.server.source` are already RGB, so the
    historical ``BGR2RGB`` conversion silently swapped red and blue and
    destroyed MediaPipe's colour cues.  Callers that still hand over BGR
    arrays can pass ``frame[:, :, ::-1]``.
    """
    if len(frame.shape) == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    return np.ascontiguousarray(frame)


@dataclass
class FaceRecognition:
    """Recognized faces with identity and confidence."""

    identities: List[Tuple[str, float, Tuple[int, int, int, int]]] = field(
        default_factory=list
    )  # (name, confidence, bbox)
    recognizer_name: str = "none"

    @property
    def count(self) -> int:
        return len(self.identities)

    def snapshot(self) -> dict:
        return {
            "count": self.count,
            "identities": [
                {
                    "name": name,
                    "confidence": round(conf, 3),
                    "bbox": list(bbox),
                }
                for name, conf, bbox in self.identities
            ],
            "recognizer": self.recognizer_name,
        }


class FaceRecognizer:
    """Face recognition using MediaPipe FaceLandmarker."""

    def __init__(self, threshold: float = 0.6) -> None:
        """
        Args:
            threshold: Confidence threshold for recognition (0-1)
        """
        self.threshold = threshold
        self.known_faces: Dict[str, np.ndarray] = {}
        self._model = None
        self._load_failed = False
        self._frame_counter = 0

    @property
    def available(self) -> bool:
        if self._model is None and not self._load_failed:
            self._load_model()
        return self._model is not None

    def _load_model(self):
        """Load MediaPipe FaceLandmarker model."""
        if self._model is not None or self._load_failed:
            return self._model

        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_tasks
            from mediapipe.tasks.python import vision as mp_vision
            import os

            model_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
                "models",
                "face_landmarker.task"
            )

            base_options = mp_tasks.BaseOptions(model_asset_path=model_path)
            options = mp_vision.FaceLandmarkerOptions(
                base_options=base_options,
                num_faces=5,
                min_face_detection_confidence=0.5,
                running_mode=mp_vision.RunningMode.IMAGE
            )
            self._model = mp_vision.FaceLandmarker.create_from_options(options)
            return self._model
        except Exception as e:
            print(f"FaceRecognizer model loading failed: {e}")
            import traceback
            traceback.print_exc()
            self._load_failed = True
            return None

    def add_known_face(self, name: str, embedding: np.ndarray) -> None:
        """Register a known face by its embedding vector."""
        self.known_faces[name] = np.array(embedding, dtype=np.float64)

    def remove_known_face(self, name: str) -> bool:
        """Remove a known face from the database."""
        if name in self.known_faces:
            del self.known_faces[name]
            return True
        return False

    def recognize(
        self, frame: np.ndarray, face_boxes: List[Tuple[int, int, int, int]]
    ) -> FaceRecognition:
        """
        Recognize faces in the frame using MediaPipe landmarks.

        Args:
            frame: RGB or grayscale image
            face_boxes: List of face bounding boxes (x, y, w, h) - ignored, we detect internally

        Returns:
            FaceRecognition with identified faces
        """
        model = self._load_model()
        if model is None:
            return FaceRecognition([], "unavailable")

        try:
            import mediapipe as mp

            rgb_frame = _ensure_rgb(frame)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # Detect faces with landmarks
            result = self._model.detect(mp_image)

            if not result.face_landmarks:
                return FaceRecognition([], "no-faces")

            # If no known faces in database, return all detected faces as "unknown"
            h, w = frame.shape[:2]
            identities = []

            if not self.known_faces:
                for i, face_landmarks in enumerate(result.face_landmarks):
                    # Calculate bounding box from landmarks
                    x_coords = [lm.x for lm in face_landmarks]
                    y_coords = [lm.y for lm in face_landmarks]
                    x_min = int(min(x_coords) * w)
                    y_min = int(min(y_coords) * h)
                    x_max = int(max(x_coords) * w)
                    y_max = int(max(y_coords) * h)
                    identities.append((f"unknown_{i+1}", 0.5, (x_min, y_min, x_max - x_min, y_max - y_min)))
                return FaceRecognition(identities, "mediapipe")

            # Compare detected faces with known faces using landmark embeddings
            for face_landmarks in result.face_landmarks:
                # Calculate bounding box
                x_coords = [lm.x for lm in face_landmarks]
                y_coords = [lm.y for lm in face_landmarks]
                x_min = int(min(x_coords) * w)
                y_min = int(min(y_coords) * h)
                x_max = int(max(x_coords) * w)
                y_max = int(max(y_coords) * h)

                # Compute embedding from landmarks
                embedding = self._compute_embedding_from_landmarks(face_landmarks)
                if embedding is None:
                    continue

                # Compare with known faces
                best_match = None
                best_distance = float("inf")

                for name, known_emb in self.known_faces.items():
                    distance = np.linalg.norm(embedding - known_emb)
                    if distance < best_distance:
                        best_distance = distance
                        best_match = name

                # Convert distance to confidence (inverse relationship)
                confidence = max(0.0, 1.0 - best_distance)
                if confidence >= self.threshold and best_match:
                    identities.append((best_match, confidence, (x_min, y_min, x_max - x_min, y_max - y_min)))

            return FaceRecognition(identities, "mediapipe")
        except Exception:
            return FaceRecognition([], "unavailable")

    def _compute_embedding_from_landmarks(self, face_landmarks) -> Optional[np.ndarray]:
        """Compute face embedding vector from MediaPipe landmarks."""
        try:
            # Flatten landmark coordinates into embedding vector
            embedding = []
            for landmark in face_landmarks:
                embedding.extend([landmark.x, landmark.y, landmark.z])
            return np.array(embedding, dtype=np.float64)
        except Exception:
            return None


@dataclass
class EmotionDetection:
    """Detected emotions with confidence scores."""

    emotions: List[Tuple[str, float, Tuple[int, int, int, int]]] = field(
        default_factory=list
    )  # (emotion, confidence, bbox)
    detector_name: str = "none"

    @property
    def count(self) -> int:
        return len(self.emotions)

    def snapshot(self) -> dict:
        return {
            "count": self.count,
            "emotions": [
                {
                    "emotion": emotion,
                    "confidence": round(conf, 3),
                    "bbox": list(bbox),
                }
                for emotion, conf, bbox in self.emotions
            ],
            "detector": self.detector_name,
        }


class EmotionRecognizer:
    """Emotion recognition using MediaPipe FaceLandmarker blendshapes."""

    EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

    def __init__(self) -> None:
        self._model = None
        self._load_failed = False
        self._frame_counter = 0

    @property
    def available(self) -> bool:
        if self._model is None and not self._load_failed:
            self._load_model()
        return self._model is not None

    def _load_model(self):
        """Load MediaPipe FaceLandmarker model with blendshapes enabled."""
        if self._model is not None or self._load_failed:
            return self._model

        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_tasks
            from mediapipe.tasks.python import vision as mp_vision
            import os

            model_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
                "models",
                "face_landmarker.task"
            )

            base_options = mp_tasks.BaseOptions(model_asset_path=model_path)
            options = mp_vision.FaceLandmarkerOptions(
                base_options=base_options,
                num_faces=5,
                min_face_detection_confidence=0.5,
                output_face_blendshapes=True,
                running_mode=mp_vision.RunningMode.IMAGE
            )
            self._model = mp_vision.FaceLandmarker.create_from_options(options)
            return self._model
        except Exception as e:
            print(f"EmotionRecognizer model loading failed: {e}")
            import traceback
            traceback.print_exc()
            self._load_failed = True
            return None

    def detect(
        self, frame: np.ndarray, face_boxes: List[Tuple[int, int, int, int]]
    ) -> EmotionDetection:
        """
        Detect emotions using MediaPipe FaceLandmarker blendshapes.

        Args:
            frame: RGB or grayscale image
            face_boxes: List of face bounding boxes (x, y, w, h) - not used with MediaPipe

        Returns:
            EmotionDetection with recognized emotions
        """
        model = self._load_model()
        if model is None:
            return EmotionDetection([], "unavailable")

        try:
            import mediapipe as mp

            frame_rgb = _ensure_rgb(frame)

            # Create MediaPipe Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            # Detect faces and get blendshapes
            result = model.detect(mp_image)

            if not result.face_landmarks:
                return EmotionDetection([], "no-faces")

            emotions = []
            for i, face_landmarks in enumerate(result.face_landmarks):
                # Get blendshapes for this face
                if i < len(result.face_blendshapes):
                    blendshapes = result.face_blendshapes[i]
                    emotion, confidence = self._classify_emotion(blendshapes)

                    # Calculate bounding box from landmarks
                    h, w = frame.shape[:2]
                    x_coords = [lm.x * w for lm in face_landmarks]
                    y_coords = [lm.y * h for lm in face_landmarks]
                    x_min, x_max = int(min(x_coords)), int(max(x_coords))
                    y_min, y_max = int(min(y_coords)), int(max(y_coords))

                    # Add padding
                    padding = 20
                    x_min = max(0, x_min - padding)
                    y_min = max(0, y_min - padding)
                    x_max = min(w, x_max + padding)
                    y_max = min(h, y_max + padding)

                    box = (x_min, y_min, x_max - x_min, y_max - y_min)
                    emotions.append((emotion, confidence, box))

            return EmotionDetection(emotions, "mediapipe")

        except Exception as e:
            print(f"EmotionRecognizer detection failed: {e}")
            import traceback
            traceback.print_exc()
            return EmotionDetection([], "error")

    def _classify_emotion(self, blendshapes) -> Tuple[str, float]:
        """
        Classify emotion from MediaPipe face blendshapes.

        Maps blendshape categories to emotions based on facial expression patterns.
        """
        # Extract relevant blendshape scores
        blendshape_dict = {bs.category_name: bs.score for bs in blendshapes}

        # Emotion classification rules based on blendshape combinations
        # Happy: mouth smile, cheek squint
        happy_score = (
            blendshape_dict.get("mouthSmileLeft", 0) * 0.3 +
            blendshape_dict.get("mouthSmileRight", 0) * 0.3 +
            blendshape_dict.get("cheekSquintLeft", 0) * 0.2 +
            blendshape_dict.get("cheekSquintRight", 0) * 0.2
        )

        # Sad: mouth frown, brow down
        sad_score = (
            blendshape_dict.get("mouthFrownLeft", 0) * 0.3 +
            blendshape_dict.get("mouthFrownRight", 0) * 0.3 +
            blendshape_dict.get("browDownLeft", 0) * 0.2 +
            blendshape_dict.get("browDownRight", 0) * 0.2
        )

        # Angry: brow down, jaw forward, mouth press
        angry_score = (
            blendshape_dict.get("browDownLeft", 0) * 0.25 +
            blendshape_dict.get("browDownRight", 0) * 0.25 +
            blendshape_dict.get("jawForward", 0) * 0.25 +
            blendshape_dict.get("mouthPressLeft", 0) * 0.125 +
            blendshape_dict.get("mouthPressRight", 0) * 0.125
        )

        # Surprise: brow outer up, eye wide, jaw open
        surprise_score = (
            blendshape_dict.get("browOuterUpLeft", 0) * 0.25 +
            blendshape_dict.get("browOuterUpRight", 0) * 0.25 +
            blendshape_dict.get("eyeWideLeft", 0) * 0.2 +
            blendshape_dict.get("eyeWideRight", 0) * 0.2 +
            blendshape_dict.get("jawOpen", 0) * 0.1
        )

        # Fear: brow inner up, eye wide
        fear_score = (
            blendshape_dict.get("browInnerUp", 0) * 0.3 +
            blendshape_dict.get("eyeWideLeft", 0) * 0.35 +
            blendshape_dict.get("eyeWideRight", 0) * 0.35
        )

        # Disgust: nose sneer, mouth upper up
        disgust_score = (
            blendshape_dict.get("noseSneerLeft", 0) * 0.35 +
            blendshape_dict.get("noseSneerRight", 0) * 0.35 +
            blendshape_dict.get("mouthUpperUpLeft", 0) * 0.15 +
            blendshape_dict.get("mouthUpperUpRight", 0) * 0.15
        )

        # Find dominant emotion
        emotion_scores = {
            "happy": happy_score,
            "sad": sad_score,
            "angry": angry_score,
            "surprise": surprise_score,
            "fear": fear_score,
            "disgust": disgust_score,
        }

        max_emotion = max(emotion_scores.items(), key=lambda x: x[1])

        # If confidence is too low, default to neutral
        if max_emotion[1] < 0.15:
            return "neutral", 0.2

        return max_emotion[0], max_emotion[1]

#!/usr/bin/env python3
"""Demo script showing all extended RVM features.

This demonstrates:
- Object tracking
- Face recognition
- Emotion recognition
- Gesture recognition
- Audio event detection

Run with: python examples/demo_features.py
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cv2
import numpy as np

from rvm.analysis import (
    CentroidTracker,
    EmotionRecognizer,
    FaceDetector,
    FaceRecognizer,
    GestureRecognizer,
    MotionDetector,
)
from rvm.audio import AudioAnalyzer, AudioCapture, AudioConfig


def demo_object_tracking():
    """Demonstrate object tracking."""
    print("\n=== Object Tracking Demo ===")

    tracker = CentroidTracker(max_disappeared=10)

    # Simulate detections across frames
    frame1_detections = [
        (10, 10, 50, 50),  # Object 1
        (100, 100, 150, 150),  # Object 2
    ]

    frame2_detections = [
        (12, 12, 52, 52),  # Object 1 moved
        (102, 102, 152, 152),  # Object 2 moved
    ]

    frame3_detections = [
        (14, 14, 54, 54),  # Object 1 moved
        # Object 2 disappeared
    ]

    print("\nFrame 1:")
    result = tracker.update(frame1_detections)
    for tid, track in result.tracks.items():
        print(f"  Track {tid}: centroid={track.centroid}, age={track.age}")

    print("\nFrame 2:")
    result = tracker.update(frame2_detections)
    for tid, track in result.tracks.items():
        print(f"  Track {tid}: centroid={track.centroid}, age={track.age}")

    print("\nFrame 3:")
    result = tracker.update(frame3_detections)
    for tid, track in result.tracks.items():
        print(
            f"  Track {tid}: centroid={track.centroid}, "
            f"age={track.age}, disappeared={track.disappeared}"
        )


def demo_face_recognition():
    """Demonstrate face recognition."""
    print("\n=== Face Recognition Demo ===")

    recognizer = FaceRecognizer(threshold=0.6)

    if recognizer.available:
        print("Face recognizer available")

        # In a real scenario, you would:
        # 1. Extract face embeddings from known individuals
        # 2. Register them with add_known_face()
        # 3. Call recognize() on new frames

        print("To use face recognition:")
        print("  1. Install: pip install deepface")
        print("  2. Register known faces with embeddings")
        print("  3. Call recognizer.recognize(frame, face_boxes)")
    else:
        print("Face recognizer not available (install deepface or dlib)")


def demo_emotion_recognition():
    """Demonstrate emotion recognition."""
    print("\n=== Emotion Recognition Demo ===")

    recognizer = EmotionRecognizer()

    if recognizer.available:
        print("Emotion recognizer available")
        print(f"Supported emotions: {EmotionRecognizer.EMOTIONS}")
    else:
        print("Emotion recognizer not available (install deepface or fer)")
        print("To enable: pip install deepface")


def demo_gesture_recognition():
    """Demonstrate gesture recognition."""
    print("\n=== Gesture Recognition Demo ===")

    recognizer = GestureRecognizer(max_hands=2)

    if recognizer.available:
        print("Gesture recognizer available")
        print("Supported gestures:")
        print("  - OPEN_PALM: Stop/Alert")
        print("  - CLOSED_FIST: Fist")
        print("  - POINTING: Index finger extended")
        print("  - THUMBS_UP: Thumbs up")
        print("  - THUMBS_DOWN: Thumbs down")
        print("  - PEACE: Peace sign (V)")
        print("  - OK_SIGN: OK gesture")
    else:
        print("Gesture recognizer not available (install mediapipe)")
        print("To enable: pip install mediapipe")


def demo_audio_monitoring():
    """Demonstrate audio event detection."""
    print("\n=== Audio Monitoring Demo ===")

    # Test audio analyzer (does not require PyAudio)
    analyzer = AudioAnalyzer(
        sample_rate=16000, energy_threshold=0.02, silence_threshold=0.005
    )

    print(f"Audio analyzer initialized")
    print(f"Status: {analyzer.get_status()}")

    # Simulate different audio scenarios
    print("\nSimulating audio events:")

    # Silence
    silent_audio = np.zeros(1024, dtype=np.float32)
    events = analyzer.analyze(silent_audio)
    print(f"  Silent audio: {len(events)} events")

    # Loud sound
    loud_audio = np.random.randn(1024).astype(np.float32) * 0.5
    events = analyzer.analyze(loud_audio)
    if events:
        print(
            f"  Loud audio: detected {events[0].event_type.value} "
            f"(energy={events[0].energy:.3f})"
        )

    # Audio capture
    capture = AudioCapture()
    if capture.available:
        print("\nAudio capture available (PyAudio installed)")
        print("To start capturing:")
        print("  capture.start()")
        print("  audio_chunk = capture.read()")
        print("  events = analyzer.analyze(audio_chunk)")
    else:
        print("\nAudio capture not available (install pyaudio)")
        print("To enable: pip install pyaudio")


def demo_integration():
    """Show how features integrate in a monitoring system."""
    print("\n=== Integration Example ===")

    print(
        """
In a complete monitoring system, these features work together:

1. Video capture from camera
2. Motion detection triggers recording
3. Face detection finds people
4. Object tracker assigns IDs to each person
5. Face recognition identifies known individuals
6. Emotion recognition analyzes mood
7. Gesture recognition detects hand signals
8. Audio analysis monitors sound events

Example pipeline:
```python
# Initialize all modules
motion_detector = MotionDetector()
face_detector = FaceDetector()
face_recognizer = FaceRecognizer()
emotion_recognizer = EmotionRecognizer()
gesture_recognizer = GestureRecognizer()
object_tracker = CentroidTracker()
audio_analyzer = AudioAnalyzer()

# Process each frame
for frame in video_stream:
    # 1. Detect motion
    luma = extract_luma(frame)
    motion_result = motion_detector.update(luma)

    if motion_result.alarm:
        # 2. Detect faces
        face_detection = face_detector.detect(luma)

        # 3. Track objects
        bboxes = [face.bbox for face in face_detection.faces]
        tracked = object_tracker.update(bboxes)

        # 4. Recognize faces
        identities = face_recognizer.recognize(frame, bboxes)

        # 5. Detect emotions
        emotions = emotion_recognizer.detect(frame, bboxes)

        # 6. Detect gestures
        gestures = gesture_recognizer.detect(frame)

        # 7. Analyze audio
        audio_chunk = audio_capture.read()
        audio_events = audio_analyzer.analyze(audio_chunk)

        # 8. Send alerts/log events
        process_monitoring_data(tracked, identities, emotions, gestures, audio_events)
```
"""
    )


def main():
    """Run all demos."""
    print("=" * 60)
    print("RVM Extended Features Demo")
    print("=" * 60)

    demo_object_tracking()
    demo_face_recognition()
    demo_emotion_recognition()
    demo_gesture_recognition()
    demo_audio_monitoring()
    demo_integration()

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

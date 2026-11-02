# RVM Implementation Status

Consolidated status of every feature in the project. Core requirements come
from the EIE4428 brief; the extended features are optional and degrade
gracefully when their dependencies are not installed.

## Core requirements (8/8 complete)

| # | Requirement | Module | Status |
|---|---|---|---|
| 1 | Live video streaming | `server/app.py` (MJPEG + WebSocket) | ✅ |
| 2 | Motion detection with alarm | `analysis/detect.py` → `MotionDetector` | ✅ |
| 3 | Face detection | `analysis/detect.py` → `FaceDetector` (Haar cascade) | ✅ |
| 4 | Video recording with timestamps | `server/recorder.py` → `FrameRecorder` | ✅ |
| 5 | Concurrent access | `server/session.py` — one encode, many viewers | ✅ |
| 6 | Browser-based viewing | `web/index.html`, `web/app.js` | ✅ |
| 7 | ROI selection | `POST /api/motion` + dashboard drag box | ✅ |
| 8 | Adjustable sensitivity | three levels (low / medium / high) | ✅ |

## Extended features (5/5 complete)

| # | Feature | Module | Backend | Notes |
|---|---|---|---|---|
| 9 | Object tracking | `analysis/track.py` → `CentroidTracker` | custom (IoU + centroid) | persistent IDs, 30-frame trail, auto-cleanup |
| 10 | Face recognition | `analysis/recognition.py` → `FaceRecognizer` | MediaPipe FaceLandmarker | embedding match against a known-face DB |
| 11 | Emotion recognition | `analysis/recognition.py` → `EmotionRecognizer` | MediaPipe blendshapes | 7 classes: happy, sad, angry, surprise, fear, disgust, neutral |
| 12 | Gesture recognition | `analysis/gesture.py` → `GestureRecognizer` | MediaPipe HandLandmarker | open palm, fist, pointing, thumbs up/down, peace, OK |
| 13 | Audio monitoring | `audio/capture.py`, `audio/analysis.py` | `pyaudio` | loud sound, speech, silence, gunshot, glass break, alarm |

### ML model files

- `models/face_landmarker.task` — face landmarks + blendshapes (shared by
  face recognition and emotion recognition)
- `models/hand_landmarker.task` — 21 hand landmarks per hand

All analyzers **lazy-load** their models on first use and report themselves
as unavailable (rather than crashing) if a model or dependency is missing.

## API endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | Operator dashboard |
| `GET` | `/api/status` | Resolution, fps, bit rate, QP, PSNR, features |
| `GET` | `/api/stream.mjpg` | Live MJPEG stream (`?view=source` / `?view=decoded`) |
| `GET` | `/api/snapshot.jpg` | Single JPEG snapshot |
| `GET` | `/api/metrics` | Full telemetry as JSON |
| `GET` | `/api/clips` | List recorded clips |
| `GET` | `/api/clips/<clip>/<frame>.jpg` | Frame of a stored clip |
| `GET` | `/api/tracks` | Tracked objects with persistent IDs |
| `GET` | `/api/features` | Which optional analyzers are available |
| `GET` | `/api/emotions` | Emotion-recognition results |
| `GET` | `/api/gestures` | Gesture-recognition results |
| `GET` | `/api/faces/identities` | Face-recognition results |
| `GET` | `/api/audio/events` | Recent audio events |
| `POST` | `/api/audio/control` | Start / stop audio capture |
| `POST` | `/api/config` | Change the encoder operating point |
| `POST` | `/api/motion` | Set the detection region and sensitivity |
| `POST` | `/api/record` | Start / stop recording |
| `WS` | `/ws` | Telemetry push + live RTP packet feed |

## Performance characteristics

Measured on the demonstration machine (Intel Core Ultra 9 185H, Python 3.13):

| Stage | Cost per frame |
|---|---|
| Face recognition | ~50 ms (MediaPipe) |
| Emotion recognition | ~50 ms (shares face landmarks) |
| Gesture recognition | ~30 ms (hand detection + classification) |
| Motion detection | < 5 ms |
| Object tracking | < 10 ms |

## Frame-processing pipeline

1. Capture a frame from the camera (or the synthetic scene).
2. Convert to the required colour space (RGB for MediaPipe, grayscale for
   motion / face detection).
3. Run the enabled analyzers.
4. Update trackers with detection results.
5. Optionally record the frame with metadata.
6. Encode and stream to clients.

## Known limitations

- Frame rate is bounded by the Python encoder (about 20 fps at 320×240,
  about 3 fps at 640×480), not by the algorithms.
- The `RVM1` codec is a teaching implementation and is less efficient than
  H.264 (no deblocking, larger transform blocks, single reference frame).
- Audio is captured and analysed locally; it is not streamed to remote
  viewers.

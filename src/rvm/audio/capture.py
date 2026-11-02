"""Audio capture from microphone or system audio.

Audio capture
-------------
Captures audio alongside video for complete monitoring. Uses PyAudio for
cross-platform microphone access. Designed to run in a separate thread
without blocking the video pipeline.

The audio path mirrors the video architecture: capture in a worker, encode
if needed, and stream to viewers. For the mini-project, audio is optional
but demonstrates a complete A/V monitoring system.
"""

from __future__ import annotations

from dataclasses import dataclass
from queue import Queue
from threading import Event, Thread
from typing import Optional

import numpy as np

__all__ = ["AudioConfig", "AudioCapture"]


@dataclass
class AudioConfig:
    """Audio capture configuration."""

    sample_rate: int = 16000  # 16 kHz sufficient for speech
    channels: int = 1  # Mono
    chunk_size: int = 1024  # Samples per read
    format: str = "int16"

    @property
    def chunk_duration_ms(self) -> float:
        """Duration of one chunk in milliseconds."""
        return (self.chunk_size / self.sample_rate) * 1000.0


class AudioCapture:
    """Audio capture from system microphone."""

    def __init__(self, config: Optional[AudioConfig] = None) -> None:
        self.config = config or AudioConfig()
        self._stream = None
        self._pyaudio = None
        self._capture_thread: Optional[Thread] = None
        self._stop_event = Event()
        self._queue: Queue = Queue(maxsize=100)
        self._running = False
        self._load_failed = False

    @property
    def available(self) -> bool:
        """Check if audio capture is available."""
        return self._pyaudio is not None or (
            not self._load_failed and self._init_pyaudio()
        )

    @property
    def running(self) -> bool:
        return self._running

    def _init_pyaudio(self) -> bool:
        """Initialize PyAudio."""
        if self._pyaudio is not None or self._load_failed:
            return self._pyaudio is not None

        try:
            import pyaudio

            self._pyaudio = pyaudio.PyAudio()
            return True
        except Exception:
            self._load_failed = True
            return False

    def start(self) -> bool:
        """
        Start audio capture.

        Returns:
            True if capture started successfully
        """
        if self._running:
            return True

        if not self._init_pyaudio():
            return False

        try:
            import pyaudio

            format_map = {
                "int16": pyaudio.paInt16,
                "int32": pyaudio.paInt32,
                "float32": pyaudio.paFloat32,
            }

            self._stream = self._pyaudio.open(
                format=format_map.get(self.config.format, pyaudio.paInt16),
                channels=self.config.channels,
                rate=self.config.sample_rate,
                input=True,
                frames_per_buffer=self.config.chunk_size,
            )

            self._stop_event.clear()
            self._capture_thread = Thread(target=self._capture_loop, daemon=True)
            self._capture_thread.start()
            self._running = True
            return True

        except Exception:
            return False

    def stop(self) -> None:
        """Stop audio capture."""
        if not self._running:
            return

        self._stop_event.set()
        if self._capture_thread:
            self._capture_thread.join(timeout=2.0)

        if self._stream:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

        self._running = False

    def _capture_loop(self) -> None:
        """Capture loop running in worker thread."""
        while not self._stop_event.is_set():
            try:
                if self._stream and self._stream.is_active():
                    data = self._stream.read(
                        self.config.chunk_size, exception_on_overflow=False
                    )
                    # Convert to numpy array
                    audio_data = np.frombuffer(data, dtype=np.int16)

                    # Put in queue (drop if full to avoid blocking)
                    if not self._queue.full():
                        self._queue.put_nowait(audio_data)
                else:
                    break
            except Exception:
                break

    def read(self, timeout: float = 0.1) -> Optional[np.ndarray]:
        """
        Read next audio chunk.

        Args:
            timeout: Maximum time to wait for data (seconds)

        Returns:
            Audio data as numpy array, or None if no data available
        """
        if not self._running:
            return None

        try:
            return self._queue.get(timeout=timeout)
        except Exception:
            return None

    def clear_buffer(self) -> None:
        """Clear the audio buffer."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except Exception:
                break

    def __del__(self):
        """Cleanup on deletion."""
        self.stop()
        if self._pyaudio:
            try:
                self._pyaudio.terminate()
            except Exception:
                pass

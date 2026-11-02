"""Audio event detection and analysis.

Audio analysis
--------------
Detects acoustic events relevant to monitoring:
- Loud sounds (alarms, breaking glass, shouting)
- Gunshots
- Abnormal silence
- Speech/voice activity detection

Uses signal processing (energy, zero-crossing rate, spectral features) for
fast detection without deep learning models, keeping it lightweight enough
to run alongside the video codec.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

import numpy as np

__all__ = ["EventType", "AudioEvent", "AudioAnalyzer"]


class EventType(str, Enum):
    """Types of audio events."""

    LOUD_SOUND = "loud_sound"
    SPEECH = "speech"
    SILENCE = "silence"
    GUNSHOT = "gunshot"
    GLASS_BREAK = "glass_break"
    ALARM = "alarm"
    UNKNOWN = "unknown"


@dataclass
class AudioEvent:
    """Detected audio event."""

    event_type: EventType
    confidence: float
    timestamp: float  # Seconds since start
    duration: float  # Event duration in seconds
    energy: float  # RMS energy
    metadata: dict = field(default_factory=dict)

    def snapshot(self) -> dict:
        return {
            "type": self.event_type.value,
            "confidence": round(self.confidence, 3),
            "timestamp": round(self.timestamp, 3),
            "duration": round(self.duration, 3),
            "energy": round(self.energy, 3),
            **self.metadata,
        }


class AudioAnalyzer:
    """Audio event detector using signal processing."""

    def __init__(
        self,
        sample_rate: int = 16000,
        energy_threshold: float = 0.02,
        silence_threshold: float = 0.005,
    ) -> None:
        """
        Args:
            sample_rate: Audio sample rate in Hz
            energy_threshold: RMS threshold for loud sound detection
            silence_threshold: RMS threshold below which is silence
        """
        self.sample_rate = sample_rate
        self.energy_threshold = energy_threshold
        self.silence_threshold = silence_threshold
        self._frame_count = 0
        self._history: List[float] = []
        self._max_history = 50  # Keep last 50 frames for context

    def reset(self) -> None:
        """Reset analyzer state."""
        self._frame_count = 0
        self._history.clear()

    def analyze(self, audio_chunk: np.ndarray) -> List[AudioEvent]:
        """
        Analyze audio chunk for events.

        Args:
            audio_chunk: Audio samples as numpy array

        Returns:
            List of detected audio events
        """
        if audio_chunk.size == 0:
            return []

        # Normalize to [-1, 1] if needed
        if audio_chunk.dtype == np.int16:
            audio_float = audio_chunk.astype(np.float32) / 32768.0
        elif audio_chunk.dtype == np.int32:
            audio_float = audio_chunk.astype(np.float32) / 2147483648.0
        else:
            audio_float = audio_chunk.astype(np.float32)

        # Compute features
        rms_energy = self._compute_rms(audio_float)
        zcr = self._compute_zcr(audio_float)
        spectral_centroid = self._compute_spectral_centroid(audio_float)

        # Update history
        self._history.append(rms_energy)
        if len(self._history) > self._max_history:
            self._history.pop(0)

        timestamp = self._frame_count * len(audio_chunk) / self.sample_rate
        duration = len(audio_chunk) / self.sample_rate
        self._frame_count += 1

        events = []

        # Detect loud sound
        if rms_energy > self.energy_threshold:
            confidence = min(1.0, rms_energy / (self.energy_threshold * 2))
            event_type = self._classify_loud_sound(
                rms_energy, zcr, spectral_centroid, audio_float
            )
            events.append(
                AudioEvent(
                    event_type=event_type,
                    confidence=confidence,
                    timestamp=timestamp,
                    duration=duration,
                    energy=rms_energy,
                    metadata={"zcr": zcr, "spectral_centroid": spectral_centroid},
                )
            )

        # Detect silence
        elif rms_energy < self.silence_threshold:
            # Only report if silence follows activity
            recent_avg = np.mean(self._history[-10:]) if len(self._history) >= 10 else 0
            if recent_avg > self.silence_threshold * 2:
                events.append(
                    AudioEvent(
                        event_type=EventType.SILENCE,
                        confidence=0.8,
                        timestamp=timestamp,
                        duration=duration,
                        energy=rms_energy,
                    )
                )

        # Detect speech (moderate energy + high ZCR)
        elif (
            self.silence_threshold < rms_energy < self.energy_threshold
            and zcr > 0.05
        ):
            events.append(
                AudioEvent(
                    event_type=EventType.SPEECH,
                    confidence=0.7,
                    timestamp=timestamp,
                    duration=duration,
                    energy=rms_energy,
                    metadata={"zcr": zcr},
                )
            )
        else:
            # Default ambient sound when no specific event detected
            events.append(
                AudioEvent(
                    event_type=EventType.UNKNOWN,
                    confidence=0.5,
                    timestamp=timestamp,
                    duration=duration,
                    energy=rms_energy,
                    metadata={"classification": "ambient"},
                )
            )

        return events

    def _classify_loud_sound(
        self, energy: float, zcr: float, spectral_centroid: float, audio: np.ndarray
    ) -> EventType:
        """Classify type of loud sound based on features."""
        # Gunshot: very high energy spike, short duration, high frequency content
        if energy > self.energy_threshold * 3 and spectral_centroid > 2000:
            return EventType.GUNSHOT

        # Glass break: high frequency, irregular pattern
        if spectral_centroid > 3000 and zcr > 0.1:
            return EventType.GLASS_BREAK

        # Alarm: sustained high energy, periodic
        if len(self._history) >= 5:
            recent = self._history[-5:]
            if all(e > self.energy_threshold * 0.8 for e in recent):
                return EventType.ALARM

        return EventType.LOUD_SOUND

    @staticmethod
    def _compute_rms(audio: np.ndarray) -> float:
        """Compute RMS energy."""
        return float(np.sqrt(np.mean(audio**2)))

    @staticmethod
    def _compute_zcr(audio: np.ndarray) -> float:
        """Compute zero-crossing rate."""
        signs = np.sign(audio)
        signs[signs == 0] = -1
        crossings = np.sum(np.abs(np.diff(signs))) / 2
        return float(crossings / len(audio))

    def _compute_spectral_centroid(self, audio: np.ndarray) -> float:
        """Compute spectral centroid (brightness measure)."""
        try:
            # Simple FFT-based spectral centroid
            spectrum = np.abs(np.fft.rfft(audio))
            freqs = np.fft.rfftfreq(len(audio), 1.0 / self.sample_rate)

            # Avoid division by zero
            spectrum_sum = np.sum(spectrum)
            if spectrum_sum < 1e-10:
                return 0.0

            centroid = np.sum(freqs * spectrum) / spectrum_sum
            return float(centroid)
        except Exception:
            return 0.0

    def set_thresholds(
        self,
        energy_threshold: Optional[float] = None,
        silence_threshold: Optional[float] = None,
    ) -> None:
        """Update detection thresholds."""
        if energy_threshold is not None:
            self.energy_threshold = float(energy_threshold)
        if silence_threshold is not None:
            self.silence_threshold = float(silence_threshold)

    def get_status(self) -> dict:
        """Get analyzer status."""
        return {
            "sample_rate": self.sample_rate,
            "energy_threshold": self.energy_threshold,
            "silence_threshold": self.silence_threshold,
            "frames_analyzed": self._frame_count,
            "history_length": len(self._history),
        }

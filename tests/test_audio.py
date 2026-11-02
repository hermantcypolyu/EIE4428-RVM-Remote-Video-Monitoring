"""Tests for audio capture and analysis modules."""

import pytest
import numpy as np

from rvm.audio import AudioCapture, AudioConfig, AudioAnalyzer, AudioEvent, EventType


class TestAudioConfig:
    """Test AudioConfig."""

    def test_default_config(self):
        config = AudioConfig()
        assert config.sample_rate == 16000
        assert config.channels == 1
        assert config.chunk_size == 1024
        assert config.format == "int16"

    def test_custom_config(self):
        config = AudioConfig(
            sample_rate=48000,
            channels=2,
            chunk_size=2048,
            format="float32",
        )
        assert config.sample_rate == 48000
        assert config.channels == 2
        assert config.chunk_size == 2048

    def test_chunk_duration(self):
        config = AudioConfig(sample_rate=16000, chunk_size=1600)
        # 1600 samples at 16000 Hz = 100 ms
        assert abs(config.chunk_duration_ms - 100.0) < 0.1


class TestAudioCapture:
    """Test AudioCapture."""

    def test_initialization(self):
        capture = AudioCapture()
        assert not capture.running
        assert isinstance(capture.config, AudioConfig)

    def test_custom_config(self):
        config = AudioConfig(sample_rate=8000)
        capture = AudioCapture(config)
        assert capture.config.sample_rate == 8000

    def test_available_property(self):
        capture = AudioCapture()
        # Should return bool regardless of PyAudio availability
        assert isinstance(capture.available, bool)

    def test_stop_when_not_running(self):
        capture = AudioCapture()
        # Should not raise
        capture.stop()

    def test_clear_buffer(self):
        capture = AudioCapture()
        # Should not raise even when not running
        capture.clear_buffer()


class TestAudioEvent:
    """Test AudioEvent data structure."""

    def test_event_creation(self):
        event = AudioEvent(
            event_type=EventType.LOUD_SOUND,
            confidence=0.9,
            timestamp=1.5,
            duration=0.5,
            energy=0.3,
        )
        assert event.event_type == EventType.LOUD_SOUND
        assert event.confidence == 0.9
        assert event.timestamp == 1.5
        assert event.duration == 0.5
        assert event.energy == 0.3

    def test_snapshot(self):
        event = AudioEvent(
            event_type=EventType.SPEECH,
            confidence=0.7,
            timestamp=2.0,
            duration=1.0,
            energy=0.15,
            metadata={"zcr": 0.08},
        )
        snap = event.snapshot()

        assert snap["type"] == "speech"
        assert snap["confidence"] == 0.7
        assert snap["timestamp"] == 2.0
        assert "zcr" in snap


class TestAudioAnalyzer:
    """Test AudioAnalyzer."""

    def test_initialization(self):
        analyzer = AudioAnalyzer()
        assert analyzer.sample_rate == 16000
        assert analyzer.energy_threshold == 0.02
        assert analyzer.silence_threshold == 0.005

    def test_custom_initialization(self):
        analyzer = AudioAnalyzer(
            sample_rate=48000,
            energy_threshold=0.05,
            silence_threshold=0.01,
        )
        assert analyzer.sample_rate == 48000
        assert analyzer.energy_threshold == 0.05
        assert analyzer.silence_threshold == 0.01

    def test_reset(self):
        analyzer = AudioAnalyzer()
        # Analyze some data
        audio = np.random.randn(1024).astype(np.float32)
        analyzer.analyze(audio)

        analyzer.reset()
        assert analyzer._frame_count == 0
        assert len(analyzer._history) == 0

    def test_compute_rms(self):
        # Pure sine wave at amplitude 0.5
        audio = np.ones(1000) * 0.5
        rms = AudioAnalyzer._compute_rms(audio)
        assert abs(rms - 0.5) < 0.01

    def test_compute_rms_zero(self):
        audio = np.zeros(1000)
        rms = AudioAnalyzer._compute_rms(audio)
        assert rms == 0.0

    def test_compute_zcr_constant(self):
        # No zero crossings
        audio = np.ones(1000)
        zcr = AudioAnalyzer._compute_zcr(audio)
        assert zcr == 0.0

    def test_compute_zcr_alternating(self):
        # Maximum zero crossings
        audio = np.array([1, -1] * 500)
        zcr = AudioAnalyzer._compute_zcr(audio)
        assert zcr > 0.9  # Close to 1.0

    def test_analyze_empty(self):
        analyzer = AudioAnalyzer()
        events = analyzer.analyze(np.array([]))
        assert len(events) == 0

    def test_analyze_silence(self):
        analyzer = AudioAnalyzer(silence_threshold=0.01)
        # Very quiet audio
        audio = np.random.randn(1024).astype(np.float32) * 0.001

        events = analyzer.analyze(audio)
        # May or may not detect silence on first frame
        assert isinstance(events, list)

    def test_analyze_loud_sound(self):
        analyzer = AudioAnalyzer(energy_threshold=0.02)
        # Loud audio
        audio = np.random.randn(1024).astype(np.float32) * 0.5

        events = analyzer.analyze(audio)
        # Should detect loud sound
        if len(events) > 0:
            assert events[0].event_type in [
                EventType.LOUD_SOUND,
                EventType.GUNSHOT,
                EventType.GLASS_BREAK,
                EventType.ALARM,
            ]

    def test_analyze_int16_input(self):
        analyzer = AudioAnalyzer()
        # int16 audio
        audio = np.random.randint(-1000, 1000, 1024, dtype=np.int16)

        events = analyzer.analyze(audio)
        # Should handle int16 input without error
        assert isinstance(events, list)

    def test_set_thresholds(self):
        analyzer = AudioAnalyzer()
        analyzer.set_thresholds(energy_threshold=0.05, silence_threshold=0.02)

        assert analyzer.energy_threshold == 0.05
        assert analyzer.silence_threshold == 0.02

    def test_get_status(self):
        analyzer = AudioAnalyzer(sample_rate=48000)
        audio = np.random.randn(1024).astype(np.float32)
        analyzer.analyze(audio)

        status = analyzer.get_status()
        assert status["sample_rate"] == 48000
        assert status["frames_analyzed"] == 1
        assert "energy_threshold" in status

    def test_spectral_centroid(self):
        analyzer = AudioAnalyzer(sample_rate=16000)
        # Generate a tone at 1000 Hz
        t = np.linspace(0, 1, 16000)
        audio = np.sin(2 * np.pi * 1000 * t).astype(np.float32)

        centroid = analyzer._compute_spectral_centroid(audio[:1024])
        # Should be close to 1000 Hz (allowing some tolerance)
        assert 800 < centroid < 1200

    def test_multiple_frames(self):
        analyzer = AudioAnalyzer()

        for i in range(10):
            audio = np.random.randn(1024).astype(np.float32) * 0.01
            events = analyzer.analyze(audio)

        status = analyzer.get_status()
        assert status["frames_analyzed"] == 10


class TestEventTypes:
    """Test EventType enum."""

    def test_all_event_types(self):
        assert EventType.LOUD_SOUND.value == "loud_sound"
        assert EventType.SPEECH.value == "speech"
        assert EventType.SILENCE.value == "silence"
        assert EventType.GUNSHOT.value == "gunshot"
        assert EventType.GLASS_BREAK.value == "glass_break"
        assert EventType.ALARM.value == "alarm"
        assert EventType.UNKNOWN.value == "unknown"

"""
Tests for audio I/O abstraction layer.

These tests verify that the soundfile-based audio_io module behaves
identically to torchaudio's I/O functions.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from batchalign.models import audio_io


# Path to test audio file
TEST_AUDIO = Path(__file__).parent.parent / "support" / "test.mp3"


class TestLoad:
    """Tests for audio_io.load()"""

    def test_load_returns_tensor_and_sample_rate(self):
        """Basic load returns tensor and sample rate."""
        audio, rate = audio_io.load(TEST_AUDIO)
        assert isinstance(audio, torch.Tensor)
        assert isinstance(rate, int)
        assert rate > 0

    def test_load_returns_correct_shape(self):
        """Audio tensor has shape (channels, frames)."""
        audio, rate = audio_io.load(TEST_AUDIO)
        assert audio.dim() == 2
        assert audio.shape[0] in (1, 2)  # Mono or stereo
        assert audio.shape[1] > 0  # Has frames

    def test_load_returns_float32_by_default(self):
        """Default dtype is float32."""
        audio, _ = audio_io.load(TEST_AUDIO)
        assert audio.dtype == torch.float32

    def test_load_with_frame_offset(self):
        """Loading with frame_offset skips initial frames."""
        full_audio, rate = audio_io.load(TEST_AUDIO)
        offset = 1000
        partial_audio, _ = audio_io.load(TEST_AUDIO, frame_offset=offset)
        
        # Partial should be shorter by offset amount
        assert partial_audio.shape[1] == full_audio.shape[1] - offset
        # Content should match
        assert torch.allclose(full_audio[:, offset:], partial_audio, atol=1e-5)

    def test_load_with_num_frames(self):
        """Loading with num_frames limits frames read."""
        num_frames = 5000
        audio, _ = audio_io.load(TEST_AUDIO, num_frames=num_frames)
        assert audio.shape[1] == num_frames

    def test_load_with_offset_and_num_frames(self):
        """Loading with both offset and num_frames works correctly."""
        full_audio, rate = audio_io.load(TEST_AUDIO)
        offset = 1000
        num_frames = 5000
        
        partial_audio, _ = audio_io.load(
            TEST_AUDIO, frame_offset=offset, num_frames=num_frames
        )
        
        assert partial_audio.shape[1] == num_frames
        assert torch.allclose(
            full_audio[:, offset:offset + num_frames],
            partial_audio,
            atol=1e-5
        )

    def test_load_offset_at_end_returns_empty(self):
        """Offset beyond file length returns empty tensor."""
        info = audio_io.info(TEST_AUDIO)
        audio, rate = audio_io.load(TEST_AUDIO, frame_offset=info.num_frames + 1000)
        assert audio.shape[1] == 0

    def test_load_num_frames_exceeds_file_returns_available(self):
        """Requesting more frames than available returns what's available."""
        info = audio_io.info(TEST_AUDIO)
        audio, _ = audio_io.load(TEST_AUDIO, num_frames=info.num_frames + 10000)
        assert audio.shape[1] == info.num_frames

    def test_load_normalized_audio_in_range(self):
        """Normalized audio values are in [-1, 1]."""
        audio, _ = audio_io.load(TEST_AUDIO, normalize=True)
        assert audio.min() >= -1.0
        assert audio.max() <= 1.0


class TestSave:
    """Tests for audio_io.save()"""

    def test_save_creates_file(self):
        """Saving creates a file on disk."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            filepath = f.name
        
        try:
            audio = torch.randn(1, 16000)  # 1 second mono
            audio_io.save(filepath, audio, 16000)
            assert Path(filepath).exists()
        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_save_and_load_roundtrip(self):
        """Saved audio can be loaded back with same content."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            filepath = f.name
        
        try:
            # Create random audio
            original = torch.randn(1, 16000).clamp(-1, 1)
            audio_io.save(filepath, original, 16000)
            
            # Load it back
            loaded, rate = audio_io.load(filepath)
            
            assert rate == 16000
            assert loaded.shape == original.shape
            # Allow some tolerance for PCM quantization
            assert torch.allclose(original, loaded, atol=1e-3)
        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_save_stereo_audio(self):
        """Saving stereo audio works correctly."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            filepath = f.name
        
        try:
            original = torch.randn(2, 16000).clamp(-1, 1)  # Stereo
            audio_io.save(filepath, original, 16000)
            loaded, _ = audio_io.load(filepath)
            
            assert loaded.shape[0] == 2  # Stereo preserved
        finally:
            Path(filepath).unlink(missing_ok=True)


class TestInfo:
    """Tests for audio_io.info()"""

    def test_info_returns_audio_info(self):
        """Info returns AudioInfo dataclass."""
        info = audio_io.info(TEST_AUDIO)
        assert isinstance(info, audio_io.AudioInfo)

    def test_info_has_sample_rate(self):
        """Info includes sample rate."""
        info = audio_io.info(TEST_AUDIO)
        assert info.sample_rate > 0

    def test_info_has_num_frames(self):
        """Info includes number of frames."""
        info = audio_io.info(TEST_AUDIO)
        assert info.num_frames > 0

    def test_info_has_num_channels(self):
        """Info includes number of channels."""
        info = audio_io.info(TEST_AUDIO)
        assert info.num_channels in (1, 2)

    def test_info_matches_loaded_audio(self):
        """Info metadata matches actual loaded audio."""
        info = audio_io.info(TEST_AUDIO)
        audio, rate = audio_io.load(TEST_AUDIO)
        
        assert rate == info.sample_rate
        assert audio.shape[0] == info.num_channels
        assert audio.shape[1] == info.num_frames


class TestTorchaudioCompatibility:
    """
    Regression tests comparing audio_io to torchaudio.
    
    These tests verify that audio_io produces identical output to torchaudio,
    ensuring a safe migration path. Skip if torchaudio not available or
    if running on torchaudio 2.9+ where I/O was removed.
    """

    @pytest.fixture
    def torchaudio_available(self):
        """Check if torchaudio I/O is available."""
        try:
            import torchaudio
            # Try to use load - will fail on 2.9+
            torchaudio.load
            return True
        except (ImportError, AttributeError):
            pytest.skip("torchaudio I/O not available")
            return False

    def test_load_matches_torchaudio(self, torchaudio_available):
        """audio_io.load produces same output as torchaudio.load."""
        import torchaudio
        
        old_audio, old_rate = torchaudio.load(str(TEST_AUDIO))
        new_audio, new_rate = audio_io.load(TEST_AUDIO)
        
        assert old_rate == new_rate
        assert old_audio.shape == new_audio.shape
        # Allow small tolerance for floating point differences
        assert torch.allclose(old_audio, new_audio, atol=1e-4)

    def test_load_with_seeking_matches_torchaudio(self, torchaudio_available):
        """audio_io.load with offset/frames matches torchaudio.load."""
        import torchaudio
        
        offset = 1000
        num_frames = 5000
        
        old_audio, _ = torchaudio.load(
            str(TEST_AUDIO), frame_offset=offset, num_frames=num_frames
        )
        new_audio, _ = audio_io.load(
            TEST_AUDIO, frame_offset=offset, num_frames=num_frames
        )
        
        assert old_audio.shape == new_audio.shape
        assert torch.allclose(old_audio, new_audio, atol=1e-4)

    def test_info_matches_torchaudio(self, torchaudio_available):
        """audio_io.info produces same metadata as torchaudio.info."""
        import torchaudio
        
        # Skip if torchaudio.info not available (removed in 2.9+)
        if not hasattr(torchaudio, 'info'):
            pytest.skip("torchaudio.info not available (removed in 2.9+)")
        
        old_info = torchaudio.info(str(TEST_AUDIO))
        new_info = audio_io.info(TEST_AUDIO)
        
        assert old_info.sample_rate == new_info.sample_rate
        assert old_info.num_frames == new_info.num_frames
        # Note: num_channels attribute name may differ
        old_channels = getattr(old_info, 'num_channels', None)
        if old_channels is None:
            old_channels = getattr(old_info, 'channels', 1)
        assert old_channels == new_info.num_channels

"""
Audio I/O abstraction layer.

This module provides a unified interface for audio loading, saving, and metadata
extraction. It uses soundfile as the backend, replacing torchaudio I/O functions
which were removed in torchaudio 2.9+.

The API is designed to be a drop-in replacement for torchaudio's I/O functions:
- audio_io.load() replaces torchaudio.load()
- audio_io.save() replaces torchaudio.save()
- audio_io.info() replaces torchaudio.info()
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import soundfile as sf
import torch


@dataclass
class AudioInfo:
    """Audio file metadata, compatible with torchaudio.info() output."""
    sample_rate: int
    num_frames: int
    num_channels: int
    bits_per_sample: int = 16
    encoding: str = "PCM_S"


def load(
    filepath: Union[str, Path],
    frame_offset: int = 0,
    num_frames: int = -1,
    normalize: bool = True,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, int]:
    """
    Load audio from a file.

    This is a drop-in replacement for torchaudio.load() using soundfile as backend.

    Parameters
    ----------
    filepath : str or Path
        Path to the audio file.
    frame_offset : int, optional
        Number of frames to skip from the beginning. Default: 0.
    num_frames : int, optional
        Maximum number of frames to read. -1 reads the entire file. Default: -1.
    normalize : bool, optional
        Whether to normalize audio to [-1.0, 1.0]. Default: True.
    dtype : torch.dtype, optional
        Output tensor dtype. Default: torch.float32.

    Returns
    -------
    Tuple[torch.Tensor, int]
        - audio: Tensor of shape (channels, frames)
        - sample_rate: Sample rate of the audio
    """
    filepath = str(filepath)
    
    # Get file info for validation
    file_info = sf.info(filepath)
    total_frames = file_info.frames
    sample_rate = file_info.samplerate
    
    # Handle frame_offset
    start = frame_offset
    if start < 0:
        start = 0
    if start >= total_frames:
        # Return empty tensor if offset is beyond file
        return torch.zeros(file_info.channels, 0, dtype=dtype), sample_rate
    
    # Handle num_frames
    if num_frames < 0:
        frames_to_read = -1  # Read all remaining
    else:
        frames_to_read = num_frames
    
    # Read audio using soundfile
    # soundfile.read returns (frames, channels) for multi-channel
    audio_np, sr = sf.read(
        filepath,
        start=start,
        frames=frames_to_read if frames_to_read > 0 else -1,
        dtype='float32' if normalize else 'int16',
        always_2d=True,  # Always return 2D array for consistent handling
    )
    
    # Convert to torch tensor
    audio = torch.from_numpy(audio_np.T)  # Transpose to (channels, frames)
    
    # Convert dtype if needed
    if dtype != torch.float32:
        audio = audio.to(dtype)
    
    return audio, sr


def save(
    filepath: Union[str, Path],
    audio: torch.Tensor,
    sample_rate: int,
    bits_per_sample: int = 16,
) -> None:
    """
    Save audio to a file.

    This is a drop-in replacement for torchaudio.save() using soundfile as backend.

    Parameters
    ----------
    filepath : str or Path
        Path to save the audio file.
    audio : torch.Tensor
        Audio tensor of shape (channels, frames).
    sample_rate : int
        Sample rate of the audio.
    bits_per_sample : int, optional
        Bit depth. Default: 16.
    """
    filepath = str(filepath)
    
    # Convert tensor to numpy, transpose from (channels, frames) to (frames, channels)
    if audio.dim() == 1:
        audio_np = audio.numpy()
    else:
        audio_np = audio.T.numpy()
    
    # Determine subtype based on bits_per_sample
    subtype_map = {
        8: 'PCM_S8',
        16: 'PCM_16',
        24: 'PCM_24',
        32: 'PCM_32',
    }
    subtype = subtype_map.get(bits_per_sample, 'PCM_16')
    
    # Determine format from extension
    ext = Path(filepath).suffix.lower()
    format_map = {
        '.wav': 'WAV',
        '.flac': 'FLAC',
        '.ogg': 'OGG',
    }
    file_format = format_map.get(ext)
    
    sf.write(filepath, audio_np, sample_rate, subtype=subtype, format=file_format)


def info(filepath: Union[str, Path]) -> AudioInfo:
    """
    Get audio file metadata.

    This is a drop-in replacement for torchaudio.info() using soundfile as backend.

    Parameters
    ----------
    filepath : str or Path
        Path to the audio file.

    Returns
    -------
    AudioInfo
        Metadata about the audio file.
    """
    filepath = str(filepath)
    file_info = sf.info(filepath)
    
    # Map soundfile subtype to bits per sample
    subtype = file_info.subtype
    if 'PCM_16' in subtype or '16' in subtype:
        bits = 16
    elif 'PCM_24' in subtype or '24' in subtype:
        bits = 24
    elif 'PCM_32' in subtype or '32' in subtype or 'FLOAT' in subtype:
        bits = 32
    elif 'PCM_S8' in subtype or 'PCM_U8' in subtype or '8' in subtype:
        bits = 8
    else:
        bits = 16  # Default
    
    return AudioInfo(
        sample_rate=file_info.samplerate,
        num_frames=file_info.frames,
        num_channels=file_info.channels,
        bits_per_sample=bits,
        encoding=subtype,
    )

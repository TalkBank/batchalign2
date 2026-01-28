"""
Diarization Pipeline Module

Provides engines for speaker diarization (who spoke when).
Uses lazy loading to avoid importing heavy dependencies until needed.
"""

def __getattr__(name):
    """Lazy loading for diarization engines to avoid circular imports and improve startup time"""
    if name == 'PyannoteEngine':
        from .pyannote import PyannoteEngine
        return PyannoteEngine
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = ['PyannoteEngine']

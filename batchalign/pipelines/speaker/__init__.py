"""
Speaker Pipeline Module

Provides engines for speaker recognition and embedding.
Uses lazy loading to avoid importing heavy dependencies until needed.
"""

def __getattr__(name):
    """Lazy loading for speaker engines to avoid circular imports and improve startup time"""
    if name == 'NemoSpeakerEngine':
        from .nemo_speaker import NemoSpeakerEngine
        return NemoSpeakerEngine
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = ['NemoSpeakerEngine']

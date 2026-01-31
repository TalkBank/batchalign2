"""
FA (Forced Alignment) Pipeline Module

Provides engines for aligning audio with transcript timing.
Uses lazy loading to avoid importing heavy dependencies until needed.
"""

def __getattr__(name):
    """Lazy loading for FA engines to avoid circular imports and improve startup time"""
    if name == 'WhisperFAEngine':
        from .whisper_fa import WhisperFAEngine
        return WhisperFAEngine
    elif name == 'Wave2VecFAEngine':
        from .wave2vec_fa import Wave2VecFAEngine
        return Wave2VecFAEngine
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = ['WhisperFAEngine', 'Wave2VecFAEngine']

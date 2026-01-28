"""
UTR (Utterance-level Transcription) Pipeline Module

Provides engines for utterance-level transcription analysis.
Uses lazy loading to avoid importing heavy dependencies until needed.
"""

def __getattr__(name):
    """Lazy loading for UTR engines to avoid circular imports and improve startup time"""
    if name == 'WhisperUTREngine':
        from .whisper_utr import WhisperUTREngine
        return WhisperUTREngine
    elif name == 'RevUTREngine':
        from .rev_utr import RevUTREngine
        return RevUTREngine
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = ['WhisperUTREngine', 'RevUTREngine']

"""
ASR (Automatic Speech Recognition) Pipeline Module

Provides engines for converting audio to text transcripts.
Uses lazy loading to avoid importing heavy dependencies until needed.
"""

def __getattr__(name):
    """Lazy loading for ASR engines to avoid circular imports and improve startup time"""
    if name == 'WhisperEngine':
        from .whisper import WhisperEngine
        return WhisperEngine
    elif name == 'RevEngine':
        from .rev import RevEngine
        return RevEngine
    elif name == 'WhisperXEngine':
        from .whisperx import WhisperXEngine
        return WhisperXEngine
    elif name == 'OAIWhisperEngine':
        from .oai_whisper import OAIWhisperEngine
        return OAIWhisperEngine
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = ['WhisperEngine', 'RevEngine', 'WhisperXEngine', 'OAIWhisperEngine']

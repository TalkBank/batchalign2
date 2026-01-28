"""
Cleanup Pipeline Module

Provides engines for cleaning up transcripts (retracing, disfluency handling).
Uses lazy loading to avoid importing heavy dependencies until needed.
"""

def __getattr__(name):
    """Lazy loading for cleanup engines to avoid circular imports and improve startup time"""
    if name == 'NgramRetraceEngine':
        from .retrace import NgramRetraceEngine
        return NgramRetraceEngine
    elif name == 'DisfluencyReplacementEngine':
        from .disfluencies import DisfluencyReplacementEngine
        return DisfluencyReplacementEngine
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = ['NgramRetraceEngine', 'DisfluencyReplacementEngine']

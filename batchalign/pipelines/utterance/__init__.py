"""
Utterance Pipeline Module

Provides engines for utterance-level language analysis.
Uses lazy loading to avoid importing heavy dependencies until needed.
"""

def __getattr__(name):
    """Lazy loading for utterance engines to avoid circular imports and improve startup time"""
    if name == 'StanzaUtteranceEngine':
        from .ud_utterance import StanzaUtteranceEngine
        return StanzaUtteranceEngine
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = ['StanzaUtteranceEngine']

"""
Morphosyntax Pipeline Module

Provides engines for morphological and syntactic analysis.
Uses lazy loading to avoid importing heavy dependencies until needed.
"""

def __getattr__(name):
    """Lazy loading for morphosyntax engines to avoid circular imports and improve startup time"""
    if name == 'StanzaEngine':
        from .ud import StanzaEngine
        return StanzaEngine
    elif name == 'CorefEngine':
        from .coref import CorefEngine
        return CorefEngine
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = ['StanzaEngine', 'CorefEngine']
